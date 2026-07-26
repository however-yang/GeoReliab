#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
. "${SCRIPT_DIR}/common.sh"

[ $# -ge 2 ] || die 'usage: finalize_p6.sh <overlay> <commit>'
overlay=$1
commit=$(project_commit_arg "$2")
root=$(runtime_root "$overlay")
artifacts=$(overlay_value "$overlay" runtime.artifacts)
worktree="$(overlay_value "$overlay" runtime.worktrees)/$commit"
bundle_path="$artifacts/p6_evidence_bundle.json"
table_path="$artifacts/p6_one_page_decision.md"
table_sha_path="$artifacts/p6_one_page_decision.md.sha256"
status_live="$artifacts/p6_status_live.json"
status_core="$artifacts/p6_status_core.json"
mkdir_under_root "$root" "$artifacts"
[ -d "$worktree" ] || die "missing deployed worktree: $worktree"
assert_project_worktree_clean "$worktree" "$commit"

env_lock="$artifacts/environment_locks/a100_environment_lock.json"
no_home="$artifacts/no_home_write_marker.json"
deploy_manifest="$artifacts/deploy/$commit.json"
native_gate="$root/stage/test/native_phenomenon_gate.json"
p3_stage="$root/stage/test/stage_evidence_p3.json"
stage_freeze="$root/stage/test/stage_freeze.json"
for required in "$status_live" "$env_lock" "$no_home" "$deploy_manifest" "$native_gate" "$p3_stage" "$stage_freeze" "$root/manifests/split_view_manifest.json" "$root/manifests/frozen_materialization.json" "$root/manifests/corruption_calibration.json" "$root/manifests/corruption_calibration_qa.json" "$root/manifests/test_render_lock.json" "$root/manifests/test_render_index.json" "$root/prepared/render_inputs_test.json" "$root/prepared/tartanair_p000_pairs.json" "$root/evidence/tartanair_native_fog_sanity.json"; do
  [ -f "$required" ] || die "missing required P6 provenance: $required"
done

gate_status=$("$(orchestrator_python "$overlay")" - "$native_gate" <<'PY'
import json, sys
status = json.loads(open(sys.argv[1], encoding='utf-8').read()).get('status')
if status not in {'PASS', 'FAIL'}:
    raise SystemExit(f'native gate is not terminal PASS/FAIL: {status}')
print(status)
PY
)
p5_terminal_failure=''
if [ "$gate_status" = "PASS" ]; then
  if [ -f "$root/stage/zero-update/terminal_failure.json" ]; then
    p5_terminal_failure="$root/stage/zero-update/terminal_failure.json"
    [ -f "$root/evidence/georeliab_p3_gate.json" ] || die 'P5 terminal failure requires P3 GeoReliab gate output'
    gate_path="$root/evidence/georeliab_p3_gate.json"
    final_stage=''
  else
    [ -f "$root/stage/test/stage_evidence.json" ] || die 'P4 PASS requires final stage_evidence.json after P5'
    [ -f "$root/evidence/georeliab_final_gate.json" ] || die 'P4 PASS requires final GeoReliab gate output'
    gate_path="$root/evidence/georeliab_final_gate.json"
    final_stage="$root/stage/test/stage_evidence.json"
  fi
else
  [ -f "$root/evidence/georeliab_p3_gate.json" ] || die 'P4 FAIL requires P3 GeoReliab gate output'
  gate_path="$root/evidence/georeliab_p3_gate.json"
  final_stage=''
fi

"$(orchestrator_python "$overlay")" - "$status_live" "$native_gate" "$gate_path" "$overlay" "$commit" "$p5_terminal_failure" <<'PY'
import json, math, sys
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

status_path, native_path, gate_path, overlay_path, commit, terminal_path = sys.argv[1:]
status = json.loads(Path(status_path).read_text(encoding='utf-8'))
native = json.loads(Path(native_path).read_text(encoding='utf-8'))
gate_payload = json.loads(Path(gate_path).read_text(encoding='utf-8'))
gate = gate_payload.get('georeliab_gate', gate_payload)
limits = tomllib.loads(Path(overlay_path).read_text(encoding='utf-8'))['execution']
if status.get('project_commit') != commit:
    raise SystemExit('P6 status project commit mismatch')
if gate.get('status') not in {'PASS', 'FAIL'}:
    raise SystemExit(f"GeoReliab gate is not terminal PASS/FAIL: {gate.get('status')}")
canonical = status.get('canonical_schedule_counts')
if not isinstance(canonical, dict) or canonical.get('status') == 'UNAVAILABLE':
    raise SystemExit('P6 canonical schedule counts are UNAVAILABLE')

def require_complete(name, expected):
    counts = canonical.get(name)
    required = ('scheduled', 'completed', 'missing', 'invalid')
    if not isinstance(counts, dict) or any(not isinstance(counts.get(key), int) for key in required):
        raise SystemExit(f'P6 {name} schedule counts are unavailable or malformed: {counts}')
    if counts['scheduled'] != expected or counts['completed'] != expected or counts['missing'] != 0:
        raise SystemExit(f'P6 {name} schedule is incomplete: {counts}')
    return counts

require_complete('smoke', 200)
require_complete('test', 400)
if native.get('status') == 'PASS':
    if terminal_path:
        terminal = json.loads(Path(terminal_path).read_text(encoding='utf-8'))
        if terminal.get('reason_code') != 'P5_INVALID_SUBSET_PREDICTION':
            raise SystemExit('unsupported P5 terminal failure reason')
        counts = canonical.get('zero-update')
        required = ('scheduled', 'completed', 'missing', 'invalid')
        if not isinstance(counts, dict) or any(not isinstance(counts.get(key), int) for key in required):
            raise SystemExit(f'P5 terminal schedule counts are unavailable: {counts}')
        if counts['scheduled'] != 480 or counts['completed'] + counts['missing'] != 480:
            raise SystemExit(f'P5 terminal schedule accounting is invalid: {counts}')
        if gate.get('status') != 'FAIL':
            raise SystemExit('P5 terminal failure requires terminal GeoReliab FAIL evidence')
    else:
        require_complete('zero-update', 480)
elif native.get('status') == 'FAIL':
    zero = canonical.get('zero-update')
    if not isinstance(zero, dict) or zero.get('status') != 'NOT_APPLICABLE' or zero.get('native_gate_status') != 'FAIL':
        raise SystemExit(f'P4 FAIL must short-circuit P5: {zero}')
    if gate.get('status') != 'FAIL':
        raise SystemExit('native-confidence FAIL requires terminal GeoReliab FAIL evidence')
else:
    raise SystemExit(f"native gate is not terminal: {native.get('status')}")
for key in ('missing_count_tail_sum', 'invalid_count_tail_sum'):
    if not isinstance(status.get(key), int):
        raise SystemExit(f'P6 {key} is unavailable')
if status.get('budget_evidence_status') != 'OK' or status.get('ledger_parse_errors') != []:
    raise SystemExit(f"BLOCKED_RESOURCE_BUDGET: ledger budget evidence is unavailable: {status.get('ledger_parse_errors')}")
gpu_hours = status.get('observed_gpu_hours')
if not isinstance(gpu_hours, (int, float)) or not math.isfinite(gpu_hours) or gpu_hours < 0 or gpu_hours > float(limits['gpu_hour_limit']):
    raise SystemExit(f'BLOCKED_RESOURCE_BUDGET: invalid/over-limit observed GPU-hours: {gpu_hours}')
output_bytes = status.get('output_bytes')
if not isinstance(output_bytes, int) or output_bytes < 0 or output_bytes > int(limits['max_storage_bytes']):
    raise SystemExit(f'BLOCKED_RESOURCE_BUDGET: invalid/over-limit output bytes: {output_bytes}')
PY

if [ -f "$bundle_path" ] && [ -f "$table_path" ] && [ -f "$table_sha_path" ]; then
  "$(orchestrator_python "$overlay")" - "$bundle_path" "$table_path" "$table_sha_path" "$commit" "$worktree" <<'PY'
import json, subprocess, sys
from pathlib import Path
bundle_path, table_path, table_sha_path, commit, worktree = sys.argv[1:]
bundle = json.loads(Path(bundle_path).read_text(encoding='utf-8'))
def sha(path): return subprocess.check_output(['sha256sum', str(path)], text=True).split()[0]
def walk(value):
    if isinstance(value, dict):
        if set(('path', 'sha256')).issubset(value):
            path = Path(value['path'])
            if not path.exists(): raise SystemExit(f'missing stored artifact: {path}')
            if sha(path) != value['sha256']: raise SystemExit(f'stored artifact digest mismatch: {path}')
        for child in value.values(): walk(child)
    elif isinstance(value, list):
        for child in value: walk(child)
if bundle.get('project_commit') != commit:
    raise SystemExit('existing P6 bundle does not match current commit/native gate: commit mismatch')
project_tree = subprocess.check_output(['git', '-C', worktree, 'rev-parse', 'HEAD^{tree}'], text=True).strip()
if bundle.get('project_tree') != project_tree:
    raise SystemExit('existing P6 bundle does not match current commit/native gate: project tree mismatch')
walk(bundle)
expected_table = Path(table_sha_path).read_text(encoding='utf-8').split()[0]
if sha(Path(table_path)) != expected_table:
    raise SystemExit('P6 decision table digest mismatch')
PY
  info "P6 immutable bundle/table already exist and all stored digests match"
  exit 0
fi

"$(orchestrator_python "$overlay")" - "$status_live" <<'PY' | write_immutable_file "$root" "$status_core"
import json, sys
from pathlib import Path
status = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
core = {key: status.get(key) for key in ('schema_version', 'project_commit', 'observed_gpu_hours', 'peak_memory_mb', 'budget_evidence_status', 'ledger_parse_errors', 'missing_count_tail_sum', 'invalid_count_tail_sum', 'canonical_schedule_counts', 'schedule_counts_tail', 'reason_codes_tail', 'output_bytes')}
print(json.dumps(core, indent=2, sort_keys=True))
PY

"$(orchestrator_python "$overlay")" - "$overlay" "$root" "$commit" "$worktree" "$status_core" "$gate_path" "$final_stage" "$p5_terminal_failure" <<'PY' | write_immutable_file "$root" "$bundle_path"
import json, subprocess, sys
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

overlay, root, commit, worktree, status_path, gate_path, final_stage, terminal_failure = sys.argv[1:]
root = Path(root); overlay_path = Path(overlay); status_path = Path(status_path); gate_path = Path(gate_path)
def sha(path: Path) -> str: return subprocess.check_output(['sha256sum', str(path)], text=True).split()[0]
def load(path: Path): return json.loads(path.read_text(encoding='utf-8'))
def req(path: Path):
    if not path.exists(): raise SystemExit(f'missing required P6 file: {path}')
    return {'path': str(path), 'sha256': sha(path)}
payload = tomllib.loads(overlay_path.read_text(encoding='utf-8'))
status = load(status_path); gate = load(gate_path); native = load(root / 'stage' / 'test' / 'native_phenomenon_gate.json')
worker_logs = []
summary_dir = root / 'stage' / '_worker_summaries'
if summary_dir.exists():
    worker_logs = [{'path': str(path), 'sha256': sha(path)} for path in sorted(summary_dir.glob('*.std*.log'))]
screen_logs = sorted(str(p) for p in (root / 'logs' / 'screens').glob('*.log'))[-40:] if (root / 'logs' / 'screens').exists() else []
project_tree = subprocess.check_output(['git', '-C', worktree, 'rev-parse', 'HEAD^{tree}'], text=True).strip()
provenance = {
    'deployment': req(root / 'artifacts' / 'deploy' / f'{commit}.json'), 'environment_lock': req(root / 'artifacts' / 'environment_locks' / 'a100_environment_lock.json'), 'no_home_write_marker': req(root / 'artifacts' / 'no_home_write_marker.json'), 'stage_freeze': req(root / 'stage' / 'test' / 'stage_freeze.json'),
    'split_view_manifest': req(root / 'manifests' / 'split_view_manifest.json'), 'frozen_materialization': req(root / 'manifests' / 'frozen_materialization.json'), 'corruption_calibration': req(root / 'manifests' / 'corruption_calibration.json'),
    'corruption_calibration_qa': req(root / 'manifests' / 'corruption_calibration_qa.json'), 'test_render_lock': req(root / 'manifests' / 'test_render_lock.json'), 'test_render_index': req(root / 'manifests' / 'test_render_index.json'),
    'test_prepared_input': req(root / 'prepared' / 'render_inputs_test.json'), 'tartanair_prepared_input': req(root / 'prepared' / 'tartanair_p000_pairs.json'), 'tartanair_native_fog_sanity': req(root / 'evidence' / 'tartanair_native_fog_sanity.json'),
}
bundle = {
    'schema_version': 'georeliab-p6-evidence-bundle-v1', 'project_commit': commit, 'project_tree': project_tree, 'overlay': {'path': overlay, 'sha256': sha(overlay_path)}, 'required_provenance': provenance,
    'stage_evidence_p3': req(root / 'stage' / 'test' / 'stage_evidence_p3.json'), 'stage_evidence_final': req(Path(final_stage)) if final_stage else None, 'p5_terminal_failure': req(Path(terminal_failure)) if terminal_failure else None,
    'native_gate': {'path': str(root / 'stage' / 'test' / 'native_phenomenon_gate.json'), 'sha256': sha(root / 'stage' / 'test' / 'native_phenomenon_gate.json'), 'payload': native},
    'georeliab_gate': {'path': str(gate_path), 'sha256': sha(gate_path), 'payload': gate}, 'status_core': {'path': str(status_path), 'sha256': sha(status_path), 'payload': status},
    'official_resource_hashes': {k: v for k, v in payload['resources'].items() if k.endswith('_sha256') or k.endswith('_etag') or k.endswith('_commit') or k.endswith('_bytes')},
    'runtime_provenance': payload['runtime'], 'execution_limits': payload['execution'], 'observed_gpu_hours': status.get('observed_gpu_hours'), 'peak_memory_mb': status.get('peak_memory_mb'),
    'missing_count_tail_sum': status.get('missing_count_tail_sum'), 'invalid_count_tail_sum': status.get('invalid_count_tail_sum'), 'output_bytes': status.get('output_bytes'),
    'worker_log_hashes': worker_logs, 'diagnostic_screen_logs': screen_logs, 'global_selection_note': 'GeoReliab PASS remains GEORELIAB_PASS_PENDING_GEOMETRY until Geometry reaches terminal PASS/FAIL.',
}
print(json.dumps(bundle, indent=2, sort_keys=True))
PY

"$(orchestrator_python "$overlay")" - "$bundle_path" <<'PY' | write_immutable_file "$root" "$table_path"
import json, sys
from pathlib import Path
bundle = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
native = bundle['native_gate']['payload']; gate = bundle['georeliab_gate']['payload'].get('georeliab_gate', {}); status = bundle['status_core']['payload']
lines = ['# GeoReliab P6 One-Page Decision Table', '', f"- Project commit: `{bundle['project_commit']}`", f"- Project tree: `{bundle['project_tree']}`", f"- Native phenomenon gate: `{native.get('status')}` `{native.get('reason_codes')}`", f"- GeoReliab gate: `{gate.get('status')}` `{gate.get('reason_codes')}`", f"- Observed GPU-hours: `{bundle.get('observed_gpu_hours')}`", f"- Peak memory MB: `{bundle.get('peak_memory_mb')}`", f"- Missing / invalid tail sums: `{bundle.get('missing_count_tail_sum')}` / `{bundle.get('invalid_count_tail_sum')}`", f"- Output bytes: `{bundle.get('output_bytes')}`", '- Global selection: `BLOCKED_PENDING_GEOMETRY` until Geometry terminal evidence exists.', '', '## Latest schedule counts']
for item in status.get('schedule_counts_tail', [])[-12:]: lines.append(f"- `{item.get('path')}`: `{item.get('status')}` `{item.get('schedule_counts')}`")
lines += ['', '## Latest reason codes']
for item in status.get('reason_codes_tail', [])[-12:]: lines.append(f"- `{item.get('path')}`: `{item.get('reason_code') or item.get('reason_codes')}`")
print('\n'.join(lines))
PY
sha256sum "$table_path" | write_immutable_file "$root" "$table_sha_path"
info "P6 evidence bundle: $bundle_path"
info "P6 decision table: $table_path"
