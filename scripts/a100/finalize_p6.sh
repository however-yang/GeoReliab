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

env_lock="$artifacts/environment_locks/a100_environment_lock.json"
no_home="$artifacts/no_home_write_marker.json"
native_gate="$root/stage/test/native_phenomenon_gate.json"
p3_stage="$root/stage/test/stage_evidence_p3.json"
stage_freeze="$root/stage/test/stage_freeze.json"
for required in "$status_live" "$env_lock" "$no_home" "$native_gate" "$p3_stage" "$stage_freeze" "$root/manifests/split_view_manifest.json" "$root/manifests/frozen_materialization.json" "$root/manifests/corruption_calibration.json" "$root/manifests/corruption_calibration_qa.json" "$root/manifests/test_render_lock.json" "$root/manifests/test_render_index.json" "$root/prepared/render_inputs_test.json" "$root/prepared/tartanair_p000_pairs.json" "$root/evidence/tartanair_native_fog_sanity.json"; do
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
if [ "$gate_status" = "PASS" ]; then
  [ -f "$root/stage/test/stage_evidence.json" ] || die 'P4 PASS requires final stage_evidence.json after P5'
  [ -f "$root/evidence/georeliab_final_gate.json" ] || die 'P4 PASS requires final GeoReliab gate output'
  gate_path="$root/evidence/georeliab_final_gate.json"
  final_stage="$root/stage/test/stage_evidence.json"
else
  [ -f "$root/evidence/georeliab_p3_gate.json" ] || die 'P4 FAIL requires P3 GeoReliab gate output'
  gate_path="$root/evidence/georeliab_p3_gate.json"
  final_stage=''
fi

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
project_tree = subprocess.check_output(['git', '-C', worktree, 'rev-parse', f'{commit}^{{tree}}'], text=True).strip()
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
core = {key: status.get(key) for key in ('schema_version', 'project_commit', 'observed_gpu_hours', 'peak_memory_mb', 'missing_count_tail_sum', 'invalid_count_tail_sum', 'canonical_schedule_counts', 'schedule_counts_tail', 'reason_codes_tail', 'output_bytes')}
print(json.dumps(core, indent=2, sort_keys=True))
PY

"$(orchestrator_python "$overlay")" - "$overlay" "$root" "$commit" "$worktree" "$status_core" "$gate_path" "$final_stage" <<'PY' | write_immutable_file "$root" "$bundle_path"
import json, subprocess, sys
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

overlay, root, commit, worktree, status_path, gate_path, final_stage = sys.argv[1:]
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
project_tree = subprocess.check_output(['git', '-C', worktree, 'rev-parse', f'{commit}^{{tree}}'], text=True).strip()
provenance = {
    'environment_lock': req(root / 'artifacts' / 'environment_locks' / 'a100_environment_lock.json'), 'no_home_write_marker': req(root / 'artifacts' / 'no_home_write_marker.json'), 'stage_freeze': req(root / 'stage' / 'test' / 'stage_freeze.json'),
    'split_view_manifest': req(root / 'manifests' / 'split_view_manifest.json'), 'frozen_materialization': req(root / 'manifests' / 'frozen_materialization.json'), 'corruption_calibration': req(root / 'manifests' / 'corruption_calibration.json'),
    'corruption_calibration_qa': req(root / 'manifests' / 'corruption_calibration_qa.json'), 'test_render_lock': req(root / 'manifests' / 'test_render_lock.json'), 'test_render_index': req(root / 'manifests' / 'test_render_index.json'),
    'test_prepared_input': req(root / 'prepared' / 'render_inputs_test.json'), 'tartanair_prepared_input': req(root / 'prepared' / 'tartanair_p000_pairs.json'), 'tartanair_native_fog_sanity': req(root / 'evidence' / 'tartanair_native_fog_sanity.json'),
}
bundle = {
    'schema_version': 'georeliab-p6-evidence-bundle-v1', 'project_commit': commit, 'project_tree': project_tree, 'overlay': {'path': overlay, 'sha256': sha(overlay_path)}, 'required_provenance': provenance,
    'stage_evidence_p3': req(root / 'stage' / 'test' / 'stage_evidence_p3.json'), 'stage_evidence_final': req(Path(final_stage)) if final_stage else None,
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
