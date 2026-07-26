#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
. "${SCRIPT_DIR}/common.sh"

overlay=${1:-$DEFAULT_OVERLAY}
commit=${2:-}
root=$(runtime_root "$overlay")
enforce_storage_cap "$overlay"

"$(orchestrator_python "$overlay")" - "$overlay" "$root" "$commit" <<'PY'
import json, subprocess, sys
from pathlib import Path

overlay, root, commit = sys.argv[1:]
root_path = Path(root)

def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f'UNAVAILABLE: {exc}'

def load_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None

runtime_by_key = {}
peak_memory_mb = 0.0
raw_schedule_counts = []
reason_codes = []
for path in sorted(root_path.glob('stage/**/*.json')) if (root_path / 'stage').exists() else []:
    payload = load_json(path)
    if not isinstance(payload, dict):
        continue
    if 'schedule_counts' in payload:
        raw_schedule_counts.append({'path': str(path), 'schedule_counts': payload['schedule_counts'], 'status': payload.get('status')})
    if 'reason_code' in payload or 'reason_codes' in payload:
        reason_codes.append({'path': str(path), 'reason_code': payload.get('reason_code'), 'reason_codes': payload.get('reason_codes')})
    identity = str(payload.get('artifact_sha256') or payload.get('prediction_sha256') or payload.get('sample_key') or path)
    for key in ('runtime_seconds', 'runtime_sec'):
        if isinstance(payload.get(key), (int, float)):
            runtime_by_key[identity] = max(runtime_by_key.get(identity, 0.0), float(payload[key]))
    for key in ('peak_memory_mb', 'peak_memory_mib'):
        if isinstance(payload.get(key), (int, float)):
            peak_memory_mb = max(peak_memory_mb, float(payload[key]))

canonical_counts = {}
try:
    from georeliab_mve.runner import (
        RunnerContext, build_zero_update_schedule, full_schedule,
        load_native_phenomenon_gate, stage_progress_counts,
    )
    ctx = RunnerContext(root=root_path, output_root=root_path, config_path=Path(overlay), device='cuda:0')
    for stage in ('smoke', 'test'):
        try:
            items = full_schedule(root_path, stage)
            canonical_counts[stage] = stage_progress_counts(root_path, stage, items)
        except Exception as exc:
            canonical_counts[stage] = {'status': 'UNAVAILABLE', 'reason': str(exc)}
    try:
        native_gate = load_native_phenomenon_gate(ctx)
        if native_gate.get('status') == 'PASS':
            items = build_zero_update_schedule(root_path, native_gate, model='all')
            canonical_counts['zero-update'] = stage_progress_counts(root_path, 'zero-update', items)
        else:
            canonical_counts['zero-update'] = {'status': 'NOT_APPLICABLE', 'native_gate_status': native_gate.get('status'), 'reason_code': native_gate.get('reason_code'), 'reason_codes': native_gate.get('reason_codes')}
    except Exception as exc:
        canonical_counts['zero-update'] = {'status': 'UNAVAILABLE', 'reason': str(exc)}
except Exception as exc:
    canonical_counts = {'status': 'UNAVAILABLE', 'reason': str(exc)}

def count_field(name):
    total = 0
    any_value = False
    for value in canonical_counts.values() if isinstance(canonical_counts, dict) else []:
        if isinstance(value, dict) and isinstance(value.get(name), int):
            total += int(value[name]); any_value = True
    return total if any_value else None

try:
    output_bytes = int(subprocess.check_output(['du', '-sb', root], text=True).split()[0])
except Exception:
    output_bytes = None

payload = {
    'schema_version': 'georeliab-a100-status-v1', 'overlay': overlay, 'runtime_root': root,
    'project_commit': commit or None, 'screens': run(['screen', '-list']),
    'gpu': run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader']),
    'observed_gpu_hours': sum(runtime_by_key.values()) / 3600.0, 'peak_memory_mb': peak_memory_mb,
    'output_bytes': output_bytes, 'missing_count_tail_sum': count_field('missing'),
    'invalid_count_tail_sum': count_field('invalid'), 'canonical_schedule_counts': canonical_counts,
    'schedule_counts_tail': raw_schedule_counts[-20:], 'reason_codes_tail': reason_codes[-20:],
    'diagnostic_screen_logs': sorted(str(p) for p in (root_path / 'logs' / 'screens').glob('*.log'))[-20:] if (root_path / 'logs' / 'screens').exists() else [],
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
