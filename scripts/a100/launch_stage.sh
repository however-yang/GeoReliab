#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
. "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
usage: launch_stage.sh <overlay> <commit> <p0|p1|p2a|p2|p3|p4|p5|p6> [device]

Launches a frozen GeoReliab real-MVE stage in GNU screen with explicit logs.
Every model stage requires an exact-commit explicit GPU receipt and runs one
sequential 0/1 shard. The script never writes below /home/smli.
EOF
}

[ $# -ge 3 ] || { usage >&2; exit 2; }
overlay=$1
commit=$(project_commit_arg "$2")
stage=$3
device=${4:-}
logical_device=cuda:0
gpu_index=
root=$(runtime_root "$overlay")
worktree="$(overlay_value "$overlay" runtime.worktrees)/$commit"
logs="$(overlay_value "$overlay" runtime.logs)/screens"
cache="$(overlay_value "$overlay" runtime.cache)"
python=$(orchestrator_python "$overlay")
control="$(overlay_value "$overlay" runtime.logs)/control"
gpu_lock="$control/georeliab-gpu-execution.lock"
mkdir_under_root "$root" "$logs" "$cache/tmp" "$control"
export_runtime_cache_env "$overlay"
enforce_storage_cap "$overlay"
[ -d "$worktree" ] || die "missing deployed worktree: $worktree"
assert_project_worktree_clean "$worktree" "$commit"

screen_prefix=$(overlay_value "$overlay" execution.screen_prefix)
name_base="${screen_prefix}-${stage}-${commit:0:12}"
py="$python -m georeliab_mve"

screen_launch() {
  local name=$1 logfile=$2 command=$3
  assert_project_worktree_clean "$worktree" "$commit"
  guard_under_root "$root" "$logfile"
  if screen -list | grep -q "[.]${name}[[:space:]]"; then die "screen already exists: $name"; fi
  screen -dmS "$name" -L -Logfile "$logfile" bash -lc "export TMPDIR='$TMPDIR' TMP='$TMP' TEMP='$TEMP' XDG_CACHE_HOME='$XDG_CACHE_HOME' HF_HOME='$HF_HOME' TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE' TORCH_HOME='$TORCH_HOME' CUDA_CACHE_PATH='$CUDA_CACHE_PATH' MPLCONFIGDIR='$MPLCONFIGDIR' PYTHONPYCACHEPREFIX='$PYTHONPYCACHEPREFIX' GEORELIAB_NO_HOME_WRITE=1; cd '$worktree' && $command"
  info "launched $name; log=$logfile"
}

assert_no_active_georeliab_execution() {
  local screen_state command_line cmdline pid
  screen_state=$(screen -list 2>/dev/null || true)
  if printf '%s\n' "$screen_state" | grep -q "[.]${screen_prefix}-"; then
    die "active GeoReliab screen/controller blocks single-GPU launch: $screen_state"
  fi
  for cmdline in /proc/[0-9]*/cmdline; do
    [ -r "$cmdline" ] || continue
    pid=${cmdline#/proc/}
    pid=${pid%/cmdline}
    if [ "$pid" = "$$" ] || [ "$pid" = "$PPID" ]; then
      continue
    fi
    command_line=$(tr '\0' ' ' < "$cmdline" 2>/dev/null || true)
    case "$command_line" in
      *run-georeliab*|*preflight-real*|*launch_stage.sh*|*georeliab-chain*)
        die "active GeoReliab process blocks single-GPU launch: pid=$pid $command_line"
        ;;
    esac
  done
}

gpu_screen_launch() {
  local name=$1 logfile=$2 command=$3
  assert_project_worktree_clean "$worktree" "$commit"
  guard_under_root "$root" "$logfile"
  guard_under_root "$root" "$gpu_lock"
  if screen -list | grep -q "[.]${name}[[:space:]]"; then die "screen already exists: $name"; fi
  assert_no_active_georeliab_execution
  if [ -e "$gpu_lock" ]; then die "single-GPU execution lock already exists: $gpu_lock"; fi
  if ! mkdir "$gpu_lock"; then die "cannot acquire single-GPU execution lock: $gpu_lock"; fi
  if ! screen -dmS "$name" -L -Logfile "$logfile" bash -lc "trap 'rmdir -- \"$gpu_lock\"' EXIT; export CUDA_VISIBLE_DEVICES='$gpu_index' GEORELIAB_PHYSICAL_GPU_DEVICE='$device' GEORELIAB_LOGICAL_GPU_DEVICE='$logical_device' TMPDIR='$TMPDIR' TMP='$TMP' TEMP='$TEMP' XDG_CACHE_HOME='$XDG_CACHE_HOME' HF_HOME='$HF_HOME' TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE' TORCH_HOME='$TORCH_HOME' CUDA_CACHE_PATH='$CUDA_CACHE_PATH' MPLCONFIGDIR='$MPLCONFIGDIR' PYTHONPYCACHEPREFIX='$PYTHONPYCACHEPREFIX' GEORELIAB_NO_HOME_WRITE=1; cd '$worktree' && $command"; then
    rmdir "$gpu_lock"
    die "failed to launch GPU screen: $name"
  fi
  info "launched $name on physical $device as logical $logical_device with exclusive single-GPU lock; log=$logfile"
}


require_p0_complete() {
  (
    cd "$worktree"
    "$python" - "$root" <<'PY'
import json, sys
from pathlib import Path
from georeliab_mve.runner import p0_completion_status

decision = p0_completion_status(Path(sys.argv[1]))
if decision.get('status') != 'PASS':
    raise SystemExit('P1_LOCKED: ' + json.dumps(decision, sort_keys=True))
PY
  )
}

require_p1_complete() {
  (
    cd "$worktree"
    "$python" - "$root" "$overlay" <<'PY'
import json, sys
from pathlib import Path
from georeliab_mve.runner import p1_completion_status

decision = p1_completion_status(
    Path(sys.argv[1]),
    config_path=Path(sys.argv[2]),
)
if decision.get('status') != 'PASS':
    raise SystemExit('P2_LOCKED: ' + json.dumps(decision, sort_keys=True))
PY
  )
}

require_superseded_archive() {
  (
    cd "$worktree"
    "$python" - "$root" <<'PY'
import json, sys
from pathlib import Path
from georeliab_mve.execution_governance import superseded_archive_status

decision = superseded_archive_status(Path(sys.argv[1]))
if decision.get('status') != 'PASS':
    raise SystemExit('P1_LOCKED_SUPERSEDED_ARCHIVE: ' + json.dumps(decision, sort_keys=True))
PY
  )
}

require_p2a_complete() {
  (
    cd "$worktree"
    "$python" - "$root" <<'PY'
import json, sys
from pathlib import Path
from georeliab_mve.execution_governance import p2a_completion_status

decision = p2a_completion_status(Path(sys.argv[1]))
if decision.get('status') != 'PASS':
    raise SystemExit('P2_LOCKED_P2A: ' + json.dumps(decision, sort_keys=True))
PY
  )
}

require_gpu_selection() {
  [ -n "$device" ] || die "GPU_SELECTION_REQUIRED: choose cuda:0 or cuda:1 for commit $commit"
  case "$device" in
    cuda:0|cuda:1) ;;
    *) die "GPU_SELECTION_REQUIRED: device must be cuda:0 or cuda:1" ;;
  esac
  (
    cd "$worktree"
    "$python" -m georeliab_mve validate-gpu-selection \
      --root "$root" \
      --project-commit "$commit" \
      --device "$device"
  )
  gpu_index=${device#cuda:}
  if ! nvidia-smi --query-gpu=index --format=csv,noheader,nounits \
      | grep -Fxq "$gpu_index"; then
    die "selected physical GPU is unavailable: $device"
  fi
}


require_native_gate_pass() {
  (
    cd "$worktree"
    "$python" - "$overlay" "$root" <<'PY'
import json, sys
from pathlib import Path
from georeliab_mve.runner import RunnerContext, load_native_phenomenon_gate, zero_update_schedule_allowed

overlay, root = sys.argv[1:]
context = RunnerContext(root=Path(root), output_root=Path(root), config_path=Path(overlay), device='cpu')
gate = load_native_phenomenon_gate(context)
allowed = zero_update_schedule_allowed(gate)
if allowed.get('status') != 'PASS':
    raise SystemExit('SHORT_CIRCUIT_P5: ' + json.dumps({'native_gate': gate, 'decision': allowed}, sort_keys=True))
PY
  )
}

require_stage_complete() {
  local runner_stage=$1 expected=$2
  (
    cd "$worktree"
    "$python" - "$root" "$runner_stage" "$expected" <<'PY'
import json, sys
from pathlib import Path
from georeliab_mve.runner import full_schedule, stage_progress_counts

root, stage, expected = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
counts = stage_progress_counts(root, stage, full_schedule(root, stage))
if counts.get('scheduled') != expected or counts.get('completed') != expected or counts.get('missing') != 0:
    raise SystemExit(f'PRIOR_STAGE_INCOMPLETE: {stage}: ' + json.dumps(counts, sort_keys=True))
PY
  )
}

enforce_stage_gpu_budget() {
  local runner_stage=$1
  (
    cd "$worktree"
    "$python" - "$overlay" "$root" "$runner_stage" <<'PY'
import json, sys
from pathlib import Path
from georeliab_mve import toml_compat as tomllib
from georeliab_mve.runner import (
    RunnerContext, build_zero_update_schedule, estimate_stage_budget,
    full_schedule, load_native_phenomenon_gate,
)

overlay, root, stage = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
context = RunnerContext(root=root, output_root=root, config_path=overlay, device='cpu')
if stage == 'zero-update':
    items = build_zero_update_schedule(root, load_native_phenomenon_gate(context), model='all')
else:
    items = full_schedule(root, stage)
override = None
if stage == 'preflight':
    payload = tomllib.loads(overlay.read_text(encoding='utf-8'))
    reservation = float(payload['execution'].get('preflight_gpu_hour_reservation', 0.0))
    if reservation <= 0.0:
        budget = {'status': 'BLOCKED_RESOURCE_BUDGET', 'stage': stage, 'reason_code': 'BUDGET_ESTIMATE_UNAVAILABLE'}
    else:
        budget = estimate_stage_budget(context, stage, items, {'next_stage_gpu_hours': reservation, 'next_stage_bytes': 0})
else:
    budget = estimate_stage_budget(context, stage, items)
print(json.dumps(budget, sort_keys=True))
if budget.get('status') != 'OK':
    raise SystemExit('BLOCKED_RESOURCE_BUDGET: ' + json.dumps(budget, sort_keys=True))
PY
  )
}

case "$stage" in
  p1) require_p0_complete; require_superseded_archive; require_gpu_selection; enforce_stage_gpu_budget preflight ;;
  p2a) require_p1_complete; require_gpu_selection; enforce_stage_gpu_budget smoke ;;
  p2) require_p1_complete; require_p2a_complete; require_gpu_selection; enforce_stage_gpu_budget smoke ;;
  p3) require_stage_complete smoke 200; require_gpu_selection; enforce_stage_gpu_budget test ;;
  p4) require_stage_complete test 400 ;;
  p5) require_native_gate_pass; require_gpu_selection; enforce_stage_gpu_budget zero-update ;;
esac

case "$stage" in
  p0)
    cmd="set -euo pipefail; ./scripts/a100/verify_prereqs.sh '$overlay'; $py prepare-georeliab --operation download --data-root '$root' --state '$root/artifacts/p0_download.json' --overlay '$overlay'; $py prepare-georeliab --operation verify --data-root '$root' --state '$root/artifacts/p0_verify.json' --overlay '$overlay'; $py prepare-georeliab --operation index --data-root '$root' --state '$root/artifacts/p0_index.json' --overlay '$overlay'; $py prepare-georeliab --operation manifests --data-root '$root' --state '$root/artifacts/p0_manifests.json' --overlay '$overlay'; $py prepare-georeliab --operation prepared --data-root '$root' --state '$root/artifacts/p0_prepared.json' --overlay '$overlay'; $py prepare-georeliab --operation calibration --data-root '$root' --state '$root/artifacts/p0_calibration.json' --overlay '$overlay'; $py prepare-georeliab --operation rendering --stage smoke --data-root '$root' --state '$root/artifacts/p0_render_smoke.json' --overlay '$overlay'; $py prepare-georeliab --operation rendering --stage test --data-root '$root' --state '$root/artifacts/p0_render_test.json' --overlay '$overlay'; $py prepare-georeliab --operation sanity --data-root '$root' --state '$root/artifacts/p0_sanity.json' --overlay '$overlay'"
    screen_launch "$name_base" "$logs/$name_base.log" "$cmd"
    ;;
  p1)
    gpu_screen_launch "$name_base" "$logs/$name_base.log" "$py preflight-real --config '$overlay' --output-root '$root' --device '$logical_device' --summary-json '$root/artifacts/p1_preflight.json'"
    ;;
  p2a)
    p2a_selection="$root/artifacts/p2a_selection_manifest.json"
    cmd="set -euo pipefail; $py prepare-p2a --root '$root' --storage-before '$root/artifacts/storage_before.json' --storage-plan '$root/artifacts/storage_plan.json' --output '$p2a_selection'; $py run-georeliab --stage smoke --model all --device '$logical_device' --shard 0/1 --config '$overlay' --output-root '$root' --selection-manifest '$p2a_selection'; $py validate-p2a --root '$root' --selection '$p2a_selection' --output '$root/artifacts/p2a_completion.json'"
    gpu_screen_launch "$name_base" "$logs/$name_base.log" "$cmd"
    ;;
  p2)
    gpu_screen_launch "$name_base" "$logs/$name_base.log" "$py run-georeliab --stage smoke --model all --device '$logical_device' --shard 0/1 --config '$overlay' --output-root '$root'"
    ;;
  p3)
    gpu_screen_launch "$name_base" "$logs/$name_base.log" "$py run-georeliab --stage test --model all --device '$logical_device' --shard 0/1 --config '$overlay' --output-root '$root'"
    ;;
  p4)
    audit="$root/stage/test/native_phenomenon_audit.json"
    cmd="set -euo pipefail; $py audit-georeliab --stage-evidence '$root/stage/test/stage_evidence_p3.json' --output '$audit'; $python -c \"import json; from pathlib import Path; from georeliab_mve.runner import RunnerContext, write_native_phenomenon_gate_from_audit_output; root=Path('$root'); overlay=Path('$overlay'); audit=json.loads(Path('$audit').read_text(encoding='utf-8')); print(write_native_phenomenon_gate_from_audit_output(RunnerContext(root=root, output_root=root, config_path=overlay, device='cpu'), audit))\""
    screen_launch "$name_base" "$logs/$name_base.log" "$cmd"
    ;;
  p5)
    gpu_screen_launch "$name_base" "$logs/$name_base.log" "$py run-georeliab --stage zero-update --model all --device '$logical_device' --shard 0/1 --config '$overlay' --output-root '$root'"
    ;;
  p6)
    native_gate="$root/stage/test/native_phenomenon_gate.json"
    gate_status=$("$python" - "$native_gate" <<'PY'
import json, sys
from pathlib import Path

status = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')).get('status')
if status not in {'PASS', 'FAIL'}:
    raise SystemExit(f'P6 requires terminal P4 gate, got {status}')
print(status)
PY
)
    if [ "$gate_status" = PASS ] && [ ! -f "$root/stage/zero-update/terminal_failure.json" ]; then
      gate_stage="$root/stage/test/stage_evidence.json"
      gate_output="$root/evidence/georeliab_final_gate.json"
    else
      gate_stage="$root/stage/test/stage_evidence_p3.json"
      gate_output="$root/evidence/georeliab_p3_gate.json"
    fi
    [ -f "$gate_stage" ] || die "P6 canonical stage evidence missing: $gate_stage"
    cmd="set -euo pipefail; $py audit-georeliab --stage-evidence '$gate_stage' --output '$gate_output'; ./scripts/a100/status.sh '$overlay' '$commit' > '$root/artifacts/p6_status_live.json'; ./scripts/a100/finalize_p6.sh '$overlay' '$commit'"
    screen_launch "$name_base" "$logs/$name_base.log" "$cmd"
    ;;
  *) die "unknown stage: $stage" ;;
esac
