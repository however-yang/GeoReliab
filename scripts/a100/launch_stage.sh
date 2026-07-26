#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
. "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
usage: launch_stage.sh <overlay> <commit> <p0|p1|p2|p3|p4|p5|p6> [device]

Launches a frozen GeoReliab real-MVE stage in GNU screen with explicit logs.
P3 and P5 launch two configured A100 shards. The script refuses writes outside
/srv/private/smli/GeoReliab and never writes below /home/smli.
EOF
}

[ $# -ge 3 ] || { usage >&2; exit 2; }
overlay=$1
commit=$(project_commit_arg "$2")
stage=$3
device=${4:-$(overlay_value "$overlay" execution.default_device)}
root=$(runtime_root "$overlay")
worktree="$(overlay_value "$overlay" runtime.worktrees)/$commit"
logs="$(overlay_value "$overlay" runtime.logs)/screens"
cache="$(overlay_value "$overlay" runtime.cache)"
python=$(orchestrator_python "$overlay")
mkdir_under_root "$root" "$logs" "$cache/tmp"
export_runtime_cache_env "$overlay"
enforce_storage_cap "$overlay"
[ -d "$worktree" ] || die "missing deployed worktree: $worktree"

screen_prefix=$(overlay_value "$overlay" execution.screen_prefix)
name_base="${screen_prefix}-${stage}-${commit:0:12}"
py="$python -m georeliab_mve"

screen_launch() {
  local name=$1 logfile=$2 command=$3
  guard_under_root "$root" "$logfile"
  if screen -list | grep -q "[.]${name}[[:space:]]"; then die "screen already exists: $name"; fi
  screen -dmS "$name" -L -Logfile "$logfile" bash -lc "export TMPDIR='$TMPDIR' TMP='$TMP' TEMP='$TEMP' XDG_CACHE_HOME='$XDG_CACHE_HOME' HF_HOME='$HF_HOME' TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE' TORCH_HOME='$TORCH_HOME' CUDA_CACHE_PATH='$CUDA_CACHE_PATH' MPLCONFIGDIR='$MPLCONFIGDIR' PYTHONPYCACHEPREFIX='$PYTHONPYCACHEPREFIX' GEORELIAB_NO_HOME_WRITE=1; cd '$worktree' && $command"
  info "launched $name; log=$logfile"
}

case "$stage" in
  p0)
    cmd="set -euo pipefail; ./scripts/a100/verify_prereqs.sh '$overlay'; $py prepare-georeliab --operation download --data-root '$root' --state '$root/artifacts/p0_download.json' --overlay '$overlay'; $py prepare-georeliab --operation verify --data-root '$root' --state '$root/artifacts/p0_verify.json' --overlay '$overlay'; $py prepare-georeliab --operation index --data-root '$root' --state '$root/artifacts/p0_index.json' --overlay '$overlay'; $py prepare-georeliab --operation manifests --data-root '$root' --state '$root/artifacts/p0_manifests.json' --overlay '$overlay'; $py prepare-georeliab --operation prepared --data-root '$root' --state '$root/artifacts/p0_prepared.json' --overlay '$overlay'; $py prepare-georeliab --operation calibration --data-root '$root' --state '$root/artifacts/p0_calibration.json' --overlay '$overlay'; $py prepare-georeliab --operation rendering --stage smoke --data-root '$root' --state '$root/artifacts/p0_render_smoke.json' --overlay '$overlay'; $py prepare-georeliab --operation rendering --stage test --data-root '$root' --state '$root/artifacts/p0_render_test.json' --overlay '$overlay'; $py prepare-georeliab --operation sanity --data-root '$root' --state '$root/artifacts/p0_sanity.json' --overlay '$overlay'"
    screen_launch "$name_base" "$logs/$name_base.log" "$cmd"
    ;;
  p1)
    screen_launch "$name_base" "$logs/$name_base.log" "$py preflight-real --config '$overlay' --output-root '$root' --device '$device'"
    ;;
  p2)
    screen_launch "${name_base}-shard0" "$logs/${name_base}-shard0.log" "$py run-georeliab --stage smoke --model all --device cuda:0 --shard 0/2 --config '$overlay' --output-root '$root'"
    screen_launch "${name_base}-shard1" "$logs/${name_base}-shard1.log" "$py run-georeliab --stage smoke --model all --device cuda:1 --shard 1/2 --config '$overlay' --output-root '$root'"
    ;;
  p3)
    screen_launch "${name_base}-shard0" "$logs/${name_base}-shard0.log" "$py run-georeliab --stage test --model all --device cuda:0 --shard 0/2 --config '$overlay' --output-root '$root'"
    screen_launch "${name_base}-shard1" "$logs/${name_base}-shard1.log" "$py run-georeliab --stage test --model all --device cuda:1 --shard 1/2 --config '$overlay' --output-root '$root'"
    ;;
  p4)
    audit="$root/stage/test/native_phenomenon_audit.json"
    cmd="set -euo pipefail; $py audit-georeliab --stage-evidence '$root/stage/test/stage_evidence_p3.json' --output '$audit'; $python -c \"import json; from pathlib import Path; from georeliab_mve.runner import RunnerContext, write_native_phenomenon_gate_from_audit_output; root=Path('$root'); overlay=Path('$overlay'); audit=json.loads(Path('$audit').read_text(encoding='utf-8')); print(write_native_phenomenon_gate_from_audit_output(RunnerContext(root=root, output_root=root, config_path=overlay, device='$device'), audit))\""
    screen_launch "$name_base" "$logs/$name_base.log" "$cmd"
    ;;
  p5)
    screen_launch "${name_base}-shard0" "$logs/${name_base}-shard0.log" "$py run-georeliab --stage zero-update --model all --device cuda:0 --shard 0/2 --config '$overlay' --output-root '$root'"
    screen_launch "${name_base}-shard1" "$logs/${name_base}-shard1.log" "$py run-georeliab --stage zero-update --model all --device cuda:1 --shard 1/2 --config '$overlay' --output-root '$root'"
    ;;
  p6)
    cmd="set -euo pipefail; if [ -f '$root/stage/test/stage_evidence.json' ]; then $py audit-georeliab --stage-evidence '$root/stage/test/stage_evidence.json' --output '$root/evidence/georeliab_final_gate.json'; else $py audit-georeliab --stage-evidence '$root/stage/test/stage_evidence_p3.json' --output '$root/evidence/georeliab_p3_gate.json'; fi; ./scripts/a100/status.sh '$overlay' '$commit' > '$root/artifacts/p6_status_live.json'; ./scripts/a100/finalize_p6.sh '$overlay' '$commit'"
    screen_launch "$name_base" "$logs/$name_base.log" "$cmd"
    ;;
  *) die "unknown stage: $stage" ;;
esac
