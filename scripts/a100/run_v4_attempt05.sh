#!/usr/bin/env bash
set -euo pipefail

# GeoReliab v4 Attempt-05 execution wrapper. This consumes the immutable
# Attempt-04 authorization, but never performs GPU selection or fallback.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
. "${SCRIPT_DIR}/common.sh"

if [[ $# -ne 7 && $# -ne 9 ]]; then
  echo "usage: $0 AUTHORIZATION DTU_ROOT RECTIFIED_CLOSURE_MANIFEST FOG_ROOT INPUT_CLOSURE_DIR EXPECTED_TOOLING_COMMIT EXPECTED_TOOLING_TREE [RECOVERY_FROM_COMMIT RECOVERY_FROM_TREE]" >&2
  exit 2
fi

authorization="$1"
dtu_root="$2"
rectified_closure_manifest="$3"
fog_root="$4"
input_closure_dir="$5"
expected_tooling_commit="$6"
expected_tooling_tree="$7"
recovery_from_commit="${8:-}"
recovery_from_tree="${9:-}"
overlay=${GEORELIAB_OVERLAY:-${GEORELIAB_PROJECT_ROOT}/${DEFAULT_OVERLAY}}

cd "${GEORELIAB_PROJECT_ROOT}"
require_cmd git
require_cmd flock
require_cmd nvidia-smi
[ -r "$overlay" ] || die "missing frozen overlay: $overlay"
export_runtime_cache_env "$overlay"
python=$(orchestrator_python "$overlay")

actual_commit="$(git rev-parse HEAD)"
actual_tree="$(git rev-parse 'HEAD^{tree}')"
sha40_re='^[0-9a-f]{40}$'
if [[ ! "$expected_tooling_commit" =~ $sha40_re || ! "$expected_tooling_tree" =~ $sha40_re ]]; then
  echo "V4_MVE_FAILED_WITH_REASON=V4_ATTEMPT05_TOOLING_REVISION_REQUIRED" >&2
  exit 2
fi
if [[ "$actual_commit" != "$expected_tooling_commit" ]]; then
  echo "V4_MVE_FAILED_WITH_REASON=V4_ATTEMPT05_TOOLING_COMMIT_MISMATCH" >&2
  exit 2
fi
if [[ "$actual_tree" != "$expected_tooling_tree" ]]; then
  echo "V4_MVE_FAILED_WITH_REASON=V4_ATTEMPT05_TOOLING_TREE_MISMATCH" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "V4_MVE_FAILED_WITH_REASON=V4_ATTEMPT05_TOOLING_TREE_DIRTY" >&2
  exit 2
fi

lock_path="$(runtime_root "$overlay")/logs/control/v4-mve-attempt-05.execution.lock"
mkdir_under_root "$(runtime_root "$overlay")" "$(dirname "$lock_path")"
exec 9>"$lock_path"
if ! flock -n 9; then
  echo "V4_MVE_FAILED_WITH_REASON=V4_ATTEMPT05_EXECUTION_LOCK_HELD" >&2
  exit 2
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export GEORELIAB_PHYSICAL_GPU_DEVICE=cuda:0
export GEORELIAB_PHYSICAL_GPU_UUID=GPU-6ae218e6-3d51-b748-e308-1f0509e87886
export GEORELIAB_PHYSICAL_GPU_PCI_BUS_ID=00000000:4D:00.0
resolved_gpu="$(nvidia-smi --query-gpu=uuid,pci.bus_id --format=csv,noheader,nounits -i 0)"
expected_gpu="${GEORELIAB_PHYSICAL_GPU_UUID}, ${GEORELIAB_PHYSICAL_GPU_PCI_BUS_ID}"
if [[ "$resolved_gpu" != "$expected_gpu" ]]; then
  echo "V4_MVE_FAILED_WITH_REASON=V4_ATTEMPT05_GPU_IDENTITY_MISMATCH" >&2
  exit 2
fi

if [[ -n "$recovery_from_commit" || -n "$recovery_from_tree" ]]; then
  if [[ ! "$recovery_from_commit" =~ $sha40_re || ! "$recovery_from_tree" =~ $sha40_re ]]; then
    echo "V4_MVE_FAILED_WITH_REASON=V4_ATTEMPT05_RECOVERY_REVISION_INVALID" >&2
    exit 2
  fi
  recovery_artifact="$(runtime_root "$overlay")/artifacts/v4-mve-attempt-05/v4-attempt05-recovery-revision.json"
  if [[ -e "$recovery_artifact" ]]; then
    "$python" -m georeliab_mve v4-attempt05-authorize-continuation \
      --authorization "$authorization" \
      --from-tooling-commit "$recovery_from_commit" \
      --from-tooling-tree "$recovery_from_tree" \
      --to-tooling-commit "$expected_tooling_commit" \
      --to-tooling-tree "$expected_tooling_tree"
  else
    "$python" -m georeliab_mve v4-attempt05-authorize-recovery \
      --authorization "$authorization" \
      --from-tooling-commit "$recovery_from_commit" \
      --from-tooling-tree "$recovery_from_tree" \
      --to-tooling-commit "$expected_tooling_commit" \
      --to-tooling-tree "$expected_tooling_tree"
  fi
fi

if [[ -e "$input_closure_dir" || -e "${input_closure_dir}.partial" ]]; then
  if [[ ! -d "$input_closure_dir" || -e "${input_closure_dir}.partial" ]]; then
    echo "V4_MVE_FAILED_WITH_REASON=V4_ATTEMPT05_INPUT_CLOSURE_RESUME_BOUNDARY_INVALID" >&2
    exit 2
  fi
  info "reusing immutable validated input closure: $input_closure_dir"
else
  "$python" -m georeliab_mve v4-attempt05-prepare-inputs \
    --authorization "$authorization" \
    --dtu-root "$dtu_root" \
    --rectified-closure-manifest "$rectified_closure_manifest" \
    --fog-root "$fog_root" \
    --output-dir "$input_closure_dir"
fi

"$python" -m georeliab_mve v4-attempt05-preflight \
  --authorization "$authorization" \
  --states "$input_closure_dir/v4-model-independent-states.json" \
  --schedule "$input_closure_dir/v4-scientific-schedule-400.json" \
  --split-assignment "$input_closure_dir/v4-split-assignment.json" \
  --calibration-schedule "$input_closure_dir/v4-calibration-l3-schedule-40.json" \
  --input-closure "$input_closure_dir/v4-attempt05-input-closure.json" \
  --tooling-commit "$expected_tooling_commit" \
  --tooling-tree "$expected_tooling_tree" \
  --resume

"$python" -m georeliab_mve v4-attempt05-run \
  --authorization "$authorization" \
  --input-closure-dir "$input_closure_dir" \
  --tooling-commit "$expected_tooling_commit" \
  --tooling-tree "$expected_tooling_tree" \
  --resume

"$python" -m georeliab_mve v4-attempt05-status \
  --authorization "$authorization" \
  --schedule "$input_closure_dir/v4-scientific-schedule-400.json"
