#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 9 ]]; then
  echo 'nine path arguments are required' >&2
  exit 2
fi

worktree=$1
runtime_root=$2
rectified_root=$3
closure_root=$4
overlay=$5
run_root=$6
artifact_root=$7
gpu_ledger=$8
final_evidence=$9
attempt_root="$runtime_root/authorization-attempts/attempt-04"
resource_receipt="$attempt_root/v4-resource-revalidation.json"
hardware_snapshot="$attempt_root/v4-hardware-snapshot.json"
gpu_receipt="$attempt_root/v4-gpu-selection-receipt.json"
authorization="$attempt_root/v4-execution-authorization.json"

cd "$worktree"

# The resource command is deliberately first. With set -e, a failed resource
# gate proves that no nvidia-smi command from this script can be reached.
python -m georeliab_mve v4-attempt04-revalidate-resources \
  --worktree "$worktree" \
  --runtime-root "$runtime_root" \
  --rectified-root "$rectified_root" \
  --closure-root "$closure_root" \
  --overlay "$overlay" \
  --output "$resource_receipt"

python -m georeliab_mve v4-attempt04-gpu-preflight \
  --worktree "$worktree" \
  --resource-receipt "$resource_receipt" \
  --output "$hardware_snapshot"

python -m georeliab_mve v4-attempt04-create-execution-authorization \
  --worktree "$worktree" \
  --runtime-root "$runtime_root" \
  --resource-receipt "$resource_receipt" \
  --hardware-snapshot "$hardware_snapshot" \
  --receipt "$gpu_receipt" \
  --authorization "$authorization" \
  --run-root "$run_root" \
  --artifact-root "$artifact_root" \
  --gpu-ledger "$gpu_ledger" \
  --final-evidence-path "$final_evidence"

python -m georeliab_mve \
  v4-attempt04-validate-execution-authorization "$authorization"
