#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo 'worktree, runtime root, overlay, and frozen materialization are required' >&2
  exit 2
fi

worktree=$1
runtime_root=$2
overlay=$3
frozen_materialization=$4

cd "$worktree"
tooling_commit=$(git rev-parse HEAD)
output_dir="$runtime_root/artifacts/v4-overlay-resource-resolution/$tooling_commit"
receipt="$output_dir/resource-resolution-receipt.json"

python -m georeliab_mve v4-resolve-overlay-resources \
  --worktree "$worktree" \
  --runtime-root "$runtime_root" \
  --overlay "$overlay" \
  --frozen-materialization "$frozen_materialization" \
  --output-dir "$output_dir"

python -m georeliab_mve \
  v4-validate-overlay-resource-resolution "$receipt"
