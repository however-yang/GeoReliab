#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
. "${SCRIPT_DIR}/common.sh"

overlay=${1:-$DEFAULT_OVERLAY}
commit=$(project_commit_arg "${2:-}")
root=$(runtime_root "$overlay")
export_runtime_cache_env "$overlay"
bare=$(overlay_value "$overlay" runtime.git_bare)
worktrees=$(overlay_value "$overlay" runtime.worktrees)
target="$worktrees/$commit"
artifacts=$(overlay_value "$overlay" runtime.artifacts)

case "$overlay" in
  "$GEORELIAB_PROJECT_ROOT"/*) overlay_rel=${overlay#"$GEORELIAB_PROJECT_ROOT"/} ;;
  /*) die "overlay must be inside deploying project worktree: $overlay" ;;
  *) overlay_rel=$overlay ;;
esac
case "$overlay_rel" in
  ""|../*|*/../*|./*|*/./*) die "overlay path must be normalized within the project: $overlay_rel" ;;
esac

mkdir_under_root "$root" "$(dirname "$bare")" "$worktrees" "$artifacts/deploy"
enforce_storage_cap "$overlay"

if [ ! -d "$bare" ]; then git init --bare "$bare"; fi
git --git-dir="$bare" rev-parse --is-bare-repository >/dev/null || die "invalid bare repository: $bare"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git cat-file -e "$commit^{commit}" || die "commit not present in current repository: $commit"
  git push "$bare" "$commit:refs/heads/deploy/$commit"
fi

git --git-dir="$bare" cat-file -e "$commit^{commit}" || die "commit not present in bare repository after deploy: $commit"

if [ -e "$target" ]; then
  git -C "$target" rev-parse --git-dir >/dev/null 2>&1 || die "conflicting non-worktree path exists: $target"
  actual=$(git -C "$target" rev-parse HEAD)
  [ "$actual" = "$commit" ] || die "existing worktree commit mismatch: $actual != $commit"
  dirty=$(git -C "$target" status --porcelain --untracked-files=all)
  [ -z "$dirty" ] || die "deployed worktree has tracked or untracked changes: $target"
else
  git --git-dir="$bare" worktree add --detach "$target" "$commit"
fi

target_overlay="$target/$overlay_rel"
[ -r "$target_overlay" ] || die "target worktree overlay is not readable: $target_overlay"
cmp -s "$overlay" "$target_overlay" || die "source and target worktree overlays differ: $overlay != $target_overlay"

manifest="$artifacts/deploy/$commit.json"
"$(orchestrator_python "$target_overlay")" - "$overlay_rel" "$target_overlay" "$commit" "$bare" "$target" <<'PY' | write_immutable_file "$root" "$manifest"
import json, subprocess, sys
overlay_name, overlay_path, commit, bare, worktree = sys.argv[1:]
def sha(path): return subprocess.check_output(['sha256sum', path], text=True).split()[0]
payload = {
    'schema_version': 'georeliab-deploy-v1',
    'overlay': overlay_name,
    'overlay_sha256': sha(overlay_path),
    'project_commit': commit,
    'bare_repository': bare,
    'worktree': worktree,
    'worktree_head': subprocess.check_output(['git', '-C', worktree, 'rev-parse', 'HEAD'], text=True).strip(),
    'dirty_policy': 'reject tracked and untracked changes',
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
sha256sum "$manifest" | write_immutable_file "$root" "$manifest.sha256"
info "deployed detached worktree: $target"
