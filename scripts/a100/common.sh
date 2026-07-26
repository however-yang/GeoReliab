#!/usr/bin/env bash
set -euo pipefail

readonly DEFAULT_OVERLAY="configs/a100_real_mve_overlay.toml"
readonly DEFAULT_ROOT="/srv/private/smli/GeoReliab"
readonly GEORELIAB_PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
info() { printf '[georeliab] %s\n' "$*"; }

python_bin() {
  if command -v python3 >/dev/null 2>&1; then printf '%s\n' python3
  elif command -v python >/dev/null 2>&1; then printf '%s\n' python
  else die 'python3 or python is required'; fi
}

shell_toml_value() {
  local overlay=$1 section_key=$2 section key
  section=${section_key%%.*}
  key=${section_key#*.}
  awk -v section="$section" -v key="$key" '
    $0 ~ "^[[:space:]]*\\[" section "\\][[:space:]]*$" { in_section=1; next }
    $0 ~ "^[[:space:]]*\\[" { in_section=0 }
    in_section && $0 ~ "^[[:space:]]*" key "[[:space:]]*=" {
      value=$0
      sub("^[^=]*=[[:space:]]*", "", value)
      sub(/[[:space:]]*(#.*)?$/, "", value)
      gsub(/^'\''|'\''$/, "", value)
      gsub(/^\"|\"$/, "", value)
      print value
      exit
    }
  ' "$overlay"
}

validate_typing_extensions_runtime() {
  local overlay=$1 typing_site typing_site_raw typing_file expected_sha expected_version actual_sha frozen_site
  frozen_site='/home/smli/miniforge3/pkgs/typing_extensions-4.15.0-pyhcf101f3_0/site-packages'
  typing_site=$(shell_toml_value "$overlay" runtime.typing_extensions_site)
  expected_sha=$(shell_toml_value "$overlay" resources.typing_extensions_sha256)
  expected_version=$(shell_toml_value "$overlay" resources.typing_extensions_version)
  [ -n "$typing_site" ] || die 'runtime.typing_extensions_site is required'
  [ -n "$expected_sha" ] || die 'resources.typing_extensions_sha256 is required'
  [ "$expected_version" = '4.15.0' ] || die "typing_extensions frozen version mismatch: $expected_version"
  typing_site_raw=${typing_site%/}
  [ "$typing_site_raw" = "$frozen_site" ] || die "typing_extensions site must equal frozen package cache: $typing_site"
  typing_site=$(CDPATH= cd -- "$typing_site" 2>/dev/null && pwd -P) || die "cannot resolve typing_extensions site: $typing_site"
  [ "$typing_site" = "$frozen_site" ] || die "typing_extensions site must resolve to frozen package cache: $typing_site"
  typing_file="$typing_site/typing_extensions.py"
  [ -r "$typing_file" ] || die "missing frozen typing_extensions.py: $typing_file"
  actual_sha=$(sha256sum "$typing_file" | awk '{print $1}')
  [ "$actual_sha" = "$expected_sha" ] || die "typing_extensions.py sha256 mismatch: $actual_sha != $expected_sha"
  printf '%s\n' "$typing_site"
}

overlay_value() {
  local overlay=$1 dotted=$2 typing_site
  typing_site=$(validate_typing_extensions_runtime "$overlay")
  PYTHONNOUSERSITE=1 PYTHONPATH="$typing_site:$GEORELIAB_PROJECT_ROOT/georeliab_mve" \
    "$(python_bin)" - "$overlay" "$dotted" <<'PY'
import sys
from pathlib import Path
import toml_compat as tomllib
payload = tomllib.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
value = payload
for part in sys.argv[2].split('.'):
    value = value[part]
if isinstance(value, list):
    print('\n'.join(str(item) for item in value))
elif isinstance(value, bool):
    print('true' if value else 'false')
else:
    print(value)
PY
}

runtime_root() { shell_toml_value "$1" runtime.root; }

orchestrator_python() {
  local overlay=$1 env_root candidate
  env_root=$(overlay_value "$overlay" runtime.vggt_env)
  for candidate in "$env_root/bin/python" "$env_root/Scripts/python.exe"; do
    [ -x "$candidate" ] && { printf '%s\n' "$candidate"; return; }
  done
  die "frozen VGGT environment Python is required: $env_root"
}

guard_under_root() {
  local root=$1 target=$2
  case "$target" in
    "$root"|"$root"/*) ;;
    *) die "refusing writable path outside runtime root: $target" ;;
  esac
  case "$target" in
    /home|/home/*) die "refusing writable path below /home: $target" ;;
  esac
}

mkdir_under_root() {
  local root=$1 path
  shift
  for path in "$@"; do guard_under_root "$root" "$path"; mkdir -p -- "$path"; done
}

export_runtime_cache_env() {
  local overlay=$1 root cache typing_site
  typing_site=$(validate_typing_extensions_runtime "$overlay")
  root=$(runtime_root "$overlay")
  cache=$(overlay_value "$overlay" runtime.cache)
  mkdir_under_root "$root" "$cache/tmp" "$cache/xdg" "$cache/hf" "$cache/hf/hub" "$cache/transformers" "$cache/torch" "$cache/torch_extensions" "$cache/cuda" "$cache/mpl" "$cache/pycache" "$cache/triton" "$cache/inductor" "$cache/cupy" "$cache/home"
  export TMPDIR="$cache/tmp"
  export TMP="$cache/tmp"
  export TEMP="$cache/tmp"
  export HOME="$cache/home"
  export XDG_CACHE_HOME="$cache/xdg"
  export HF_HOME="$cache/hf"
  export TRANSFORMERS_CACHE="$cache/transformers"
  export HF_HUB_CACHE="$cache/hf/hub"
  export TORCH_HOME="$cache/torch"
  export TORCH_EXTENSIONS_DIR="$cache/torch_extensions"
  export CUDA_CACHE_PATH="$cache/cuda"
  export TRITON_CACHE_DIR="$cache/triton"
  export TORCHINDUCTOR_CACHE_DIR="$cache/inductor"
  export CUPY_CACHE_DIR="$cache/cupy"
  export MPLCONFIGDIR="$cache/mpl"
  export PYTHONPYCACHEPREFIX="$cache/pycache"
  export PYTHONNOUSERSITE=1
  export PYTHONPATH="$typing_site:$GEORELIAB_PROJECT_ROOT"
  export GEORELIAB_NO_HOME_WRITE=1
}

require_cmd() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }

sha256_check() {
  local path=$1 expected=$2 actual
  [ -r "$path" ] || die "missing readable file: $path"
  actual=$(sha256sum "$path" | awk '{print $1}')
  [ "$actual" = "$expected" ] || die "sha256 mismatch for $path: $actual != $expected"
}

assert_git_commit_clean() {
  local path=$1 expected=$2 actual dirty
  git -C "$path" rev-parse --git-dir >/dev/null 2>&1 || die "not a git repository: $path"
  actual=$(git -C "$path" rev-parse HEAD)
  [ "$actual" = "$expected" ] || die "commit mismatch for $path: $actual != $expected"
  git -C "$path" diff --quiet HEAD -- || die "tracked worktree has unstaged changes: $path"
  git -C "$path" diff --cached --quiet || die "tracked worktree has staged changes: $path"
}

assert_project_worktree_clean() {
  local path=$1 expected=$2 actual dirty
  git -C "$path" rev-parse --git-dir >/dev/null 2>&1 || die "not a project git worktree: $path"
  actual=$(git -C "$path" rev-parse HEAD)
  [ "$actual" = "$expected" ] || die "project worktree commit mismatch: $actual != $expected"
  git -C "$path" diff --quiet HEAD -- || die "project worktree has unstaged changes: $path"
  git -C "$path" diff --cached --quiet || die "project worktree has staged changes: $path"
  dirty=$(git -C "$path" status --porcelain --untracked-files=all)
  [ -z "$dirty" ] || die "project worktree has untracked files: $path"
}

project_commit_arg() {
  local commit=${1:-}
  [ -n "$commit" ] || die 'project commit is required'
  case "$commit" in *[!0-9a-f]*) die "commit must be a full lowercase hexadecimal sha: $commit" ;; esac
  [ ${#commit} -eq 40 ] || die "commit must be 40 hex characters: $commit"
  printf '%s\n' "$commit"
}

write_immutable_file() {
  local root=$1 target=$2 tmp
  guard_under_root "$root" "$target"
  if [ -e "$target" ]; then
    cmp -s - "$target" || die "immutable artifact mismatch: $target"
  else
    tmp="${target}.partial.$$"
    cat >"$tmp"
    mv -- "$tmp" "$target"
  fi
}

enforce_storage_cap() {
  local overlay=$1 root max_bytes used
  root=$(runtime_root "$overlay")
  max_bytes=$(overlay_value "$overlay" execution.max_storage_bytes)
  used=$(du -sb "$root" 2>/dev/null | awk '{print $1}')
  [ -n "${used:-}" ] || used=0
  [ "$used" -le "$max_bytes" ] || die "runtime storage exceeds cap: $used > $max_bytes"
}
