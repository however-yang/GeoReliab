#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
. "${SCRIPT_DIR}/common.sh"

overlay=${1:-$DEFAULT_OVERLAY}
root=$(runtime_root "$overlay")
export_runtime_cache_env "$overlay"

for cmd in bash git sha256sum awk sed df screen nvidia-smi du; do require_cmd "$cmd"; done
python=$(orchestrator_python "$overlay")
[ "$root" = "$DEFAULT_ROOT" ] || die "unexpected runtime root: $root"

mkdir_under_root "$root" \
  "$(overlay_value "$overlay" runtime.data)" "$(overlay_value "$overlay" runtime.cache)" \
  "$(overlay_value "$overlay" runtime.logs)" "$(overlay_value "$overlay" runtime.artifacts)" \
  "$(overlay_value "$overlay" runtime.worktrees)" "$(overlay_value "$overlay" runtime.stage)" \
  "$(overlay_value "$overlay" runtime.prepared)" "$(overlay_value "$overlay" runtime.rendered)" \
  "$(overlay_value "$overlay" runtime.manifests)" "$(overlay_value "$overlay" runtime.evidence)"

enforce_storage_cap "$overlay"

for key in runtime.vggt_source runtime.mast3r_source runtime.dust3r_source runtime.croco_source runtime.vggt_env runtime.mast3r_env resources.vggt_checkpoint resources.mast3r_checkpoint resources.mast3r_config; do
  path=$(overlay_value "$overlay" "$key")
  case "$path" in /home/smli/*) ;; *) die "reuse path must remain under /home/smli: $key=$path" ;; esac
  [ -r "$path" ] || die "reuse path is not readable: $key=$path"
done

assert_git_commit_clean "$(overlay_value "$overlay" runtime.vggt_source)" "$(overlay_value "$overlay" resources.vggt_source_commit)"
assert_git_commit_clean "$(overlay_value "$overlay" runtime.mast3r_source)" "$(overlay_value "$overlay" resources.mast3r_source_commit)"
assert_git_commit_clean "$(overlay_value "$overlay" runtime.dust3r_source)" "$(overlay_value "$overlay" resources.dust3r_source_commit)"
assert_git_commit_clean "$(overlay_value "$overlay" runtime.croco_source)" "$(overlay_value "$overlay" resources.croco_source_commit)"

sha256_check "$(overlay_value "$overlay" resources.vggt_checkpoint)" "$(overlay_value "$overlay" resources.vggt_checkpoint_sha256)"
sha256_check "$(overlay_value "$overlay" resources.mast3r_checkpoint)" "$(overlay_value "$overlay" resources.mast3r_checkpoint_sha256)"
sha256_check "$(overlay_value "$overlay" resources.mast3r_config)" "$(overlay_value "$overlay" resources.mast3r_config_sha256)"
typing_site=$(overlay_value "$overlay" runtime.typing_extensions_site)
typing_version=$(overlay_value "$overlay" resources.typing_extensions_version)
typing_sha=$(overlay_value "$overlay" resources.typing_extensions_sha256)
sha256_check "$typing_site/typing_extensions.py" "$typing_sha"

"$(overlay_value "$overlay" runtime.vggt_env)/bin/python" -I -B - "$typing_site" "$typing_sha" "$typing_version" "$(overlay_value "$overlay" runtime.vggt_python)" "$(overlay_value "$overlay" runtime.vggt_torch)" <<'PY'
import hashlib, platform, sys
from pathlib import Path
site = Path(sys.argv[1])
expected_sha, expected_version = sys.argv[2], sys.argv[3]
expected_python, expected_torch = sys.argv[4], sys.argv[5]
sys.path.insert(0, str(site))
import typing_extensions
from importlib import metadata
actual_file = Path(typing_extensions.__file__).resolve()
expected_file = (site / "typing_extensions.py").resolve()
assert actual_file == expected_file, f"typing_extensions import escaped frozen site: {actual_file}"
assert hashlib.sha256(actual_file.read_bytes()).hexdigest() == expected_sha
assert metadata.version("typing_extensions") == expected_version
import torch
assert platform.python_version() == expected_python
assert torch.__version__ == expected_torch
PY
"$(overlay_value "$overlay" runtime.mast3r_env)/bin/python" -I -B - "$typing_site" "$typing_sha" "$typing_version" "$(overlay_value "$overlay" runtime.mast3r_python)" "$(overlay_value "$overlay" runtime.mast3r_torch)" <<'PY'
import hashlib, platform, sys
from pathlib import Path
site = Path(sys.argv[1])
expected_sha, expected_version = sys.argv[2], sys.argv[3]
expected_python, expected_torch = sys.argv[4], sys.argv[5]
sys.path.insert(0, str(site))
import typing_extensions
from importlib import metadata
actual_file = Path(typing_extensions.__file__).resolve()
expected_file = (site / "typing_extensions.py").resolve()
assert actual_file == expected_file, f"typing_extensions import escaped frozen site: {actual_file}"
assert hashlib.sha256(actual_file.read_bytes()).hexdigest() == expected_sha
assert metadata.version("typing_extensions") == expected_version
import torch
assert platform.python_version() == expected_python
assert torch.__version__ == expected_torch
PY

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | awk '{print $1}')
required_devices=$(overlay_value "$overlay" execution.devices | wc -l | awk '{print $1}')
[ "$gpu_count" -ge "$required_devices" ] || die "insufficient GPUs: $gpu_count < $required_devices"
available_kb=$(df -Pk "$root" | awk 'NR==2 {print $4}')
[ "${available_kb:-0}" -gt 10485760 ] || die "less than 10 GiB available under $root"

artifacts=$(overlay_value "$overlay" runtime.artifacts)
lock_dir="$artifacts/environment_locks"
mkdir_under_root "$root" "$lock_dir"
lock_path="$lock_dir/a100_environment_lock.json"
"$python" - "$overlay" <<'PY' | write_immutable_file "$root" "$lock_path"
import json, platform, subprocess, sys
from pathlib import Path
from georeliab_mve import toml_compat as tomllib
overlay = Path(sys.argv[1])
payload = tomllib.loads(overlay.read_text(encoding='utf-8'))
def run(cmd): return subprocess.check_output(cmd, text=True).strip()
out = {
    'schema_version': 'a100-environment-lock-v1',
    'overlay': str(overlay),
    'overlay_sha256': run(['sha256sum', str(overlay)]).split()[0],
    'host_python': platform.python_version(),
    'git_version': run(['git', '--version']),
    'gpu_query': run(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total', '--format=csv,noheader']),
    'no_home_write': True,
    'tmpdir': str(Path(payload['runtime']['cache']) / 'tmp'),
    'dependencies': {
        'typing_extensions': {
            'site': payload['runtime']['typing_extensions_site'],
            'version': payload['resources']['typing_extensions_version'],
            'typing_extensions_file_path': str(Path(payload['runtime']['typing_extensions_site']) / 'typing_extensions.py'),
            'typing_extensions_file_sha256': payload['resources']['typing_extensions_sha256'],
        }
    },
    'runtime': payload['runtime'],
    'resources': payload['resources'],
    'execution': payload['execution'],
}
print(json.dumps(out, indent=2, sort_keys=True))
PY
sha256sum "$lock_path" | write_immutable_file "$root" "$lock_path.sha256"
printf '{"schema_version":"no-home-write-marker-v1","status":"ENFORCED","root":"%s"}\n' "$root" | write_immutable_file "$root" "$artifacts/no_home_write_marker.json"
info "P0 prerequisites verified; environment lock: $lock_path"
