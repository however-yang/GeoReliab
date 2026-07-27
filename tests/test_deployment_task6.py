from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from georeliab_mve import toml_compat as tomllib
from georeliab_mve.preparation import A100Overlay


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / 'configs' / 'a100_real_mve_overlay.toml'
SCRIPTS = sorted((ROOT / 'scripts' / 'a100').glob('*.sh'))


def _payload() -> dict:
    return tomllib.loads(OVERLAY.read_text(encoding='utf-8'))


def _bash_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(':').lower()
    if not drive:
        return resolved.as_posix()
    rest = resolved.as_posix().split(':', 1)[1]
    return f'/mnt/{drive}{rest}'


def test_a100_overlay_contains_only_deployment_tables_and_loads() -> None:
    payload = _payload()
    assert set(payload) == {'runtime', 'resources', 'execution'}
    overlay = A100Overlay.load(OVERLAY)
    assert overlay.runtime_root == '/srv/private/smli/GeoReliab'
    text = OVERLAY.read_text(encoding='utf-8')
    for token in (
        'geometry_gate', 'georeliab_gate', 'bootstrap_resamples',
        'rho_decline', 'failure_auroc', 'zero_update_auroc_gain',
        'delta_geom', 'recovery', 'confidence_failure_gate',
    ):
        assert token not in text


def test_a100_overlay_freezes_paths_resources_and_limits() -> None:
    payload = _payload()
    runtime = payload['runtime']
    resources = payload['resources']
    execution = payload['execution']
    for key in ('root', 'data', 'cache', 'logs', 'artifacts', 'git_bare', 'worktrees', 'stage', 'prepared', 'rendered', 'manifests', 'evidence'):
        assert str(runtime[key]).startswith('/srv/private/smli/GeoReliab')
    for key in ('vggt_source', 'mast3r_source', 'dust3r_source', 'croco_source', 'vggt_env', 'mast3r_env'):
        assert str(runtime[key]).startswith('/home/smli/')
    assert runtime['typing_extensions_site'] == '/home/smli/miniforge3/pkgs/typing_extensions-4.15.0-pyhcf101f3_0/site-packages'
    assert resources['typing_extensions_version'] == '4.15.0'
    assert resources['typing_extensions_sha256'] == '433d11d170d3a24d2eb065ebc1bfe848cea7e3d7ce68567ab52bea2d4c2f7ed8'
    assert resources['vggt_source_commit'] == 'a288dd0f14786c93483e45524328726ab7b1b4ce'
    assert resources['mast3r_source_commit'] == 'f5209afc300cec36239a7ac992263f36847bbba0'
    assert resources['dust3r_source_commit'] == '3cc8c88c413bb9e34c41db0e0eef99c2ee010b12'
    assert resources['croco_source_commit'] == 'd7de0705845239092414480bd829228723bf20de'
    assert resources['vggt_checkpoint_sha256'] == 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
    assert resources['mast3r_checkpoint_sha256'] == '0a615eb05fa9db654050aa655945ee5696e7c6c1b7f93f1ee8c37249010f6feb'
    assert resources['mast3r_config_sha256'] == '718eb93dc4f9e4332b60cc0041af962d712cbd346d7770ce35c5b22cff68eae4'
    assert execution['gpu_hour_limit'] == 50
    assert execution['preflight_gpu_hour_reservation'] == 2.0
    assert execution['max_storage_bytes'] == 1_000_000_000_000
    assert execution['devices'] == ['cuda:0', 'cuda:1']
    assert execution['force_tmp_under_runtime'] is True


@pytest.mark.parametrize('script', SCRIPTS, ids=lambda p: p.name)
def test_a100_shell_scripts_parse(script: Path) -> None:
    result = subprocess.run(['bash', '-n', _bash_path(script)], cwd=ROOT, text=True, capture_output=True)
    combined = (result.stdout or '') + (result.stderr or '')
    if 'E_ACCESSDENIED' in combined.replace('\x00', '') or 'Win32 error 5' in combined.replace('\x00', ''):
        pytest.skip('local Bash/WSL unavailable in this sandbox')
    assert result.returncode == 0, combined


def test_a100_shell_scripts_are_executable_in_git() -> None:
    for script in SCRIPTS:
        relative = script.relative_to(ROOT).as_posix()
        staged = subprocess.run(
            ['git', 'ls-files', '--stage', '--', relative],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        assert staged.startswith('100755 '), f'{relative} is not executable: {staged}'


def test_a100_scripts_fail_closed_on_paths_and_disallowed_git_operations() -> None:
    assert {path.name for path in SCRIPTS} == {
        'common.sh', 'deploy_commit.sh', 'finalize_p6.sh', 'launch_stage.sh', 'status.sh', 'verify_prereqs.sh',
    }
    all_text = '\n'.join(path.read_text(encoding='utf-8') for path in SCRIPTS)
    assert 'guard_under_root' in all_text
    assert 'refusing writable path below /home' in all_text
    assert 'export TMPDIR=' in all_text
    assert 'git reset' not in all_text
    assert 'push --force' not in all_text
    assert 'git push -f' not in all_text
    assert 'rm -rf' not in all_text
    assert 'rm -f' not in all_text
    assert 'screen -dmS' in all_text


def test_launch_script_exposes_p0_to_p6_and_runner_commands() -> None:
    text = (ROOT / 'scripts' / 'a100' / 'launch_stage.sh').read_text(encoding='utf-8')
    for stage in ('p0)', 'p1)', 'p2)', 'p3)', 'p4)', 'p5)', 'p6)'):
        assert stage in text
    for command in ('prepare-georeliab', 'preflight-real', 'run-georeliab', 'audit-georeliab'):
        assert command in text
    assert 'write_native_phenomenon_gate_from_audit_output' in text
    assert '--stage smoke --model all --device cuda:0 --shard 0/2' in text
    assert '--stage test --model all --device cuda:0 --shard 0/2' in text
    assert '--stage zero-update --model all' in text
    assert 'require_native_gate_pass; enforce_stage_gpu_budget zero-update' in text
    assert 'require_p0_complete; enforce_stage_gpu_budget preflight' in text
    assert 'require_p1_complete; enforce_stage_gpu_budget smoke' in text
    assert "--summary-json '$root/artifacts/p1_preflight.json'" in text
    assert 'SHORT_CIRCUIT_P5' in text
    assert 'require_stage_complete smoke 200' in text
    assert 'require_stage_complete test 400' in text


def test_overlay_value_uses_dependency_light_top_level_toml_import() -> None:
    common = (ROOT / 'scripts' / 'a100' / 'common.sh').read_text(encoding='utf-8')
    overlay_value = common.split('overlay_value() {', 1)[1].split('\nruntime_root()', 1)[0]
    assert 'PYTHONNOUSERSITE=1 PYTHONPATH="$typing_site:$GEORELIAB_PROJECT_ROOT/georeliab_mve"' in overlay_value
    assert 'import toml_compat as tomllib' in overlay_value
    assert 'from georeliab_mve import toml_compat' not in overlay_value
    assert 'from georeliab_mve import' not in overlay_value


def test_scripts_pin_all_runtime_caches_off_home_and_mark_no_home_write() -> None:
    launch = (ROOT / 'scripts' / 'a100' / 'launch_stage.sh').read_text(encoding='utf-8')
    common = (ROOT / 'scripts' / 'a100' / 'common.sh').read_text(encoding='utf-8')
    verify = (ROOT / 'scripts' / 'a100' / 'verify_prereqs.sh').read_text(encoding='utf-8')
    for name in (
        'HOME', 'TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'HF_HOME', 'HF_HUB_CACHE',
        'TRANSFORMERS_CACHE', 'TORCH_HOME', 'TORCH_EXTENSIONS_DIR', 'CUDA_CACHE_PATH',
        'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'CUPY_CACHE_DIR', 'MPLCONFIGDIR',
        'PYTHONPYCACHEPREFIX',
    ):
        assert f'export {name}=' in common or f"{name}='" in launch
    assert 'GEORELIAB_NO_HOME_WRITE=1' in common
    assert 'PYTHONNOUSERSITE=1' in common
    assert 'typing_site=$(CDPATH= cd -- "$typing_site" 2>/dev/null && pwd -P)' in common
    assert 'typing_extensions site must equal frozen package cache' in common
    assert 'typing_extensions site must resolve to frozen package cache' in common
    assert 'validate_typing_extensions_runtime "$overlay"' in common
    assert 'PYTHONPATH="$typing_site:$GEORELIAB_PROJECT_ROOT"' in common
    assert 'PYTHONPATH="$typing_site:$GEORELIAB_PROJECT_ROOT/georeliab_mve"' in common
    assert 'import toml_compat as tomllib' in common
    assert 'from georeliab_mve import toml_compat' not in common
    assert 'no_home_write_marker.json' in verify
    assert 'runtime.cache' in common


def test_deploy_and_verify_are_idempotent_and_reject_correct_dirty_state() -> None:
    deploy = (ROOT / 'scripts' / 'a100' / 'deploy_commit.sh').read_text(encoding='utf-8')
    verify = (ROOT / 'scripts' / 'a100' / 'verify_prereqs.sh').read_text(encoding='utf-8')
    common = (ROOT / 'scripts' / 'a100' / 'common.sh').read_text(encoding='utf-8')
    assert 'status --porcelain --untracked-files=all' in deploy
    assert 'diff --quiet HEAD --' in common
    assert 'write_immutable_file' in deploy
    assert 'write_immutable_file' in verify
    assert '<<\'PY\' | write_immutable_file "$root" "$manifest"' in deploy
    assert '<<\'PY\' | write_immutable_file "$root" "$lock_path"' in verify
    assert 'immutable artifact mismatch' in common
    assert 'mv -f' not in deploy
    assert 'mv -f' not in verify


def test_deploy_manifest_is_bound_to_target_worktree_overlay() -> None:
    deploy = (ROOT / 'scripts' / 'a100' / 'deploy_commit.sh').read_text(encoding='utf-8')
    assert 'target_overlay="$target/$overlay_rel"' in deploy
    assert 'cmp -s "$overlay" "$target_overlay"' in deploy
    assert 'source and target worktree overlays differ' in deploy
    assert 'orchestrator_python "$target_overlay"' in deploy
    assert '- "$overlay_rel" "$target_overlay" "$commit" "$bare" "$target"' in deploy
    assert "'overlay': overlay_name" in deploy
    assert "'overlay_sha256': sha(overlay_path)" in deploy
    assert '- "$overlay" "$commit" "$bare" "$target"' not in deploy


def test_every_stage_is_bound_to_clean_exact_commit_and_resource_budget() -> None:
    launch = (ROOT / 'scripts' / 'a100' / 'launch_stage.sh').read_text(encoding='utf-8')
    common = (ROOT / 'scripts' / 'a100' / 'common.sh').read_text(encoding='utf-8')
    finalize = (ROOT / 'scripts' / 'a100' / 'finalize_p6.sh').read_text(encoding='utf-8')
    assert 'assert_project_worktree_clean "$worktree" "$commit"' in launch
    assert 'assert_project_worktree_clean "$worktree" "$commit"' in finalize
    assert 'status --porcelain --untracked-files=all' in common
    assert 'project worktree commit mismatch' in common
    assert 'enforce_stage_gpu_budget preflight' in launch
    assert 'enforce_stage_gpu_budget smoke' in launch
    assert 'enforce_stage_gpu_budget test' in launch
    assert 'BUDGET_ESTIMATE_UNAVAILABLE' in launch
    assert 'BLOCKED_RESOURCE_BUDGET' in launch


def test_status_reports_canonical_budget_relevant_observations() -> None:
    status = (ROOT / 'scripts' / 'a100' / 'status.sh').read_text(encoding='utf-8')
    for token in ('observed_gpu_hours', 'peak_memory_mb', 'budget_evidence_status', 'ledger_parse_errors', 'missing_count_tail_sum', 'invalid_count_tail_sum', 'canonical_schedule_counts', 'schedule_counts_tail', 'reason_codes_tail', 'output_bytes'):
        assert token in status
    assert 'stage_progress_counts' in status
    assert 'runtime_by_key' in status
    assert "glob('*/ledger.jsonl')" in status
    assert "glob('*/stage/*/ledger.jsonl')" in status
    assert 'enforce_storage_cap' in status


def test_p6_generates_fail_closed_evidence_bundle_and_one_page_decision_table() -> None:
    launch = (ROOT / 'scripts' / 'a100' / 'launch_stage.sh').read_text(encoding='utf-8')
    finalize = (ROOT / 'scripts' / 'a100' / 'finalize_p6.sh').read_text(encoding='utf-8')
    assert 'p6_status_live.json' in launch
    assert 'p6_evidence_bundle.json' in finalize
    assert 'p6_one_page_decision.md' in finalize
    assert 'BLOCKED_PENDING_GEOMETRY' in finalize
    assert 'GeoReliab PASS remains GEORELIAB_PASS_PENDING_GEOMETRY' in finalize
    for token in (
        "status not in {'PASS', 'FAIL'}", 'missing required P6 provenance',
        'stage_freeze.json', 'split_view_manifest.json', 'frozen_materialization.json',
        'corruption_calibration.json', 'corruption_calibration_qa.json',
        'test_render_lock.json', 'test_render_index.json', 'render_inputs_test.json',
        'tartanair_p000_pairs.json', 'tartanair_native_fog_sanity.json',
        'official_resource_hashes', 'project_tree', 'worker_log_hashes',
        'p6_status_core.json', 'existing P6 bundle does not match current commit/native gate',
        'P6 canonical schedule counts are UNAVAILABLE', "require_complete('smoke', 200)",
        "require_complete('test', 400)", "require_complete('zero-update', 480)",
        'P4 FAIL must short-circuit P5', 'P5_INVALID_SUBSET_PREDICTION',
        'BLOCKED_RESOURCE_BUDGET', "gate.get('status') not in {'PASS', 'FAIL'}",
        "'deployment': req", "rev-parse', 'HEAD^{tree}'",
        "status.get('budget_evidence_status') != 'OK'", "status.get('ledger_parse_errors') != []",
    ):
        assert token in finalize


def test_runbook_has_one_fail_closed_authoritative_p6_path() -> None:
    text = (ROOT / 'docs' / 'A100_REAL_MVE_RUNBOOK.md').read_text(encoding='utf-8')
    assert 'The only authoritative outputs are:' in text
    assert '$ROOT/artifacts/p6_evidence_bundle.json' in text
    assert 'bundle_path.write_text' not in text
    assert "evidence_dir / 'p6_evidence_bundle.json'" not in text


def test_prereqs_enforce_storage_cap_device_count_and_frozen_env_orchestrator() -> None:
    verify = (ROOT / 'scripts' / 'a100' / 'verify_prereqs.sh').read_text(encoding='utf-8')
    common = (ROOT / 'scripts' / 'a100' / 'common.sh').read_text(encoding='utf-8')
    assert 'enforce_storage_cap' in verify
    assert 'gpu_count' in verify and 'required_devices' in verify
    assert 'orchestrator_python' in common
    assert 'runtime.vggt_env' in common
    assert 'die "frozen VGGT environment Python is required' in common
    assert 'toml_compat' in common
    assert 'import tomli' not in common
    assert '-I -B - "$typing_site" "$typing_sha" "$typing_version"' in verify
    assert 'expected_python, expected_torch = sys.argv[4], sys.argv[5]' in verify
    assert 'assert platform.python_version() == "$(' not in verify
    assert 'import typing_extensions' in verify
    assert 'metadata.distribution("typing_extensions")' in verify
    assert 'typing_extensions-4.15.0.dist-info' in verify
    assert 'dist_info == expected_dist_info' in verify
    assert 'typing_extensions_file_sha256' in verify
    assert 'typing_extensions_dist_info_path' in verify
    assert 'typing_extensions_file_path' in verify
