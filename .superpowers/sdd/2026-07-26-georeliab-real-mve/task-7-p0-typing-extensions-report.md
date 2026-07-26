# Task 7 P0 typing_extensions fail-closed report

## Outcome

Implemented the P0 frozen `typing_extensions` dependency path so Torch imports no longer depend on `/home/smli/.local` after `export_runtime_cache_env` redirects `HOME`. The frozen source is `/home/smli/miniforge3/pkgs/typing_extensions-4.15.0-pyhcf101f3_0/site-packages/typing_extensions.py`, version `4.15.0`, SHA-256 `433d11d170d3a24d2eb065ebc1bfe848cea7e3d7ce68567ab52bea2d4c2f7ed8`.

## Changes

- Overlay now freezes `runtime.typing_extensions_site` and keeps the immutable identity values in `resources.typing_extensions_version` plus `resources.typing_extensions_sha256`.
- `scripts/a100/common.sh` validates the frozen site and SHA before Python-backed overlay imports, sets `PYTHONNOUSERSITE=1`, and exports exact `PYTHONPATH` as frozen site plus project root.
- `scripts/a100/verify_prereqs.sh` validates import origin/hash under `-I -B` and writes dependency path/version/hash into the environment lock.
- `georeliab_mve/materialization.py` hashes and records `typing_extensions.py` in runtime identity and frozen materialization provenance, with hash mismatch fail-closed checks.
- `georeliab_mve/adapters.py` and `georeliab_mve/runner.py` carry the dependency site/version/hash through `FrozenRuntime`, model specs, isolated probes, manifests, and worker environment setup.
- Tests cover overlay contract, required keys, provenance dependency recording, isolated probe dependency arguments, hash mismatch, and HOME/PYTHONNOUSERSITE semantics.

## Verification

- `python -m py_compile georeliab_mve/adapters.py georeliab_mve/materialization.py georeliab_mve/prepare_dispatch_round1.py georeliab_mve/runner.py tests/test_deployment_task6.py tests/test_preparation_round2.py tests/test_materialization_round3.py tests/test_real_adapters_task4.py tests/test_runner_task5.py` passed.
- `bash -n scripts/a100/common.sh scripts/a100/verify_prereqs.sh` passed.
- `python -m pytest tests/test_deployment_task6.py tests/test_real_adapters_task4.py tests/test_materialization_round3.py tests/test_preparation_round2.py tests/test_runner_task5.py::test_model_spec_records_mast3r_config_sha_and_preflight_device_default tests/test_runner_task5.py::test_default_adapter_factory_reuses_upstream_but_isolates_outputs_and_cache tests/test_runner_task5.py::test_isolated_model_workers_build_frozen_typing_extensions_env -q` passed: 57 tests, one pytest-asyncio deprecation warning.

## Notes

`apply_patch` could not run in this Windows restricted-token sandbox because the tool could not prepare split writable roots. Edits were made with deterministic local scripts after the failure was observed. No scientific thresholds, schedules, or test grids were changed.