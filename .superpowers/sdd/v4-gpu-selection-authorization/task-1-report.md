# Task 1 Report - CPU-only v4 execution authorization

## Status
DONE_WITH_CONCERNS

## Files changed
- `georeliab_mve/v4_authorization.py` - New CPU-only v4 GPU preflight and execution authorization module.
  - Adds rich two-sample hardware snapshot publication with injectable sampler/sleeper/probe seams.
  - Gates PASS on stable requested/resolved index, UUID, and model; exact `NVIDIA A100 80GB PCIe`; MIG disabled; no compute processes; zero utilization; >=16 GiB free memory; and OK health/ECC state.
  - Gives active external compute processes precedence over utilization and fails with exact reason `V4_GPU_RECEIPT_NO_ACTIVE_COMPUTE_PROCESS`.
  - Runs lightweight per-model Torch visibility probes only after basic hardware PASS, using frozen-env Python executables from `GEORELIAB_V4_VGGT_PYTHON` and `GEORELIAB_V4_MAST3R_PYTHON`; the default path does not fall back to generic `python`.
  - Captures process owner as username on POSIX systems instead of raw UID when `/proc` owner lookup is available.
  - Publishes FAIL snapshots only; failed reruns remove any previous generated PASS receipt at the same output location.
  - PASS writes a separate strict `V4ExecutionReceipt` bound to the snapshot SHA and schedule SHA.
  - Adds versioned `V4ExecutionAuthorization` binding receipt, protocol, anchor commit/tree, schedule, resource inventory, roots, final evidence path, budget ceilings, forbidden expansions, and finalizer callable.
  - Removes any previous generated authorization at the requested output before re-creation so a failed rerun cannot leave an old PASS authorization discoverable at that path.
  - Validates authorization fail-closed for stale anchor, receipt tamper, preflight tamper, schedule/resource tamper, path escape, expanded scope/budget/fallback/parallel/retry controls, and malicious receipt controls.
- `georeliab_mve/cli.py` - Adds public CLI commands:
  - `python -m georeliab_mve v4-gpu-preflight`
  - `python -m georeliab_mve v4-create-execution-authorization`
  - `python -m georeliab_mve v4-validate-execution-authorization`
- `tests/test_v4_authorization.py` - New focused tests for hardware gates, external-process precedence, PASS/FAIL side effects, stale PASS cleanup, authorization creation/validation, tamper/expansion/path controls, malicious receipt controls, and CLI fail-closed behavior.

## Commands and exact results
- `python -m py_compile georeliab_mve\v4_authorization.py georeliab_mve\cli.py tests\test_v4_authorization.py` -> PASS.
- `python -m pytest tests\test_v4_authorization.py -q` -> PASS: `24 passed`, with one pre-existing pytest-asyncio deprecation warning.
- `python -m pytest tests\test_v4_execution_governance.py tests\test_storage_audit_refactor.py tests\test_storage_audit_cli_refactor.py tests\test_storage_science_lock_refactor.py -q` -> PASS: `73 passed`, with one pre-existing pytest-asyncio deprecation warning.
- Attempted broader `test_v4*.py` sweep with explicit file list -> NOT RELIABLE: failed from Windows temp/cache/staging permission errors before/inside existing tmp-path fixtures, including denied access to `C:\Users\SZ597\AppData\Local\Temp\pytest-of-SZ597`, `C:\tmp\georeliab-v4-pytest`, and ignored `.pytest-local` staging directories. This was not used as completion evidence. The focused v4 execution/storage regression command above was rerun afterward and passed.

## Plan-critical self-check
1. External compute process precedence: implemented and tested. If a formal GPU sample has both active compute processes and nonzero utilization, the preflight reason is `V4_GPU_RECEIPT_NO_ACTIVE_COMPUTE_PROCESS`.
2. Frozen VGGT/MASt3R Python envs: implemented. Default probes require `GEORELIAB_V4_VGGT_PYTHON` and `GEORELIAB_V4_MAST3R_PYTHON`; no generic Python fallback is used.
3. Process owner: implemented for POSIX `/proc` by converting UID to username via `pwd.getpwuid`; Windows remains `None` because no new dependency was allowed.
4. Failed rerun stale PASS artifacts: implemented and tested for the generated receipt and authorization output paths.
5. Strict receipt invariants: implemented. Creation and validation explicitly recheck explicit selection, protocol, v4 scientific scope/stage, max concurrency 1, sequential model/unit execution, no fallback/switch/retry, and exact authorized model before accepting a receipt.

## Self-review
- No GPU/model/scientific work was run.
- No protocol/config/split/corruption/metrics/adapters/evidence files were modified.
- The new preflight has injectable seams and the default path shells out only to metadata commands plus lightweight Torch introspection after basic PASS.
- Basic-sample failures do not run probes and do not publish receipts or authorizations.
- Authorization creation requires an already PASS preflight receipt, exact implementation anchor commit/tree, exact resource inventory keys, schedule digest match, bounded paths, and frozen scope/budget controls.
- Validation recomputes the authorization digest and rechecks receipt/preflight/resource bindings.

## Known concerns
- The repository's existing compact `validate_v4_gpu_receipt` preflight schema is left unchanged for backwards-compatible tests; the new rich snapshot is validated in `v4_authorization.py` rather than weakening the legacy validator.
- The broad `tests/test_v4*.py` run could not be completed due Windows filesystem permission errors unrelated to source diffs. Focused new tests plus relevant v4 execution/storage regressions passed cleanly.
- Some ignored pytest temp/cache directories are currently Windows ACL-locked and could not be removed with `Remove-Item -Recurse -Force`; they are not tracked by git.