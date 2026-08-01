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

## Round 1 Fix Addendum - 2026-08-01

### Reviewer findings addressed
1. CLI GPU selection now requires explicit `--requested-index`; no reusable code hardcodes GPU 1, and no CLI default can silently fall back to `cuda:0`. Formal invocation must pass `--requested-index 1` explicitly.
2. Frozen-env Torch probes now independently report logical cuda:0 device name and total memory, then re-query physical UUID/model/total-memory/process state after each probe.
3. `nvidia-smi` compute-process enumeration failures now fail closed with `V4_GPU_PROCESS_ENUMERATION_UNPROVEN` instead of being treated as an empty process list.
4. Failed preflight reruns now remove sibling authorization-owned PASS artifacts (`v4-execution-receipt.json`, `v4-execution-authorization.json`, `authorization.json`, `v4-execution-schedule.json`, `v4-state-inventory.json`) while preserving unrelated evidence.
5. Authorization now binds `root`, re-resolves `run_root`, `artifact_root`, `final_evidence_path`, and resource paths during validation, and rejects any rehashed finalizer forgery.
6. Atomic JSON promotion now requires artifact-specific staged validation for hardware preflight, receipt, and authorization payloads before replacement.

### Round 1 files changed
- `georeliab_mve/v4_authorization.py` - tightened fail-closed process enumeration, independent frozen-env post-probe binding, root/finalizer validation, stale sibling cleanup, and staged artifact validators.
- `georeliab_mve/cli.py` - made `v4-gpu-preflight --requested-index` mandatory.
- `tests/test_v4_authorization.py` - added regression coverage for mandatory CLI index, process enumeration failure, post-probe process leftovers, stale sibling cleanup, rehashed finalizer/root forgery, and invalid staged artifact non-promotion.

### Round 1 commands and exact results
- `python -m py_compile georeliab_mve\v4_authorization.py georeliab_mve\cli.py tests\test_v4_authorization.py` -> PASS.
- `python -m pytest tests\test_v4_authorization.py -q` -> PASS: `30 passed`, with one pre-existing pytest-asyncio deprecation warning.
- `python -m pytest tests\test_v4_execution_governance.py tests\test_storage_audit_refactor.py tests\test_storage_audit_cli_refactor.py tests\test_storage_science_lock_refactor.py -q` -> PASS: `73 passed`, with one pre-existing pytest-asyncio deprecation warning.

### Round 1 self-review
- No GPU/model/scientific execution was run.
- No protocol/config/split/corruption/metrics/adapters/evidence files were modified.
- The reusable preflight API still accepts any explicit nonnegative physical GPU index; the production CLI now forces the operator to pass the index explicitly, so the formal run can use exactly `--requested-index 1` without an implicit default.
- New tests exercise all six round-1 findings with fail-closed assertions and stale-artifact non-promotion checks.
- Remaining concern: broad `tests/test_v4*.py` remains unsuitable as completion evidence in this Windows worktree because of pre-existing temp/cache/staging ACL failures documented above; focused authorization and related governance/storage regressions passed.

## Round 2 Fix Addendum - 2026-08-01

### Blocking finding addressed
- `_evaluate_probes` no longer accepts missing post-probe physical evidence by falling back to the hardware sample. Every PASS probe must explicitly include typed `mapped_device_uuid`, `mapped_device_model`, `mapped_total_memory_bytes`, `post_probe_physical_model`, `post_probe_physical_total_memory_bytes`, and `compute_process_count` fields, and their values must match the sampled physical GPU with zero post-probe compute processes.
- `_pass_probe` test fixture now includes the explicit post-probe fields, so tests cannot construct PASS snapshots from old/incomplete probe payloads.
- Added regression coverage where omitted post-probe fields fail with `V4_GPU_TORCH_PROBE_SCHEMA_REQUIRED`, publish only a FAIL snapshot, leave no receipt, and leave no `.partial` artifact.
- Artifact-specific preflight staged validation continues to call `_evaluate_probes`, so staged PASS preflight artifacts with incomplete probe payloads are rejected before promotion.

### Round 2 commands and exact results
- `python -m py_compile georeliab_mve\v4_authorization.py tests\test_v4_authorization.py` -> PASS.
- `python -m pytest tests\test_v4_authorization.py -q` -> PASS: `31 passed`, with one pre-existing pytest-asyncio deprecation warning.
- `python -m pytest tests\test_v4_execution_governance.py tests\test_storage_audit_refactor.py tests\test_storage_audit_cli_refactor.py tests\test_storage_science_lock_refactor.py -q` -> PASS: `73 passed`, with one pre-existing pytest-asyncio deprecation warning.

### Round 2 self-review
- No GPU/model/scientific execution was run.
- No protocol/config/split/corruption/metrics/adapters/evidence files were modified.
- Remaining concern unchanged: broad `tests/test_v4*.py` is still not used as completion evidence because of pre-existing Windows temp/cache/staging ACL failures documented above.

## Round 3 Fix Addendum - 2026-08-01

### Missing regression added
- Added a direct `_atomic_json(..., validator=_validate_preflight_payload)` regression for otherwise PASS staged preflight payloads whose first probe omits each required field: `mapped_device_uuid`, `mapped_device_model`, `mapped_total_memory_bytes`, `post_probe_physical_model`, `post_probe_physical_total_memory_bytes`, and `compute_process_count`.
- Each parameterized case asserts exact `V4_GPU_TORCH_PROBE_SCHEMA_REQUIRED`, no target promotion, and no `.partial` artifact.
- No production code changed in round 3.

### Round 3 commands and exact results
- `python -m py_compile tests\test_v4_authorization.py` -> PASS.
- `python -m pytest tests\test_v4_authorization.py -q` -> PASS: `37 passed`, with one pre-existing pytest-asyncio deprecation warning.

### Round 3 self-review
- No GPU/model/scientific execution was run.
- No protocol/config/split/corruption/metrics/adapters/evidence files were modified.
- Remaining concern unchanged: broad `tests/test_v4*.py` is still not used as completion evidence because of pre-existing Windows temp/cache/staging ACL failures documented above.

## Final Bounded Fix Addendum - 2026-08-01

### Changes made
- Added `georeliab-v4-preflight-decision-1.0` blocked-decision artifacts for failed preflight runs as `v4-preflight-decision.json` next to the hardware snapshot.
- The blocked decision binds exact implementation anchor commit/tree, authorization revision, explicit requested index, hardware snapshot path and SHA, terminal `BLOCKED` status/reason, and `NO_SCIENTIFIC_RESULT`.
- Decision artifacts are written through `_atomic_json` with `_validate_preflight_decision_payload`, which revalidates schema, anchor, revision, requested-index type, blocked terminal fields, no-science marker, and snapshot SHA before replace.
- Failed preflight still removes stale PASS receipt/authorization/schedule artifacts; PASS preflight removes stale blocked decisions.
- Removed only newly surfaced unused imports from `tests/test_v4_authorization.py` so scoped ruff passes.

### Final bounded fix commands and exact results
- `ruff check georeliab_mve\v4_authorization.py tests\test_v4_authorization.py` -> PASS: `All checks passed!`.
- `python -m py_compile georeliab_mve\v4_authorization.py tests\test_v4_authorization.py` -> PASS.
- `python -m pytest tests\test_v4_authorization.py -q` -> PASS: `39 passed`, with one pre-existing pytest-asyncio deprecation warning.
- `git diff --check` -> PASS.

### Final bounded fix self-review
- No GPU/model/scientific execution was run.
- No checkpoint/model-forward, protocol/config/split/corruption/metrics/adapters/evidence files were touched.
- Decision artifacts are blocked-only; they are not PASS receipts, authorizations, or schedules.
- Remaining concern unchanged: broad `tests/test_v4*.py` is still not used as completion evidence because of pre-existing Windows temp/cache/staging ACL failures documented above.

## Final Bounded Fix Validator Strengthening - 2026-08-01

### Additional requested tightening
- Strengthened `_validate_preflight_decision_payload` to load the referenced hardware snapshot and require snapshot schema `georeliab-v4-hardware-preflight-1.0`, `status=FAIL`, matching `reason_code`, exact implementation anchor commit/tree, and matching requested physical index.
- Added parameterized mismatch regressions for PASS snapshots, reason mismatch, commit mismatch, tree mismatch, and requested-index mismatch; each asserts no decision promotion and no `.partial` artifact.

### Superseding final bounded fix commands and exact results
- `ruff check georeliab_mve\v4_authorization.py tests\test_v4_authorization.py` -> PASS: `All checks passed!`.
- `python -m py_compile georeliab_mve\v4_authorization.py tests\test_v4_authorization.py` -> PASS.
- `python -m pytest tests\test_v4_authorization.py -q` -> PASS: `44 passed`, with one pre-existing pytest-asyncio deprecation warning.
- `git diff --check` -> PASS.

### Validator-strengthening self-review
- No GPU/model/scientific execution was run.
- No checkpoint/model-forward, protocol/config/split/corruption/metrics/adapters/evidence files were touched.
- The blocked decision cannot bind an unrelated, PASS, stale-anchor, wrong-index, or wrong-reason snapshot even when the referenced snapshot SHA is self-consistent.

## Review Follow-up: Immutable Auth Revision and Same-Run Binding - 2026-08-01

### HIGH findings fixed
1. Blocked decisions and hardware snapshots now bind full immutable authorization code revision fields: `authorization_commit` and `authorization_tree`. Runtime preflight resolves the current git `HEAD` and `HEAD^{tree}` immediately before artifact generation when explicit full values are not supplied; tests use immutable dummy SHAs to avoid depending on the live worktree revision. The implementation anchor `7381e60050143a78fca6a3ebde5706ae27d2c145` / `f4e2b1104496c817693aaa5989d0276d2ebe03e9` remains separate.
2. Hardware snapshots and blocked decisions now share a cryptographic `run_id`; the decision validator cross-checks snapshot path, SHA, run ID, reason, implementation anchor, authorization commit/tree, and requested index. Runtime decision staging uses a validator closure with expected snapshot path, expected authorization revision, and expected run ID, so a decision cannot bind another self-consistent FAIL snapshot.

### Added regressions
- External-process failure still publishes snapshot + blocked decision and no PASS receipt/authorization/schedule.
- Staged decision rejects stale authorization commit/tree under the expected-runtime-revision closure.
- Staged decision rejects a different matching FAIL snapshot under the expected-snapshot-path closure.
- Existing snapshot mismatch tests still reject PASS, wrong reason, stale implementation anchor, and wrong requested-index snapshots.

### Commands and exact results
- `ruff check georeliab_mve\v4_authorization.py tests\test_v4_authorization.py` -> PASS: `All checks passed!`.
- `python -m py_compile georeliab_mve\v4_authorization.py tests\test_v4_authorization.py` -> PASS.
- `python -m pytest tests\test_v4_authorization.py -q` -> PASS: `46 passed`, with one pre-existing pytest-asyncio deprecation warning.
- `git diff --check` -> PASS.

### Self-review
- No GPU/model/scientific execution was run.
- No checkpoint/model-forward, protocol/config/split/corruption/metrics/adapters/evidence files were touched.
- No future commit SHA is hardcoded; runtime resolution uses the current git HEAD/tree or explicit full values.
- Remaining concern unchanged: broad `tests/test_v4*.py` is still not used as completion evidence because of pre-existing Windows temp/cache/staging ACL failures documented above.

## CUDA Runtime Field-Completeness Fix - 2026-08-01

### Formal dry preflight follow-up
- A formal dry preflight engineering artifact exposed `cuda_runtime: unknown`; that artifact is superseded as engineering evidence only.
- The dry preflight did not produce a PASS receipt, authorization, schedule, checkpoint/model forward, or scientific result.

### Changes made
- `nvidia_smi_hardware_sample` now proves CUDA runtime from a separate read-only `nvidia-smi` banner parse, recording values such as `CUDA Version 13.0` from the host banner.
- Removed the previous `nvcc --version` fallback and the silent `unknown` write path.
- CUDA runtime is fail-closed with `V4_GPU_CUDA_RUNTIME_UNPROVEN` when the banner is unavailable or malformed.
- Snapshot staging now rejects any present sample or device summary whose `cuda_runtime` is empty or `unknown`.
- External-process failure precedence remains intact when CUDA runtime is available: process enumeration is still proven and active processes still block with `V4_GPU_RECEIPT_NO_ACTIVE_COMPUTE_PROCESS`.

### Commands and exact results
- `ruff check georeliab_mve\v4_authorization.py tests\test_v4_authorization.py` -> PASS: `All checks passed!`.
- `python -m py_compile georeliab_mve\v4_authorization.py tests\test_v4_authorization.py` -> PASS.
- `python -m pytest tests\test_v4_authorization.py -q` -> PASS: `52 passed`, with one pre-existing pytest-asyncio deprecation warning.
- `git diff --check` -> PASS.

### Self-review
- No GPU/model/scientific execution was run for this fix.
- No checkpoint/model-forward, protocol/config/split/corruption/metrics/adapters/evidence files were touched.
- No permanent cuda index or UUID was hardcoded; tests continue to use injected samplers and dummy immutable SHAs.
- Remaining concern unchanged: broad `tests/test_v4*.py` is still not used as completion evidence because of pre-existing Windows temp/cache/staging ACL failures documented above.

## Whole-Branch Review Fix: Interval, Driver, and Reason Codes - 2026-08-01

### REQUEST CHANGES findings fixed
1. Formal preflight sample interval is now exactly `5.0` seconds. The CLI no longer exposes `--sample-interval-seconds`, and the core preflight rejects non-5.0 before sampling or artifact creation with `V4_GPU_SAMPLE_INTERVAL_MUST_BE_5_SECONDS`. Tests remain fast by passing `5.0` with injected no-op sleepers.
2. Every present hardware sample and device summary must have nonempty/proven `driver_version`; `_evaluate_basic` rejects missing driver versions and rejects two-sample driver drift with `V4_GPU_DRIVER_VERSION_DRIFT`.
3. `create_hardware_preflight` now preserves exact `V4ExecutionError` reason codes raised by samplers and probe runners; only unexpected exceptions map to generic `V4_GPU_BASIC_SAMPLE_UNAVAILABLE` or `V4_GPU_TORCH_PROBE_FAILED`.

### Added regressions
- Non-5.0 interval rejects before sampler invocation and creates no artifact.
- Driver version drift blocks preflight; missing sample/device driver versions fail staged snapshot validation.
- Sampler-raised `V4_GPU_CUDA_RUNTIME_UNPROVEN` and probe-raised `V4_GPU_TORCH_PROBE_VISIBLE_DEVICE_MISMATCH` remain exact published reason codes.
- CLI preflight tests no longer pass interval overrides.

### Commands and exact results
- `ruff check georeliab_mve\v4_authorization.py georeliab_mve\cli.py tests\test_v4_authorization.py` -> PASS: `All checks passed!`.
- `python -m py_compile georeliab_mve\v4_authorization.py georeliab_mve\cli.py tests\test_v4_authorization.py` -> PASS.
- `python -m pytest tests\test_v4_authorization.py -q` -> PASS: `58 passed`, with one pre-existing pytest-asyncio deprecation warning.
- `python -m pytest tests\test_v4_execution_governance.py tests\test_storage_audit_refactor.py tests\test_storage_audit_cli_refactor.py tests\test_storage_science_lock_refactor.py -q` -> PASS: `73 passed`, with one pre-existing pytest-asyncio deprecation warning.
- `git diff --check` -> PASS.

### Self-review
- No GPU/model/scientific execution was run.
- No checkpoint/model-forward, protocol/config/split/corruption/metrics/adapters/evidence files were touched.
- No permanent cuda index or UUID was hardcoded; existing evidence entries remain preserved as superseded engineering evidence where noted.
- Remaining concern unchanged: broad `tests/test_v4*.py` is still not used as completion evidence because of pre-existing Windows temp/cache/staging ACL failures documented above.
