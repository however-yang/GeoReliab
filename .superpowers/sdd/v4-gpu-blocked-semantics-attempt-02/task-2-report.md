# Task 2 execution report — attempt-02 selection and authorization tooling

## Execution identity

- Brief: `.superpowers/sdd/v4-gpu-blocked-semantics-attempt-02/task-2-brief.md`
- Branch: `codex/v4-gpu-selection-attempt-02`
- Parent: `3785a38f29ae9b8466fffb10c403e93538dfa7ef`
- Scientific implementation anchor: commit `7381e60050143a78fca6a3ebde5706ae27d2c145`, tree `f4e2b1104496c817693aaa5989d0276d2ebe03e9`
- Attempt ID: `attempt-02`

No live GPU enumeration, Torch execution, model/checkpoint load, forward pass,
metric computation, dispatcher, MVE, merge, tag, PR update, or push was
performed. All GPU and Torch behavior was exercised through injected test
doubles only.

## Changed files

- `georeliab_mve/v4_authorization.py`
  - Added attempt-scoped all-GPU inventory, deterministic UUID-first selection,
    frozen-environment mapping-only probe validation, atomic preflight/decision/
    PASS-receipt publication, production resource closure materialization, and
    execution authorization creation/validation.
  - Extended the existing v4 authorization module and reused its frozen scope,
    path, hashing, atomic JSON, schedule, and science-lock contracts instead of
    adding a separate authorization subsystem.
- `tests/test_v4_attempt02_authorization.py`
  - Added attempt-02 regression coverage using CPU-only fixtures and injected
    samplers/probes.
- `.superpowers/sdd/v4-gpu-blocked-semantics-attempt-02/task-2-report.md`
  - Added this execution and verification record.

No dependency was added and no frozen scientific configuration was changed.

## Requirement self-check

### Identity and isolation: ✅ complete

- Exact attempt ID, implementation commit, and implementation tree are required
  by preflight, receipt, resource snapshot, and authorization validators.
- Every generated evidence/output path must contain the exact `attempt-02` path
  segment and must not contain `attempt-01`.
- Existing targets and owned `.partial` siblings are rejected before work.
- Supplied historical evidence is recursively checked for attempt-01 run IDs,
  evidence timestamps, and snapshot/decision/receipt/authorization digests;
  cross-attempt linkage and reuse fail closed.

### All-GPU inventory and selection: ✅ complete

- CPU/read-only inventory records all visible devices, including index, UUID,
  exact model, total/free/used memory, utilization, temperature, driver/CUDA,
  MIG, ECC/health, hostname/timestamp, and full compute-process identity.
- Exactly two inventory samples are taken exactly five seconds apart.
- The excluded UUID is removed before eligibility and ordering.
- Eligibility requires stable UUID/index identity in both samples, exact A100
  80GB PCIe model, MIG disabled, zero processes, zero utilization, at least
  16 GiB free, and proven health/identity/process ownership.
- Ordering is total memory descending, free memory descending, UUID lexical
  ascending. UUID is authoritative; index is carried only for runtime mapping.
- One selected candidate is immutable. VGGT and MASt3R probes validate exactly
  one visible device mapped at logical `cuda:0`, matching UUID/model/memory,
  with no model/checkpoint/forward work and no residual process.
- No fallback, retry, or device switch is authorized. Empty eligibility is the
  terminal `V4_NO_ELIGIBLE_IDLE_GPU`; all other failures retain exact evidence.

### Production CPU-only resources: ✅ complete

- Materialization validates exact VGGT/MASt3R weight/config/upstream commit
  bindings; DTU SampleSet/Points/Rectified archive SHA, ETag, central-directory,
  and referenced-member digests; v4 split; state/fog manifests; environment
  locks; and science lock.
- Pytest cache, attempt-01, and prior authorization evidence cannot be canonical
  production sources.
- The model-independent inventory is exactly 200 states and rebuilds the exact
  400-unit VGGT/MASt3R × 20-scene schedule. L3 is deduplicated to exactly 40
  model/scene inferences; the fog manifest is Koschmieder beta-only.
- State, schedule, and resource snapshot publication is `.partial`, validated,
  hash-bound, atomically replaced, and collision rejecting.

### Execution authorization: ✅ complete

- Authorization binds the PASS receipt, resource snapshot, schedule, exact
  selected UUID/index, exact anchor/protocol, roots, final evidence path, and
  authorized finalizer.
- Frozen scope is inherited from the existing authorization contract: VGGT and
  MASt3R only; 20 DTU scenes and 200/400 state/schedule closure; Pose primary;
  Fusion/F-score supporting; 35 GPU-h/150 GB stop; 50 GPU-h/1 TB disaster
  ceiling; single-GPU sequential execution; and no fallback/retry/switch.
- UAVLight, v4.1, third model, second corruption family, and seed/severity/grid
  expansion remain forbidden.
- Validation rejects stale attempt/anchor/protocol, UUID/index drift, receipt,
  schedule, resource, signature, and scientific-config tamper, scope expansion,
  unsafe paths, parallel execution, fallback, retry, and device switching.
- FAIL evidence never creates a PASS receipt or authorization.

### Tests and prohibited actions: ✅ complete, with one verification concern

- Required focused, governance/authorization, budget/storage, Ruff, py_compile,
  shell syntax, whitespace, and frozen science/config checks were executed.
- No prohibited live or scientific operation was performed.
- The complete pytest suite was attempted twice but did not produce a terminal
  result: the first run timed out after 604 seconds with no failure output; the
  second was stopped after about 11 minutes, also with no failure output, to
  avoid indefinite waiting. This is the only remaining verification concern.

## Verification evidence

- `python -m pytest tests/test_v4_attempt02_authorization.py -q -p no:cacheprovider`
  — 21 passed.
- `python -m pytest tests/test_v4_authorization.py tests/test_v4_execution_governance.py tests/test_v4_attempt02_authorization.py -q -p no:cacheprovider`
  — 136 passed.
- Budget/storage matrix covering `test_storage_audit_refactor.py`,
  `test_storage_science_lock_refactor.py`, `test_artifact_storage_refactor.py`,
  two resource-governance tests, and attempt-02 tests — 54 passed.
- `ruff check georeliab_mve\\v4_authorization.py tests\\test_v4_attempt02_authorization.py`
  — passed.
- `python -m py_compile` for all tracked Python files — passed.
- `bash -n` for all tracked shell scripts — passed.
- `git diff --check` — passed.
- Science-lock/config zero-diff against the exact parent for
  `configs/georeliab_v4_protocol.toml`, `docs/GEORELIAB_V4_PROTOCOL.md`,
  `georeliab_mve/v4_science_lock.py`, `configs/dual_mve_protocol.toml`, and
  `configs/a100_real_mve_overlay.toml` — passed.
- `ruff check .` — not clean because the exact parent already contains 51
  unrelated Ruff findings outside the Task 2 files. No such finding exists in
  either modified Python file.

## Remaining risks

- The repository-wide pytest suite has no terminal pass/fail result within the
  allotted waits. Focused and related regression matrices are green, and neither
  incomplete full-suite run emitted a failure before termination.
- Live hardware and frozen-environment mapping evidence is intentionally absent
  from Task 2 and remains Task 3 work.

## Review change round

Review source:
`.superpowers/sdd/v4-gpu-blocked-semantics-attempt-02/task-2-review-report.md`.

Both HIGH findings and the one LOW finding are addressed:

- Resource validation now derives the immutable attempt root from the snapshot
  path and reuses the same production path-containment, forbidden-source, schema,
  and digest helpers used during materialization. Model/config, DTU archives,
  environment locks, split, fog, science lock, source-state inventory, source
  manifest, and generated state/schedule bindings are revalidated. Rehashed paths
  under `.pytest_cache`, `attempt-01`, or old authorization artifact names fail
  closed.
- The default frozen-environment Torch subprocess now obtains logical `cuda:0`'s
  UUID directly from `torch.cuda.get_device_properties(0).uuid`. A wrong or
  unavailable logical UUID returns `V4_GPU_TORCH_PROBE_DEVICE_MISMATCH`. The
  post-probe physical sample contributes only MIG/ECC health and residual-process
  evidence; it can no longer supply the logical-device UUID/model/memory proof.
- The test module no longer shadows pytest's built-in `tmp_path` fixture or
  writes routine test artifacts below the repository. Every test uses
  pytest-managed temporary storage.

Review-round regression evidence:

- Attempt-02 tests: 47 passed, including 24 production binding/source-policy
  combinations and wrong/unavailable logical UUID cases.
- Authorization/governance plus attempt-02 matrix: 162 passed.
- Budget/storage plus attempt-02 matrix: 80 passed.
- Focused Ruff, py_compile, and `git diff --check`: passed.
