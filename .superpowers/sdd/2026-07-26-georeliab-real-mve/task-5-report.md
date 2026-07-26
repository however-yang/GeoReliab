# Task 5 Report — Resumable Runner, Preflight, Budget, and CLI

Status: implemented, locally verified, and ready for independent review.

## Commits

- `983b538` — initial resumable runner governance and frozen schedules.
- `bf68876` — frozen-DTU production audit binding for runner artifacts.
- `3046b013e6b079f24e45841d3cd9c3b79743c0f9` — final fail-closed orchestration, P3/P5 evidence integration, worker isolation, budget, CLI, and regression fixes.
- `7157c9c` — independent-review fixes for preflight namespaces, runtime-tree hygiene, and explicit terminal P5 invalid evidence.

## Files and public surfaces

- `georeliab_mve/runner.py`
  - exact P1/P2/P3/P5 schedule construction and deterministic hash sharding;
  - immutable test-stage fingerprints and 400-run completeness checks;
  - atomic `.partial` bundle commits, validated resume/skip, portable claims, stale-lock recovery, append-safe ledgers, and artifact digests;
  - frozen VGGT/MASt3R environment workers with in-process checkpoint reuse per model worker;
  - one-scene repeated preflight, hard 50 GPU-hour/1 TB budget checks, downstream evidence, zero-update evidence, and native-gate recomputation;
  - `preflight-real` and `run-georeliab` CLI implementation.
- `georeliab_mve/runner_audit.py`
  - production binding from a v1.1 prediction artifact to frozen DTU GT, observability mask, Sim(3), dense audit, and validated scene summary.
- `georeliab_mve/cli.py`
  - dispatches the two real runner commands through the existing package entry point.
- `tests/test_runner_task5.py`
  - schedules, sharding, smoke validity, atomicity/resume, conflicts, invalid evidence, locks, fingerprints, budgets, P4/P5 binding, full 400+6+480 evidence chain, repeatability, worker isolation/logging, and CLI.
- `tests/test_runner_audit_task5.py`
  - frozen DTU production audit and invalid-prediction behavior.

## Frozen execution and evidence design

- P1/P2/P3 canonical counts are 8/200/400; P5 is 480 six-view subset runs and remains inaccessible until a digest-bound P4 native-confidence decision passes.
- `sample_key` and ordered rendered-PNG digests match the Task 3 evidence loader. Scientific completeness is always evaluated against both models even when one model or one shard is dispatched.
- REAL/SMOKE execution requires a clean project worktree, exact source commit/tree, upstream/checkpoint/config/environment hashes, verified prepared inputs, and an output root exactly equal to the overlay `runtime.root`; every `/home` output root is refused.
- Model processes run in their frozen Python/Torch environments. A worker loads each upstream checkpoint once and reuses it across schedule items while keeping every item’s output/cache under its own atomic partial bundle.
- Complete artifacts are skipped only after contract, linkage, stage-fingerprint, and digest validation. Invalid predictions remain explicit schedule completions and retain their failure evidence.
- Test and zero-update evidence are bound to the immutable P3 stage freeze. The P4 decision is recomputed from current P3 evidence, so a caller cannot inject or edit PASS. Zero-update result and subset NPZ digests are both bound by `stage_item.json`.
- Claims default to a six-hour stale window. Ledger locks recover only after a 60-second stale threshold; each row records state, attempt, timestamp, runtime, peak memory, reason, artifact digests, and bytes.
- `preflight-real/repeat-a` and `repeat-b` are the only controlled subdirectories allowed below the frozen runtime root. Runtime `stage`, preflight, prepared/rendered, manifest/evidence, bare-repository, and detached-worktree directories are ignored so runner outputs cannot dirty the source-tree provenance guard.
- P5 aggregation iterates the exact expected 480-item schedule. Missing uncommitted items remain resumable; digest/linkage conflicts hard-fail; any committed invalid subset creates a digest-bound `P5_INVALID_SUBSET_PREDICTION` terminal artifact and machine-readable FAIL summary.
- Successful and failed frozen-environment worker stdout/stderr are atomically retained with SHA-256 in machine-readable summaries.
- Budget estimation uses observed earlier-stage ledgers and materialized output size. Missing estimates block P2/P3/P5, cumulative use is charged, and the 400-run test grid cannot be reduced to fit.

## RED/GREEN evidence

- RED: the first independent review rejected the initial runner because stage fingerprints were incomplete, adapter outputs escaped atomic bundles, default audit was a fixture, model checkpoints could reload per item, P3 did not reach the Task 3 loader, and native/P5 evidence was caller-trustable.
- RED: the first six-view zero-update integration run failed because the canonical RGB digest incorrectly required eight views.
- GREEN: the runner now binds all frozen artifacts/config hashes, writes model payloads inside the atomic bundle, uses the production DTU audit, reuses upstream models, reaches the Task 3 loader with exact 400 P3 bundles plus six downstream and six zero-update evidence records, recomputes P4, and accepts canonical six-view subset manifests.
- GREEN: stale ledger recovery, `/home` refusal, exact overlay-root matching, worker log persistence, invalid zero-update resume, and zero-update tamper checks have dedicated regression assertions.

## Verification

- `python -m pytest tests\test_runner_task5.py tests\test_runner_audit_task5.py -q` → 21 passed in 531.1s; one third-party `pytest-asyncio` deprecation warning.
- `python -m pytest -q --basetemp C:\tmp\georeliab-full-final-20260727-0411` → 181 passed in 655.9s; one third-party `pytest-asyncio` deprecation warning. A prior default-basetemp attempt was discarded after a host/temp-permission process interruption without a pytest traceback; explicit isolated basetemp produced the authoritative result.
- `python -m compileall -q georeliab_mve tests` → pass.
- `git diff --check` → pass; only Git LF→CRLF working-copy notices were emitted.

## Self-review and remaining concerns

- No scientific result has been generated. Local tests use synthetic artifacts and do not establish a GeoReliab claim.
- Official DTU/TartanAir acquisition, real A100 checkpoint inference, measured GPU-hours/peak memory, and the P0–P6 operator execution remain Task 6 and remote-run work.
- Successful worker logs are preserved, but external scheduler logs and detached-worktree lifecycle still need Task 6 scripts and operator documentation.
- The six-hour claim timeout assumes one model-condition item completes within six hours. If real MASt3R timings violate that assumption, add a claim heartbeat; do not shorten the frozen schedule or silently reclaim a live item.
- P4 failure must remain terminal for GeoReliab and must continue to short-circuit downstream/zero-update. A GeoReliab PASS remains pending Geometry evidence under the global selector.

## Independent review round 1

Status: `REQUEST_CHANGES`, all three blocking findings addressed in `7157c9c`.

- Fixed P1: repeat contexts now pass the root policy only for the two frozen preflight namespaces; regression tests exercise the actual policy instead of replacing it with a no-op.
- Fixed resume hygiene: all generated runtime roots are anchored in `.gitignore`, with `git check-ignore` regression coverage.
- Fixed P5 fail-closed semantics: relevant committed bundles can no longer be skipped on stage/linkage/digest errors, invalid subsets persist an immutable terminal FAIL record, and both tamper and normal 480-subset aggregate paths are tested.

## Independent review round 2

Status: `APPROVE`; zero critical/high/medium blockers. Task 6 deployment may begin.

- Reviewer confirmed the exact-root/repeat namespace policy, runtime ignore coverage, exact 480-item P5 schedule, tamper/linkage hard failure, invalid-subset terminal evidence, and machine-readable `run_stage` FAIL branches.
- Non-blocking note: terminal artifact creation and loading are directly tested, while the thin `run_stage()` exception-to-summary wrapper is code-reviewed rather than asserted separately. Preserve this as a future low-priority coverage addition.
- Remaining risks are remote-only: real A100 adapters/checkpoints, official data paths/hashes, measured budget, and detached-worktree/scheduler behavior.
