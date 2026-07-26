# Task 6 Report — A100 deployment and fail-closed execution control

## Outcome

Task 6 is implementation-complete and independently approved. The repository now
contains one canonical A100 overlay, exact-commit deployment, P0–P6 launch/status
scripts, and a fail-closed final evidence path. Real P0–P6 execution is deliberately
not claimed in this report; it begins only after this exact commit is pushed and
deployed to `/srv/private/smli/GeoReliab`.

## Implemented files

- `configs/a100_real_mve_overlay.toml`
- `scripts/a100/common.sh`
- `scripts/a100/verify_prereqs.sh`
- `scripts/a100/deploy_commit.sh`
- `scripts/a100/launch_stage.sh`
- `scripts/a100/status.sh`
- `scripts/a100/finalize_p6.sh`
- `docs/A100_REAL_MVE_RUNBOOK.md`
- `tests/test_deployment_task6.py`
- `.gitignore`

## Controls closed during final review

- Environment-lock and deploy-manifest JSON are piped into immutable writers; empty
  provenance files can no longer be created by a disconnected heredoc.
- Every stage launch and P6 finalization re-checks exact project HEAD plus staged,
  unstaged, and untracked cleanliness.
- P3 requires the complete 200-item smoke schedule; P4 requires the complete
  400-item test schedule.
- P5 validates the bound native-confidence gate before launching either shard and
  short-circuits unless it is `PASS`.
- The 50 GPU-hour limit is checked before GPU stages. P1 uses a frozen 2 GPU-hour
  reservation; later stages use observed ledger timing to estimate the next frozen
  schedule. Missing estimates fail as `BLOCKED_RESOURCE_BUDGET`.
- Status accounting uses append-only ledger rows. Unreadable JSON, missing/boolean,
  negative, or non-finite runtime values make budget evidence `UNAVAILABLE`; P6
  rejects it.
- P6 requires terminal gates, exact 200/400 schedules, exact 480 zero-update schedule
  after P4 PASS, resource evidence within 50 GPU-hours and 1 TB, and all frozen
  provenance digests. A validated `P5_INVALID_SUBSET_PREDICTION` is preserved as a
  terminal FAIL path.
- The runbook no longer contains an alternate hand-written P6 bundle path.

## Verification evidence

- Full repository suite: 198 tests passed, 0 failed; 891.8 seconds.
- Final deployment suite after all fixes: 17 passed, 0 failed.
- All six `scripts/a100/*.sh` files passed `bash -n` on `a100-smli`.
- All 13 embedded Python heredocs compiled successfully.
- Changed Python files passed `python -m py_compile`.
- `git diff --check` passed; only Windows line-ending notices were emitted.
- Final independent code review: APPROVE, 0 issues.

## Remote readiness evidence

- SSH and the `/srv/private/smli/GeoReliab/git/GeoReliab.git` bare repository are
  reachable.
- Two idle A100 80 GB devices and about 1.7 TB free space were observed.
- Frozen VGGT/MASt3R/DUSt3R/CroCo commits, environments, checkpoints, and hashes
  matched the overlay.
- DTU SampleSet and Points are present; Rectified and TartanAir remain P0
  materialization work through the validated range/sparse acquisition path.
- `/home/smli` has insufficient scratch capacity and remains read-only; every cache,
  temporary file, log, data file, and artifact is redirected under `/srv`.

## Remaining execution risk

No scientific evidence has been produced yet. The next authorized step is to push
the final Task 6 commit, deploy its detached worktree, and run P0. P1 must not start
until P0 is terminal-ready; P2–P6 remain sequential and P5 remains conditional on
P4 PASS. Global track selection stays `BLOCKED_PENDING_GEOMETRY` even if GeoReliab
later passes.
