---
status: supporting
authority: GEORELIAB_TASKBOOK.md
execution_entry: false
superseded_by: null
---

# GeoReliab v2.2 Experiment Tracker

This file mirrors status only. It does not define protocol, thresholds, schedules, model choices, or execution authorization. The authoritative specification is the [Taskbook](../GEORELIAB_TASKBOOK.md#v2.2).

## Current terminal state

- ATTEMPT05_TERMINAL_INFRASTRUCTURE_FAILURE
- PARTIAL_CORPUS_NOT_RESUMABLE
- RECOVERY_EXECUTOR_NOT_QUALIFIED
- V4_ATTEMPT05_ROOT_CAUSE_UNOBSERVABLE_DUE_TO_EXCEPTION_COLLAPSE
- NO_SCIENTIFIC_RESULT

## Qualification and experiment gates

| Gate | Scope | Status | Evidence required | Authorization |
|---|---|---|---|---|
| G0 | clean canonical source and locked toolchain | BLOCKED / NOT_RUN | provenance, zero-drift manifest, Ruff/toolchain check | none |
| G1 | CPU fault matrix | BLOCKED / NOT_RUN | all cases classified, deterministic semantic result, no unknowns | none |
| G2 | 12-unit non-scientific GPU smoke | BLOCKED_UNTIL_G1 | exactly-once recovery, GPU0-only, no scientific marker | new explicit GPU/budget authorization |
| P1 | 3-scene development Pilot, 60 units | BLOCKED_UNTIL_G2 | ranking, warning gap, model agreement, LOSO and taxonomy | new explicit execution authorization |
| P1-EXT | two-scene development extension | NOT_AUTHORIZED | only after P1 INCONCLUSIVE and 15-scene power gate | new explicit execution authorization |
| C0 | synthetic power and confirmation scope freeze | BLOCKED_UNTIL_PILOT_GO | power manifest, 17/15 scope manifests, disjointness proof | new explicit budget contract |
| C1 | formal confirmation | NOT_AUTHORIZED | scoped finalizer, canonical closure, formal statistics | new explicit execution authorization |
| M1 | warning-aware method development | NOT_AUTHORIZED | confirmed positive warning gap | new scientific authorization |

## Monitoring contract

When a future authorized stage exists, report stage progress, elapsed wall time, cumulative materialization time, GPU0/GPU1 ownership, GPU-active hours, card hours, storage, invalid/duplicate/identity-mismatch counts, and terminal markers. Internal heartbeat may be 60 seconds; external reporting is hourly. Historical Attempt-05 accounting is never reset.

## Stop rules

- Do not interpret the historical 199/400 record as scientific progress.
- Do not resume Attempt-05 or combine it with a new Attempt.
- Do not automatically advance from one gate to another.
- Keep NO_SCIENTIFIC_RESULT until formal finalization.
- Preserve all frozen run, artifact, log, and ledger roots.