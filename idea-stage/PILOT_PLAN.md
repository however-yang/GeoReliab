---
status: supporting
authority: GEORELIAB_TASKBOOK.md
execution_entry: false
superseded_by: null
---

# GeoReliab v2.2 Development Pilot Plan

## Gate status

This is a development-only specification. It cannot authorize a run. The [authoritative Taskbook](../GEORELIAB_TASKBOOK.md#v2.2) is the only execution entry. The current status remains:

- ATTEMPT05_TERMINAL_INFRASTRUCTURE_FAILURE
- PARTIAL_CORPUS_NOT_RESUMABLE
- RECOVERY_EXECUTOR_NOT_QUALIFIED
- NO_SCIENTIFIC_RESULT

The Pilot is blocked until Gate 1 CPU fault-matrix qualification and Gate 2 12-unit GPU recovery qualification both pass. A new explicit GPU and budget authorization is required. Attempt-05 outputs, including its 199 partial units, are not inputs.

## Purpose

The Pilot tests whether a fresh, frozen native confidence signal exhibits:

1. ranking of pose failure above chance;
2. a positive failure-warning gap;
3. directionally consistent behavior across the two frozen models;
4. robustness that is not driven by one scene.

It is not a formal confirmation, a model-training exercise, or a license to alter the protocol.

## Frozen Pilot inventory

- Three preregistered scenes selected from the frozen schedule using the bound selector and schedule identity.
- Ten ordered states per scene.
- Two frozen models with their approved adapters.
- 3 scenes × 10 states × 2 models = 60 unique units.
- Full-state paired/severity coverage is retained; no convenience subset is permitted.
- A new Attempt/Pilot run and artifact root are mandatory.
- Each unit binds schedule identity, scene/state identity, model, adapter, split, GPU identity, and canonical output hash.
- Development scenes and outputs remain disjoint from any future confirmation inventory.

No Attempt-05 prediction, receipt, ledger segment, output hash, or intermediate metric may be used to fill or select Pilot units.

## Metrics and preregistered gates

### Ranking

Use native_warning_score to rank pose_failure.

- Compute AUROC over every model and selected-scene state set.
- Report the equal-weight macro AUROC across the two models.
- GO ranking requires macro AUROC at least 0.60 and each model strictly above 0.50.
- A missing positive/negative class makes the metric undefined and prevents a GO decision.

### Warning gap

Boundary Lag is alarm time minus failure-onset time; a positive value means a late warning.

- Report pooled median Boundary Lag.
- GO warning requires pooled median Boundary Lag greater than 0 and late-warning proportion at least 60 percent.
- Both models must have positive median lag.
- A missing failure group or undefined lag is not silently removed from the denominator.

### Agreement and robustness

- The two models must move in the same direction for every core metric; strict opposite signs are a model conflict. A zero effect is not a conflict, but it cannot satisfy a positive GO gate.
- Leave-one-scene-out subsets must keep macro AUROC above 0.50 and median Boundary Lag above 0.
- For three scenes, at least two of three scenes must be directionally consistent for both ranking and lag.
- Scene dominance is reported; the Pilot cannot GO if one scene explains the entire effect.
- Failure taxonomy cases must include Type A ranking-good/warning-bad, Type B early warning, and Type C correct rejection when present.

## Decision precedence

1. Any invalid, duplicate, overwrite, identity mismatch, incomplete canonical output, or coverage failure yields V4_PILOT_BLOCKED_EXECUTION.
2. Undefined ranking or warning metrics yield V4_PILOT_INCONCLUSIVE; no denominator deletion.
3. If macro AUROC is below 0.55, pooled median Boundary Lag is not positive, or the models conflict, publish V4_PILOT_SCIENTIFIC_NO_GO.
4. If all GO conditions and robustness conditions are met, publish V4_PILOT_GO_TO_FULL_MVE.
5. Otherwise publish V4_PILOT_INCONCLUSIVE.

All Pilot outcomes are DEVELOPMENT_EVIDENCE_ONLY and NO_FORMAL_SCIENTIFIC_RESULT. The Pilot finalizer must not write MVE_FINALIZED or a formal scientific marker.

## Extension rule

A two-scene extension is permitted only when:

- the three-scene result is INCONCLUSIVE rather than blocked or NO-GO;
- the 15-scene power gate has passed;
- a new explicit GPU, storage, and execution authorization exists;
- the extension manifest is frozen before any extension output is inspected.

The extension is five-scene development evidence in total (the original three plus two new scenes). It is the only allowed development expansion. A NO-GO permanently forbids the 400-unit full run for this route.

## Non-negotiable boundaries

- Do not train or fine-tune a confidence head.
- Do not modify VGGT, models, adapters, schedule, split, corruption, or thresholds.
- Do not use Attempt-05 partial outputs or formal confirmation outcomes for Pilot tuning.
- Do not inspect formal-set metrics before the finalizer.
- Do not auto-start the extension, formal confirmation, P3, Warning-aware method development, or any future stage.
- Do not emit scientific claims from Pilot evidence.

## Required machine-readable evidence

Before execution, freeze the PilotPartitionManifest with scene IDs, unit IDs, selector version, schedule identity, model/adapter identity, scope hash, and disjointness proof. During execution, record completion receipts, canonical hashes, invalid/duplicate counters, GPU ownership, active hours, card hours, wall time, storage, and resume sessions. After execution, publish only the development decision and its evidence manifest.