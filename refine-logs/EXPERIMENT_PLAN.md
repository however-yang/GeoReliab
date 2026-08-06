---
status: supporting
authority: GEORELIAB_TASKBOOK.md
execution_entry: false
superseded_by: null
---

# GeoReliab v2.2 Experiment Plan

## Role and current state

This file is the claim-driven experiment specification for the single authoritative [mainline](../GEORELIAB_TASKBOOK.md#v2.2). It is not an execution entry and it cannot authorize compute. The current state is:

- ATTEMPT05_TERMINAL_INFRASTRUCTURE_FAILURE
- PARTIAL_CORPUS_NOT_RESUMABLE
- RECOVERY_EXECUTOR_NOT_QUALIFIED
- NO_SCIENTIFIC_RESULT

Attempt-05 outputs, including the historical 199/400 materialization record, are excluded from all Pilot, confirmation, statistical, and claim decisions.

## Main claim map

| Claim | Evidence required | Decision boundary |
|---|---|---|
| C1: native confidence can rank degradation and task failure | CRR/SFR on paired counterfactuals, ranking metrics, and model-direction agreement | Ranking must be separated from warning; no method claim follows from ranking alone |
| C2: ranking ability can differ from failure-warning ability | Boundary Lag, late-warning proportion, paired failure cases, and LOSO stability | A positive warning gap must be demonstrated before method development |
| C3: scene reliability can transfer from point confidence to task failure | Pose/fusion failure linkage on the frozen task axis | The task-level result must be evaluated independently of point-level calibration |
| C4: failure awareness can support admission or defer decisions | Scoped warning evidence after C1/C2 are confirmed | No admission-control method is designed before the warning gap is established |

Anti-claims: this project does not train a confidence head, modify VGGT, learn uncertainty, replace the native signal with a new estimator, or optimize thresholds after seeing formal outcomes.

## Run order

The order is a hard gate sequence, not a suggestion.

### Block 0 — Infrastructure qualification

Purpose: establish durable, auditable execution before collecting development or formal evidence.

1. Qualify the locked toolchain and Ruff environment.
2. Run the Gate 1 CPU fault matrix. Every injected failure must classify deterministically as COMPLETE, SAFE_RETRY, QUARANTINED, or FATAL_IDENTITY_MISMATCH; UNKNOWN and the old broad audit reason are forbidden in new evidence.
3. Obtain separate authorization for the Gate 2 12-unit non-scientific GPU recovery smoke.
4. Stop fail-closed on any qualification, identity, budget, or evidence-integrity failure.

Block 0 creates engineering evidence only. It never creates a paper result, scientific marker, or formal claim.

### Block 1 — Fresh development Pilot

Prerequisites: Gate 1 and Gate 2 pass, a clean canonical worktree is identified, and a new explicit execution authorization is recorded.

- Use a fresh run and artifact root; do not read or reference Attempt-05 predictions.
- Use two frozen models, the frozen adapters and protocol, three preregistered scenes, all ten states, and 60 units total.
- Evaluate native warning score against pose failure. Report development-only ranking, warning-gap, model-agreement, LOSO, and failure-taxonomy evidence.
- Mark every output DEVELOPMENT_EVIDENCE_ONLY and NO_FORMAL_SCIENTIFIC_RESULT.
- Do not write MVE_FINALIZED or any formal scientific marker.

The Pilot decision is GO, INCONCLUSIVE, NO-GO, or EXECUTION_BLOCKED. A NO-GO ends the full-run route. An INCONCLUSIVE result may request a two-scene extension only after a new authorization.

### Block 2 — Confirmation preparation

Only a Pilot GO or an explicitly permitted inconclusive extension can open this block.

- Run the preregistered synthetic power design without using Attempt-05 or Pilot outcomes as tuning data.
- Freeze the confirmation scope manifest, selector version, raw/semantic/domain-separated schedule identities, seed design, and union/disjointness proof.
- Prepare the 17-scene and 15-scene alternatives and their expected unit counts.
- Do not start full confirmation until the power and scope gates are closed and a new budget contract is present.

### Block 3 — Formal confirmation

- Run only the preregistered confirmation scope.
- Generate StaticRank, CRR, RWG, SFR, and Boundary Lag evidence through the locked evaluator.
- Keep development scenes outside the confirmatory inventory.
- Allow recovery sessions only within the same Attempt-06 identity through SameAttemptSessionUnionManifest; cross-Attempt unions and Attempt-05 references are rejected.
- The finalizer must verify identity, hashes, transaction closure, coverage, and scope before any formal scientific result is published.

### Block 4 — Failure Case Taxonomy

Every development and confirmation report should classify representative cases:

- Type A: ranking is correct but warning fails — high confidence with severe task failure.
- Type B: warning is early or conservative — confidence falls before task failure.
- Type C: correct rejection — confidence falls and failure occurs.

Taxonomy is diagnostic evidence and cannot replace preregistered quantitative gates.

### Block 5 — Method development

Only after a positive, reproducible warning gap is confirmed may a warning-aware repair, admission policy, or defer mechanism be designed. The method cannot be proposed first and justified by a later search for an effect.

## Development Pilot evidence contract

The Pilot must answer only:

1. Does native reliability rank pose failure above chance?
2. Is there a positive Boundary Lag / late-warning pattern?
3. Do both frozen models agree in direction?
4. Does the pattern survive leave-one-scene-out analysis and the taxonomy review?

Minimum Pilot thresholds are specified in idea-stage/PILOT_PLAN.md. Metrics are viewed only for development evidence; no formal confirmation decision may use an unregistered threshold adjustment.

## Confirmation evidence contract

Formal evidence must retain the paired counterfactual DTU axis and the TartanAir severity axis:

- CRR and SFR belong to the paired counterfactual axis.
- Boundary Lag belongs to the ordered severity axis.
- The complete 20-scene/400-unit schedule remains a frozen candidate inventory, not the current Pilot.
- The final confirmatory scope is the approved 17-scene or 15-scene manifest, not an opportunistic subset.

## Resource and risk controls

- No automatic progression from qualification to Pilot or from Pilot to confirmation.
- GPU and storage budgets are cumulative across historical Attempt-05 accounting, recovery smoke, Pilot/extension, and confirmation.
- Missing telemetry fails closed; wall time, GPU-active time, card hours, and storage are recorded separately.
- No new dependency, code path, protocol TOML, data split, corruption rule, threshold, adapter, or model is introduced by this document.
- If a Gate fails, preserve NO_SCIENTIFIC_RESULT and produce a machine-readable blocker; do not infer a scientific no-go from infrastructure failure.

## Execution checklist

- [ ] Taskbook is the only execution entry.
- [ ] Gate 1 CPU matrix is complete and fully classified.
- [ ] Gate 2 GPU smoke has explicit authorization and passes exactly-once checks.
- [ ] Pilot manifest and 60-unit inventory are frozen before execution.
- [ ] Pilot outputs are isolated as development evidence.
- [ ] Power and confirmation scope manifests are frozen before formal execution.
- [ ] Finalizer verifies the complete scoped inventory before publication.
- [ ] NO_SCIENTIFIC_RESULT remains true until formal finalization.