---
status: authoritative
authority: GEORELIAB_TASKBOOK.md
execution_entry: true
superseded_by: null
---

# GeoReliab v2.2 Mainline Specification

**Ranking Is Not Warning: Reliability Awareness Evaluation for 3D Reconstruction under Observation Degradation**

- **Version**: v2.2 Mainline Specification
- **Target**: CVPR 2027; mid-September 2026 arXiv placeholder is a hard gate
- **Current phase**: Phase 0 — Infrastructure Qualification
- **Current state**: ATTEMPT05_TERMINAL_INFRASTRUCTURE_FAILURE / PARTIAL_CORPUS_NOT_RESUMABLE / RECOVERY_EXECUTOR_NOT_QUALIFIED
- **Scientific state**: NO_SCIENTIFIC_RESULT
- **Execution entry**: this file is the only mainline execution authority; supporting documents may not redefine the protocol.

## 0. One-sentence positioning

GeoReliab is a **Reliability Awareness Evaluation Framework** that tests whether native confidence in existing 3D reconstruction systems knows when it should stop trusting itself.

Core proposition:

    confidence ranking ability != failure warning ability

This is not a new reliability model. It does not train a confidence head, learn uncertainty, modify VGGT/MASt3R/CUT3R, or optimize benchmark scores.

## 1. Claims and anti-claims

### Primary claim C1: ranking and warning can separate

Under fixed scene and geometry, native confidence may rank geometric quality correctly. When the same scene becomes harder, it may remain high while failing to warn before task-level failure.

### Supporting claim C2: point-to-task transfer

Aggregate point confidence into a scene reliability score and test transfer to pose/fusion failure. This enters method and paper claims only after a fresh pilot confirms a warning gap.

### Anti-claims

- GeoReliab is not the first 3D UQ, confidence-calibration, or corruption benchmark.
- It does not claim universal confidence failure or a first training-free internal score.
- Non-rejection is not robustness, and synthetic-only evidence is not a deployment guarantee.

## 2. Frozen protocol and metrics

### Two axes

- **Axis A — DTU paired counterfactual**: scene, geometry, camera, and view identity stay fixed while seven illumination conditions change. This is an unordered discrete pairing; lighting must not be treated as a severity ladder. It supports CRR/SFR.
- **Axis B — ordered severity**: TartanAir or one preregistered degradation family forms a severity 0→4 sequence. Boundary Lag is legal only on this axis.

### Frozen metrics

1. **CRR (Confidence Ranking Reliability)**: Spearman rho_s between confidence ranking and true geometric-error ranking; higher is better.
2. **SFR (Severe Failure Recall)**: fraction of task-level severe failures successfully warned by native confidence. Failure is defined by pose, fusion, or reconstruction events, not pixel error.
3. **Boundary Lag**: only on Axis B. If failure occurs at severity k and warning triggers at j, Lag = k - j; positive means delayed warning and negative means premature warning.

Thresholds, aggregation, bootstrap, and multiplicity rules must be preregistered before the relevant gate. Intermediate results may not be used to retune them.

## 3. Novelty boundaries NC1–NC4

- **NC1 — DTU counterfactual reliability protocol**: VGGT-UQ uses DTU but only a fixed L3 illumination; RealX3D has physical degradation but does not evaluate native-confidence warning and lacks the required full paired/GT coverage.
- **NC2 — Ranking != Warning**: the contribution is an operational warning-failure boundary, not a claim to have invented uncertainty; Ovadia is the principal shift-UQ boundary.
- **NC3 — point confidence -> task failure**: point confidence is tested as a predictor of pose/fusion failure, rather than only as a local uncertainty score.
- **NC4 — reliability-aware view admission**: RobustVGGT rejects wrong-scene/distractor views; GeoReliab judges whether correct-scene but degraded observations remain trustworthy.

[Novelty clearance](idea-stage/NOVELTY_CLEARANCE.md) records supporting search evidence and does not alter this execution boundary. See the [claim-driven experiment plan](refine-logs/EXPERIMENT_PLAN.md), [development Pilot plan](idea-stage/PILOT_PLAN.md), and [status tracker](refine-logs/EXPERIMENT_TRACKER.md) for non-authoritative mirrors.

## 4. Current phase and hard gates

### Phase 0 — Infrastructure Qualification (current)

The goal is to prove that long jobs preserve scientific identity; this phase produces no paper evidence.

1. **Toolchain qualification**: independently validate Ruff and required tooling without changing the scientific environment, lock files, or protocol.
2. **Gate 1 — CPU Fault Matrix**: every transaction boundary must classify as COMPLETE, SAFE_RETRY, QUARANTINED, or FATAL_IDENTITY_MISMATCH; UNKNOWN and AUDIT_OR_RECORD_FAILED are forbidden in new evidence.
3. **Gate 2 — 12-unit GPU smoke**: use a fresh non-scientific smoke with fixed GPU/model/adapter identity and controlled kill/resume; require exactly-once completion, no duplicate, no overwrite, and no identity mismatch.

Until Gate 1 and Gate 2 pass, do not start Development Pilot, Attempt-06, warning-aware methods, P3, or any automatic downstream stage.

### Phase 1 — Development Pilot

Only after both gates pass and new GPU/budget authorization is issued:

- fresh three-scene full-state pilot: 3 scenes x 10 states x 2 models = 60 units;
- two frozen models and independent run/artifact/ledger roots;
- no Attempt-05 prediction reuse;
- outputs are DEVELOPMENT_EVIDENCE_ONLY and NO_FORMAL_SCIENTIFIC_RESULT;
- inspect macro AUROC, Boundary Lag, late-warning proportion, model-direction agreement, and LOSO robustness.

Pilot outcomes are GO, INCONCLUSIVE, NO-GO, or V4_PILOT_BLOCKED_EXECUTION. Only an inconclusive pilot with new authorization may add two extension scenes; a no-go forbids the full confirmation run.

### Phase 2 — Confirmation Preparation

After pilot GO, perform synthetic power analysis, scene/scope freeze, confirmation manifests, disjointness proofs, and claim qualification. A 17-scene/15-scene scope is selected by the preregistered power gate; no full run starts before scope freeze.

### Phase 3 — Formal Confirmation

Materialize the formal scope from scratch. The finalizer must first verify unit identity, canonical hashes, transaction closure, and one science anchor, then generate StaticRank/CRR/RWG/SFR, Boundary Lag, and paper-claim qualification. Until finalization, remain NO_SCIENTIFIC_RESULT.

### Phase 4 — Method Development

Only after confirmation establishes a warning gap may warning-aware repair, admission control, or another method be designed. A method may not be proposed first and used to search for a confirming gap.

## 5. Frozen full schedule

The 20-scene/400-unit schedule remains a frozen full-confirmation candidate, not the current Phase 0 or Development Pilot. Its scene, model, adapter, split, corruption, threshold, and unit identity may not change before confirmation scope freeze.

Attempt-05's 199/400 is an unreusable infrastructure record. It may not be concatenated, analyzed, cited in a paper, used for a gate decision, or used to choose a method.

## 6. Failure Case Taxonomy

- **Type A — Ranking good, warning bad**: confidence ranks quality but misses a high-confidence geometry/task failure.
- **Type B — Early warning**: confidence drops before task failure, showing over-conservatism.
- **Type C — Correct rejection**: confidence drops and failure follows, showing useful rejection.

This taxonomy is qualitative diagnosis, not a new protocol metric and not a replacement for CRR/SFR/Lag gates.

## 7. Boundaries and responsibilities

The junior researcher owns frozen-protocol infrastructure, metric implementation, baseline reproduction, failure taxonomy, and evidence packaging. They must not modify models, train a network or confidence head, tune gates, inspect formal-confirmation intermediate metrics, or start downstream stages automatically.

Long-term relation:

    GeoReliab (reliability definition)
        -> SURE-3D (reliable medical 3D perception)
        -> SurgWorldModel (reliable world model)

## 8. Version history

| Version | State |
|---|---|
| v1 | learned reliability/corruption route; superseded |
| v2 | Ranking Is Not Warning, DTU pairing, three metrics, admission control |
| v2.1 | severity gap, two-axis separation, NC3, threat map, statistical boundary |
| v2.2 | current Mainline Specification; recovery qualification precedes Pilot and Confirmation |

All supporting documents, trackers, and historical reports must point to this file. This is the only source of truth for scientific intent and execution order.