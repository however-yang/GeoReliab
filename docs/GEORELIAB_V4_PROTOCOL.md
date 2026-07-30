# GeoReliab v4 protocol and claim reset

## Authority

**Title:** GeoReliab: Ranking Is Not Warning — Paired Counterfactual
Reliability and Task Transfer in Geometric Foundation Models

**Venue route:** v4 is the sole CVPR 2027 main line. The CVPR 2027
submission deadline is not yet official and remains `TBA`; no 2026 deadline
may be extrapolated into this protocol.

**Core proposition:** Point-level confidence may rank geometry errors
correctly within a fixed condition, yet fail to warn when the same scene
becomes harder and fail to transfer into a reliable warning of relative-pose
failure.

This document and
[`configs/georeliab_v4_protocol.toml`](../configs/georeliab_v4_protocol.toml)
are the v4 governance boundary. They authorize no GPU work and report no
scientific result. The immutable machine lock is implemented by
`georeliab_mve.v4_science_lock`.

## Claim-reset status

| Subject | Frozen status | Meaning |
| --- | --- | --- |
| v4 protocol | `GEORELIAB_V4_PROTOCOL_READY` | The protocol is frozen and may be implemented. |
| Initial executable state | `GPU_SELECTION_REQUIRED` | Every future GPU stage requires a new explicit physical-GPU selection. |
| Current scientific result | `NO_SCIENTIFIC_RESULT` | No v4 evidence has been admitted and no scientific conclusion exists. |
| old v1 route | `SUPERSEDED_BY_PRIOR_ART_CHANGE` | The prior claim was superseded because prior art changed the defensible claim. |
| old v1 result | `NO_SCIENTIFIC_RESULT` | v1 is not a scientific FAIL and must never be relabeled as one. |
| Geometry causal audit | independent backlog, non-blocking | It is outside the v4 MVE route. |
| Deformable World | independent backlog, non-blocking | It is outside the v4 MVE route. |

Old v1 configurations, lock digests, runtime artifacts, and evidence are
immutable historical records. Existing frozen model I/O adapters, checkpoints,
DTU geometry/cameras/masks/FPS, provenance, deterministic lossless storage,
atomic commits, and resource accounting may be reused. Data addressed by a
`PredictionArtifact` v1.1 may be reused only through a new v4 artifact or
evidence record that binds the source URI and SHA-256 digest to an explicit v4
origin containing the exact protocol ID, version, and hash. The raw v1.1
mapping/object, any old `AuditRecord`, and old evidence path/digest wrappers
remain forbidden, even when nested inside an otherwise valid v4 envelope.

## Frozen claims

- **C1 Paired counterfactual protocol:** hold scene, geometry, cameras and view
  identities fixed while changing only illumination or synthetic fog.
- **C2 Ranking–warning separation:** distinguish static point-level ranking
  from cross-condition task-level warning.
- **C3 Point-to-pose transfer:** measure whether native point confidence
  transfers to warning relative-pose failure.
- **C4 Reliability boundary map:** identify model-, task- and
  condition-specific silent-failure boundaries.
- **C5 Conditional method contribution:** only after MVE GO, create a separate
  protocol for View-Set Admission Control.

The following claims are excluded:

- first GFM UQ;
- first corruption benchmark;
- universal confidence failure;
- conformal guarantees;
- non-rejection means robustness;
- synthetic-only extrapolation to real deployment.

## Novelty boundary

The defensible contribution is the paired same-scene separation between
static point ranking, counterfactual warning, and transfer to relative-pose
failure. It is bounded by the following adjacent work:

- [Trust3R](https://arxiv.org/abs/2605.19539) covers probabilistic point
  UQ/ranking/risk-coverage/downstream weighting.
- [VGGT-UQ](https://arxiv.org/abs/2606.16479) and its
  [ISPRS Annals DOI page](https://doi.org/10.5194/isprs-annals-XI-2-2026-665-2026)
  cover static VGGT uncertainty on DTU.
- [E3D-Bench](https://arxiv.org/abs/2506.01933) is a breadth-first benchmark.
- [Can These Views Be One Scene?](https://arxiv.org/abs/2605.18754) covers
  invalid-view-set hallucination.
- [Ovadia et al.](https://arxiv.org/abs/1906.02530) is shift-UQ precedent.
- [UAVLight](https://arxiv.org/abs/2511.21565) is post-GO external paired
  illumination.
- [RL3DEdit](https://arxiv.org/abs/2603.03143) uses VGGT confidence maps and
  pose-estimation error jointly as rewards for 3D-consistent editing. v4 must
  not claim first confidence-to-pose use.
- The [official DTU page](https://roboimagedata.compute.dtu.dk/?page_id=36)
  provides seven lighting configurations, not an ordered severity ladder.

## Frozen MVE data and schedule

The MVE permits only:

- VGGT and MASt3R;
- DTU paired lighting;
- one synthetic fog family;
- TartanAir as physical sanity only.

UAVLight, a third model, learned heads, fine-tuning, new or learned scientific
adapters, threshold or seed searches, feature-map retention, and post-test
tuning are forbidden. This does not prohibit reuse of the frozen model I/O
adapters named above.

The frozen test scenes are:

`1, 9, 10, 11, 12, 13, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118`.

Each scene uses one ordered eight-view identity. Calibration, dev, reference,
and test splits must be deterministic, complete, scene-disjoint, exclude scans
4 and 15, and use `SHA256("GeoReliab-v4:scan<ID>")`.

DTU `L1`–`L7` are unordered discrete lighting conditions. Encoding or
inferring a severity order is forbidden. Synthetic fog is the only ordered
axis, and Boundary Lag is legal only on
`L3 -> fog-s1 -> fog-s2 -> fog-s3`.

The test schedule is exactly:

`20 scenes x 10 unique states x 2 models = 400` scientific units.

The ten states are `L1`–`L7` and `fog-s1`–`fog-s3`; `L3` is reused as the fog
clean reference and is not duplicated.

## Frozen tasks and metrics

### Point task

- Main loss: `1 - F-score@2mm`.
- Diagnostics: median GT error and F-score sensitivity at 1 mm and 5 mm.
- StaticRank: within-scene static Spearman ranking.

### Relative-pose task

All 28 unordered pairs from the eight views are used.

`pair_error = max(rotation geodesic error, translation-direction error)`.

- Main loss: `1 - AUC@10deg`.
- Diagnostics: `AUC@5deg` and `AUC@20deg`.
- A scene is a pose failure iff its median pair error is greater than 10
  degrees.
- The endpoint is blocked when fewer than eight distinct test scenes have pose
  failures.

### Native warning and paired metrics

The native warning is:

`median_over_views(Q90(higher-is-worse native point risk))`.

Its threshold is the model-specific calibration-L3 90th percentile and is
applied unchanged to every test state. No task-specific learned head is
allowed.

All paired changes are relative to L3. The frozen metrics are StaticRank, CRR,
RWG, SFR, Boundary Lag, nAURC, and TTG:

- CRR uses a scene-block bootstrap.
- Boundary Lag uses virtual alarm level 4 and excludes no-failure scenes from
  the main lag.
- `nAURC=(AURC_signal-AURC_oracle)/(AURC_random-AURC_oracle)`.
- `TTG_pose=nAURC_pose-nAURC_point`.

Invalid model outputs are retained and counted as failures.

## Frozen statistics and scientific gate

All main confidence intervals use 10,000 scene-level bootstrap resamples.
Holm correction applies within the two models and two main gate families.
Repeats are numerical-reproducibility checks only and are never independent
statistical observations.

Strong evidence requires either:

1. the ranking–warning branch: StaticRank lower bound `>=0.35`, CRR-pose upper
   bound `<=0.15`, and RWG lower bound `>=0.20`; or
2. the transfer branch: nAURC-point upper bound `<=0.50` and TTG-pose lower
   bound `>=0.20`;

and in both cases SFR-pose lower bound `>=0.30`.

The cross-model rule is one-strong-one-directional: the second model point
estimate must have the same direction, and its corrected confidence interval
must exclude a reverse effect below `-0.10`.

Real paired DTU evidence is required for GO. Synthetic-only evidence is
scientific NO-GO.

## Execution and resource governance

Every GPU stage starts at `GPU_SELECTION_REQUIRED`. Prior selection receipts
are not reusable. Exactly one physical GPU must be explicitly selected.
Models and items run sequentially. Resume is fail-closed; automatic fallback,
device switching, retry, grid reduction, or downstream advancement is
forbidden.

Authorization stops at any one of:

- 35 GPU-hours;
- 150 GB new logical storage;
- 150 GB new allocated storage.

The hard catastrophe fuse stops at either 50 GPU-hours or 1 TB storage. The
extra 15 GPU-hours and 850 GB above the authorization stop are not
automatically usable.

An exact-recipe repeat is allowed only when a main metric is within 0.02 of a
threshold, with at most two repeats. Repeats are excluded from confidence
interval sample size.

Final v4 decision statuses are:

- `MVE_GO_TO_EXTERNAL_VALIDATION`;
- `MVE_SCIENTIFIC_NO_GO`;
- `MVE_BLOCKED_ENDPOINT`;
- a specific `MVE_BLOCKED_*`.

No decision may automatically start v4.1, UAVLight, a P3-style continuation,
another paper route, plot generation, or claim generation.

## Conditional v4.1 after GO

Only `MVE_GO_TO_EXTERNAL_VALIDATION` may permit a new, separately
preregistered v4.1 protocol:

`native warning + preregistered cross-view residual => admit/defer/refuse`.

The fallback is fixed classical geometry. The comparators are always-admit,
native-only, residual-only, matched-random, and oracle.

UAVLight, Trust3R, a third model, and the fallback each require a new protocol,
budget, and GPU authorization. None may write back into the v4 MVE.

## Frozen figure plan

1. same-scene trajectory;
2. DTU paired silent failure;
3. fog Boundary Lag;
4. point-to-pose nAURC/TTG;
5. admission utility-coverage only after GO.

Figure 5 is illegal before GO.

## Internal dates

| Dates (2026) | Frozen activity |
| --- | --- |
| 7/30–8/04 | claim reset, novelty, protocol |
| 8/05–8/10 | pair identity, L1–L7, metric tests |
| 8/11–8/13 | ask GPU, then dual-model sanity |
| 8/14–8/21 | 400-unit MVE |
| 8/22–8/24 | statistics and gate |
| GO only: 8/25–9/15 | v4.1 and UAVLight |
| 9/15 | internal evidence freeze |

These dates are planning constraints, not permission to skip a gate or select
a GPU.
