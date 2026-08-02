# GeoReliab v4 MVE Execution Attempt-05

Status: `IMPLEMENTATION_ONLY` until the immutable execution-start receipt is
validated. This document does not authorize a different GPU, schedule, model,
dataset, corruption, endpoint, or resource budget.

## Immutable upstream authorization

Attempt-05 consumes, but does not modify, the Attempt-04 authorization rooted
at commit `b7faf490280c50bc821ab157e035a68bc64a3090` and tree
`0a934f75c99ac0aa65467d6d424dd6964cb16cbc`. The authorized physical device is
identified by UUID `GPU-6ae218e6-3d51-b748-e308-1f0509e87886` and PCI bus ID
`00000000:4D:00.0`; index `0` is only the mapping recorded for this run. There
is no fallback, cross-device execution, or device switch.

The scientific anchor remains commit
`7381e60050143a78fca6a3ebde5706ae27d2c145`, tree
`f4e2b1104496c817693aaa5989d0276d2ebe03e9`. The frozen scientific schedule is
the 400-unit schedule with SHA-256
`47ed0464409d0189cb301930ecaf8db5b40b540ef0c5459dfea01fd92444a6c3`.

## Execution population

Attempt-05 first executes 40 calibration-L3 units: 20 scene-disjoint
calibration scenes for each of VGGT and MASt3R. These units freeze the native
warning Q90 scale and alarm threshold for each model. They consume runtime and
storage budget but never enter the scientific record set, bootstrap, confidence
intervals, or model comparison.

The scientific population is exactly 400 units:

```text
2 models x 20 test scenes x (L1-L7 + fog-s1-fog-s3)
```

L3 is executed once per model and test scene. It is the paired reference for
DTU lighting and the clean reference for fog. The 960-member Rectified closure
contains only the six non-reference illuminations `L1/L2/L4/L5/L6/L7`; it does
not remove L3 from the scientific schedule.

## Order and atomicity

The only legal order is:

```text
VGGT calibration-L3 20
MASt3R calibration-L3 20
freeze both Q90 calibrations
VGGT scientific 200
MASt3R scientific 200
read-only aggregate validation
read-only finalizer
```

Models and units are serial. Every unit is committed through a `.partial`
staging directory followed by validation and atomic promotion. A pre-existing
partial, conflicting artifact, adapter exception, linkage failure, or digest
failure stops the run. There is no automatic retry and no overwrite. A verified
complete artifact may be skipped read-only only when all frozen identities and
hashes match.

Contract-valid `invalid_prediction` output remains evidence as a failure and
uses the frozen finite higher-is-worse sentinel `1e12` for the native warning,
GT error, and selection score. Adapter exceptions, malformed artifacts, digest
drift, linkage failures, invalid calibration predictions, and partial outputs
still stop fail-closed; the sentinel is only for contract-valid scientific
invalid-output records.

## Metrics and evidence boundary

Each valid scientific unit produces PredictionArtifact v1.1, the frozen DTU
Sim(3)/observability/0.2 mm voxel audit, point F-score and StaticRank, and the
28-pair relative-pose endpoint. Pose reliability transfer is primary. The
existing eight-view fused geometry F-score is auxiliary actionability evidence
and cannot override the pose gate. No task-specific warning head, threshold
refit, extra model, extra state, seed expansion, or additional corruption is
permitted.

Exactly 400 TaskAuditRecords are required by the finalizer. Calibration units
are linked as calibration provenance only and cannot appear among those 400
records.

## Resource governance

The authorization stop lines are 35 GPU-hours and 150 GB of new logical or
allocated storage. Reaching either line stops before the next dispatch and
requires new user authorization. The 50 GPU-hour and 1 TB limits are disaster
circuit breakers, not automatically usable capacity.

The append-only hash-chained ledger records GPU inference time, wall time,
logical and allocated storage, completed/invalid/failed units, retry count,
peak memory, and artifact inventory identity. Recovery is allowed only at a
clean atomic boundary.

## Terminal states

Before finalization the scientific state is `NO_SCIENTIFIC_RESULT`. Legal
terminal states are:

```text
V4_MVE_COMPLETED / SCIENTIFIC_RESULT_AVAILABLE
V4_MVE_BLOCKED_BUDGET / NO_INVALID_PARTIAL_RESULT
V4_MVE_FAILED_WITH_REASON=<reason> / NO_INVALID_PARTIAL_RESULT
```

