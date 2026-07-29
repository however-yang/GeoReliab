# GeoReliab Storage Architecture Refactor Protocol

## Authority and scope

This document governs the storage-only refactor based on project commit
`f5397b25806dcbf5b527b83c836b6c5f344122ae`.

The refactor may change:

- byte-level NPZ encoding while preserving every decoded array name, dtype,
  shape, and C-order byte sequence;
- content-addressed placement of shared DTU GT payloads;
- retention of MASt3R working caches and other explicitly classified
  ephemeral files;
- storage projections, wall-runtime reporting, GPU-inference accounting, and
  orchestration needed to enforce those controls;
- the non-scientific P2-A canary and superseded engineering archive.

The refactor must not change:

- the scientific protocol, thresholds, split, test scenes, corruption
  parameters, severities, view selection, or frozen schedule sizes;
- model checkpoints, upstream source commits, calibration artifacts, rendered
  scientific inputs, or evidence semantics;
- Artifact Contract v1.1 top-level fields;
- any P3/P4/P5/P6 scientific result from another project commit.

No model inference may start from this refactor until an operator has supplied
a fresh, explicit GPU selection for the exact deployed commit. A GPU choice
from an earlier run is not reusable.

## Science lock

The following SHA-256 values are calculated from the exact base-commit blobs
after canonicalizing text line endings to LF. This is equivalent to the Git
blob content on the A100 checkout and prevents CRLF/LF checkout policy from
changing the lock. `georeliab_mve.science_lock` validates the canonical text
while still failing closed on content changes.

| Locked input | SHA-256 |
| --- | --- |
| `configs/dual_mve_protocol.toml` | `91eddc87c3bfdf2a6de413174dc758163c11d7c37e224b1deb4ac23288fafb6c` |
| `configs/a100_real_mve_overlay.toml` | `c65ef97684adeda6b1bba8d8c152eb559df44367bc35973f549d7e8049136011` |
| `georeliab_mve/splits.py` | `a270f114a649941abed5123a918e238273b8bc63ee9b9492509d428a6081a213` |
| `georeliab_mve/preparation_round2.py` | `304e60b50368354fa9623f7145832c140fb45a2d663a26c19f638c22119d3359` |
| `georeliab_mve/gates.py` | `cb77a1c37f30e813d86097722874cda118b34aa9c9d0bfa084d0950d752d5c51` |

These locks cover the frozen thresholds/budgets, A100 resource overlay,
scene/view split construction, corruption calibration implementation, and
gate semantics. Changing any lock requires a new protocol hash and is outside
this refactor.

## Storage levels

- **L0 Immutable Evidence**: manifests, hashes, JSON evidence, audit records,
  frozen configurations, shared GT, receipts, and final decision artifacts.
- **L1 Scientific Cache**: geometry, raw confidence, valid mask, dense audit,
  and zero-update subset NPZ payloads. They use deterministic lossless
  compression without dtype conversion.
- **L2 Ephemeral**: verified sparse download indexes, `.partial` files,
  per-item MASt3R work caches, zero-update adapter intermediates, and
  unreferenced diagnostics. Removal requires a retention receipt and must not
  remove an input still referenced before P6.

Frozen smoke/test PNGs and prepared arrays are retained through P6 because
the schedule and P3 audit validate them directly.

## Fail-closed apply protocol

`python -m georeliab_mve storage-audit` is read-only by default. It writes
`artifacts/storage_before.json` and `artifacts/storage_plan.json`. Mutation is
accepted only with both:

    --apply-plan <storage_plan.json>
    --expected-plan-sha256 <exact digest>

Every target is resolved below the recorded runtime root. Transformations use
`.partial`, validate semantic equivalence or archive inventories, atomically
replace the destination, and then write a receipt. Original content is not
removed before validation succeeds.

## Resource gates

R1 uses the complete worst path:

    current logical bytes
    + remaining P2 retained bytes
    + P3 retained bytes
    + conditional P5 retained bytes
    + one sequential-task temporary peak
    + 10% reserve

R1 PASS requires the projection to be below 900 GB; the immutable hard limit
remains 1 TB. Remaining-retained target is below 500 GB and the reporting
target is below 300 GB.

GPU accounting uses validated `PredictionArtifact.runtime_seconds` or the
explicit `gpu_inference_seconds` ledger field. End-to-end operations stay in
`wall_runtime_seconds`. The 50 GPU-hour gate uses only inference time.

## Rollout boundary

The old f539 P1 and 75 P2 bundles are superseded engineering evidence. They
must be archived with a member SHA inventory and must not be imported into the
new commit's canonical stage. The new commit may reuse only validated P0
materialization and corruption artifacts. It stops at
`GPU_SELECTION_REQUIRED` before P1 unless the user explicitly selects a GPU.
