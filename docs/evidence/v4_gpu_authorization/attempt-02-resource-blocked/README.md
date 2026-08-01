# v4 attempt-02 resource-blocked evidence

This additive evidence freezes attempt-02 at its prerequisite production
resource gate. It is not a GPU-eligibility decision and must not be interpreted
as `V4_NO_ELIGIBLE_IDLE_GPU`: GPU selection was never run.

## Terminal

```text
GPU_SELECTION_BLOCKED_WITH_REASON=V4_RESOURCE_RECTIFIED_MEMBER_CLOSURE_UNAVAILABLE
NO_SCIENTIFIC_RESULT
```

The immutable tooling identity is commit
`982997038a0c9db19b9d1eb38910df82e3661e39`, tree
`7e69bf6f7167430ad3921bc2f4a73b6d9354d9ce`. The scientific anchor remains
commit `7381e60050143a78fca6a3ebde5706ae27d2c145`, tree
`f4e2b1104496c817693aaa5989d0276d2ebe03e9`.

## Why the attempt stopped

The frozen v4 inventory requires 1,120 official Rectified members (20 scenes ×
seven lighting states × eight views). The server holds 160 required L3 members,
but the remaining 960 L1/L2/L4–L7 members were not materialized. Two bounded
reads of the official Rectified central directory did not complete (304.0 and
588.9 seconds); no third attempt was made.

An offline recovery path was then checked. The canonical
`cache/Rectified.sparse-index.zip` had already been deleted by the verified
retention receipt at
`artifacts/storage-retention/actions/000003-56746f2b047b8132.json`. The only two
remaining files with that name are 17-byte pytest fixtures; their SHA does not
match the deleted canonical sparse index and their paths are forbidden for
production evidence. They were rejected.

Without the local offsets/sizes/CRCs for those missing members, the exact dual
150 GB storage gate, production 200-state inventory, and deterministic 400-unit
schedule cannot be established. The implementation therefore stopped before
creating the attempt root.

## Non-admission audit

- Attempt root or `.partial`: absent before and after.
- GPU preflight and GPU selection: zero invocations.
- Torch imports and mapping probes: zero.
- Checkpoint loads, model instantiations, and model forward calls: zero.
- Receipt and execution authorization: absent.
- Scientific schedule execution, metrics, artifacts, and results: absent.
- GeoReliab GPU inference ledger delta: zero.

See [resource-blocker.json](./resource-blocker.json) for the machine-readable
terminal and [audit.json](./audit.json) for source hashes, archive identity,
retention evidence, storage prediction, and the complete negative-execution
audit. [validation.json](./validation.json) records the focused/full test and
static-check matrix, including unchanged pre-existing failures.
