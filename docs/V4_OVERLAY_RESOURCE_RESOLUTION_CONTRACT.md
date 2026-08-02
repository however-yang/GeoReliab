# GeoReliab v4 Overlay Resource Resolution Contract

Status: `IMPLEMENTATION_ONLY / GPU_SELECTION_REQUIRED / NO_SCIENTIFIC_RESULT`

This contract repairs the resource-path defect recorded by Attempt-03. It does
not reopen or supersede that attempt. The original Attempt-03 snapshot,
resource revalidation evidence, blocked decision, and historical
`docs/evidence/v4_gpu_authorization/CURRENT.md` remain immutable.

## Authority

The exact-base Git blob of
`configs/a100_real_mve_overlay.toml` is the only resource mapping authority.
The projected schema is `georeliab-v4-overlay-resource-map-1.0`.

- `runtime.data` is the declared root.
- The case-sensitive POSIX URL basename is the relative path.
- Expected byte length and SHA-256 come from the overlay and must equal the
  corresponding identity in `frozen_materialization.json`.
- Historical materialization `path` values are provenance only.
- Cache paths, defaults, environment guesses, recursive discovery, and
  first-name matches are never fallback sources.
- A CLI or environment path is only an equality assertion against the
  overlay-derived target.

The logical resources are exactly `dtu_sampleset` (`SampleSet.zip`) and
`dtu_points` (`Points.zip`).

## File identity

Resolution is POSIX-host-only. Each resource must stay within the real
declared root, resolve to a regular file, retain a stable device/inode/size and
mtime/ctime while hashing, and match the frozen size and SHA-256. Symlink
chains are recorded; an escape, loop, or broken target is rejected.

An exact `runtime.cache/<basename>` candidate is reported only as
`IGNORED_SHADOW`. Its bytes never affect the selected source or source hash
result.

## Evidence and scope

The CPU-only resolver writes signed JSON through
`.partial -> validation -> atomic directory rename` under:

`artifacts/v4-overlay-resource-resolution/<tooling_commit>/`

Failure produces no PASS artifact set. Neither resolver command invokes GPU
enumeration, Torch, model loading, inference, scientific metrics, GPU receipts,
execution authorization, or Attempt-04.
