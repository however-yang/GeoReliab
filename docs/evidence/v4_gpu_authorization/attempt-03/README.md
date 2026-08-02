# GeoReliab v4 GPU selection Attempt-03

Attempt-03 stopped at the resource revalidation gate. The immutable runtime
evidence is [v4-resource-revalidation.json](./v4-resource-revalidation.json),
whose file SHA-256 is
`f8e34fd93857f017bfb2d5495180fb57c3a868af4aacb19e7c4ad832c6e25131`.

The canonical terminal state is:

```text
GPU_SELECTION_BLOCKED_WITH_REASON=V4_RESOURCE_CLOSURE_REVALIDATION_FAILED
GPU_SELECTION_ATTEMPT=03
NO_SCIENTIFIC_RESULT
```

The underlying reason is `V4_ATTEMPT03_RESOURCE_HASH_MISMATCH`. Read-only
diagnosis found the frozen `SampleSet.zip` and `Points.zip` under the
overlay-declared `/srv/private/smli/GeoReliab/data` root, with the expected
SHA-256 values. The Attempt-03 tooling resolved those archives under
`/srv/private/smli/GeoReliab/cache`, where they do not exist. This is a
resource-binding implementation mismatch, not scientific evidence and not a
GPU eligibility result.

No NVIDIA inventory sample, Torch probe, model load, model forward, GPU
inference, execution receipt, execution authorization, run directory,
scientific artifact, or GPU ledger was created. The published FAIL artifact is
immutable and must not be overwritten or reused. A future attempt requires a
separate CPU-only resource-path repair followed by a new attempt ID.

The authorization tooling identity recorded by the runtime artifact is commit
`1d6d1d1d8400d2e1a857c1c6b4d4ada9849429c0`, tree
`e688a5b991e5e5837d5c14d45ed2887eb1730f34`. The evidence-archive commit that
contains this directory does not grant execution authority.
