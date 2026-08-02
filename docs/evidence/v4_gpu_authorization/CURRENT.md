# Current v4 GPU authorization evidence

The current Attempt-03 outcome is the immutable resource-gate
[BLOCKED evidence](./attempt-03/v4-resource-revalidation.json), with its
[derived audit](./attempt-03/audit.json). Its canonical terminal reason is
`V4_RESOURCE_CLOSURE_REVALIDATION_FAILED`; the underlying reason is
`V4_ATTEMPT03_RESOURCE_HASH_MISMATCH`.

Read-only diagnosis confirmed that the frozen `SampleSet.zip` and `Points.zip`
exist under the overlay-declared `data` root and match their expected hashes.
Attempt-03 tooling instead resolved them under `cache`. Consequently this is a
resource-binding implementation mismatch, not a GPU result and not scientific
evidence. No `nvidia-smi` inventory sample, PASS receipt, execution
authorization, GPU ledger, run, or scientific artifact was created. The
Attempt-03 evidence is terminal and must not be overwritten or retried under
the same attempt ID.

## Retained attempt-02 history

The attempt-02 outcome remains the additive resource-gate [BLOCKED evidence](./attempt-02-resource-blocked/resource-blocker.json), with the supporting [audit](./attempt-02-resource-blocked/audit.json). Its terminal reason is `V4_RESOURCE_RECTIFIED_MEMBER_CLOSURE_UNAVAILABLE`. GPU selection was not run, so this must not be reported as `V4_NO_ELIGIBLE_IDLE_GPU` and it grants no execution authority.

## Retained attempt-01 correction history

The canonical attempt-01 outcome is the corrected fail-closed [BLOCKED decision](./fa9a784c449303de0bb4ba67db92d0fbd418e10b-attempt-01-corrected/v4-preflight-decision.corrected.json), supported by its immutable [erratum](./fa9a784c449303de0bb4ba67db92d0fbd418e10b-attempt-01-corrected/erratum.json). The corrected terminal reason is `V4_GPU_NON_GEORELIAB_COMPUTE_PROCESS_PRESENT`.

The original `fa9a784c449303de0bb4ba67db92d0fbd418e10b` [hardware snapshot](./fa9a784c449303de0bb4ba67db92d0fbd418e10b/v4-hardware-preflight.json) and [blocked decision](./fa9a784c449303de0bb4ba67db92d0fbd418e10b/v4-preflight-decision.json) remain byte-for-byte integrity-preserved superseded history. Their legacy inverted `V4_GPU_RECEIPT_NO_ACTIVE_COMPUTE_PROCESS` value must not be used as new canonical evidence.

The `5d862c87e8e5c391ed1b53280b0dd9ffd8236bdf` [hardware snapshot](./5d862c87e8e5c391ed1b53280b0dd9ffd8236bdf/v4-hardware-preflight.json) and [blocked decision](./5d862c87e8e5c391ed1b53280b0dd9ffd8236bdf/v4-preflight-decision.json) are also retained unchanged as earlier superseded engineering evidence.

No directory in this tree contains a PASS receipt, execution authorization,
scientific schedule, model output, scientific metric, or scientific result.
Attempt-01 consumed zero GeoReliab GPU inference and zero GeoReliab GPU budget;
the attempt-02 resource-gate audit likewise records zero GPU inference ledger
delta and no GPU preflight invocation.
