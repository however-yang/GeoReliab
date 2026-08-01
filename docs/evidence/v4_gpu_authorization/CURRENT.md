# Current v4 GPU authorization evidence

The canonical attempt-01 outcome is the corrected fail-closed [BLOCKED decision](./fa9a784c449303de0bb4ba67db92d0fbd418e10b-attempt-01-corrected/v4-preflight-decision.corrected.json), supported by its immutable [erratum](./fa9a784c449303de0bb4ba67db92d0fbd418e10b-attempt-01-corrected/erratum.json). The corrected terminal reason is `V4_GPU_NON_GEORELIAB_COMPUTE_PROCESS_PRESENT`.

The original `fa9a784c449303de0bb4ba67db92d0fbd418e10b` [hardware snapshot](./fa9a784c449303de0bb4ba67db92d0fbd418e10b/v4-hardware-preflight.json) and [blocked decision](./fa9a784c449303de0bb4ba67db92d0fbd418e10b/v4-preflight-decision.json) remain byte-for-byte integrity-preserved superseded history. Their legacy inverted `V4_GPU_RECEIPT_NO_ACTIVE_COMPUTE_PROCESS` value must not be used as new canonical evidence.

The `5d862c87e8e5c391ed1b53280b0dd9ffd8236bdf` [hardware snapshot](./5d862c87e8e5c391ed1b53280b0dd9ffd8236bdf/v4-hardware-preflight.json) and [blocked decision](./5d862c87e8e5c391ed1b53280b0dd9ffd8236bdf/v4-preflight-decision.json) are also retained unchanged as earlier superseded engineering evidence.

No directory in this tree contains a PASS receipt, execution authorization, scientific schedule, model output, scientific metric, or scientific result. Attempt-01 consumed zero GeoReliab GPU inference and zero GeoReliab GPU budget.
