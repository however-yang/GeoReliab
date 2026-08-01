# Current v4 GPU authorization evidence

The canonical authorization attempt is the fail-closed preflight under [`fa9a784c449303de0bb4ba67db92d0fbd418e10b`](./fa9a784c449303de0bb4ba67db92d0fbd418e10b/README.md).

The `5d862c87e8e5c391ed1b53280b0dd9ffd8236bdf` directory is retained as superseded engineering evidence. Its observed external-process blocker was valid, but later whole-branch review strengthened the future PASS path by fixing the sample interval, driver stability gate, and exact reason preservation. It must not be used as the current authorization decision.

No directory in this tree contains a PASS receipt, execution authorization, scientific schedule, model output, or scientific result.
