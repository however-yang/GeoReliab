# GeoReliab v4 canonical GPU authorization audit

This directory records the final fail-closed, CPU-only hardware preflight for authorization revision `fa9a784c449303de0bb4ba67db92d0fbd418e10b` (tree `4c2c3bf2cdc0a72d5f3124eafe907bc02b0f412d`). The scientific implementation anchor remains `7381e60050143a78fca6a3ebde5706ae27d2c145` / `f4e2b1104496c817693aaa5989d0276d2ebe03e9`.

The CLI-enforced 5-second sampling interval produced samples at `10:53:16Z` and `10:53:22Z`. Both resolved physical index 1 to the expected A100 UUID with stable driver and CUDA evidence, but both observed the same non-GeoReliab SURE-3D compute process. Preflight therefore stopped before either frozen Torch environment was probed.

Only the hardware snapshot and blocked decision were generated. No PASS receipt, execution authorization, schedule, model output, scientific metric, or GPU-inference ledger entry was created. The stage inventory remained 31 files with SHA-256 `e71d4f229ba3f3f1b63cfa0831ff5b540e9c993a229300b409e8a8c940f50147` before and after preflight.

Terminal state:

```text
GPU_SELECTION_BLOCKED_WITH_REASON=V4_GPU_RECEIPT_NO_ACTIVE_COMPUTE_PROCESS
NO_SCIENTIFIC_RESULT
```
