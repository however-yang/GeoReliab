# GeoReliab v4 GPU authorization audit

This directory records the fail-closed, CPU-only hardware preflight for authorization revision `5d862c87e8e5c391ed1b53280b0dd9ffd8236bdf` (tree `c08f947a12d4fcbd2b09d92c4c032a4a9ffa813a`). The scientific implementation anchor remains `7381e60050143a78fca6a3ebde5706ae27d2c145` / `f4e2b1104496c817693aaa5989d0276d2ebe03e9`.

Both samples resolved physical index 1 to the expected A100 UUID, but both also observed the same non-GeoReliab SURE-3D compute process. The preflight therefore stopped before either frozen Torch environment was probed. It generated only the hardware snapshot and blocked decision; it generated no PASS receipt, execution authorization, schedule, model output, or scientific metric.

The stage inventory remained 31 files with SHA-256 `e71d4f229ba3f3f1b63cfa0831ff5b540e9c993a229300b409e8a8c940f50147` before and after preflight. Formal GPU inference budget consumption is zero.

Terminal state:

```text
GPU_SELECTION_BLOCKED_WITH_REASON=V4_GPU_RECEIPT_NO_ACTIVE_COMPUTE_PROCESS
NO_SCIENTIFIC_RESULT
```

The earlier preflight under `5aa81f26cfc4f96fce507cfe9ef901d4353617d6` is retained remotely as superseded engineering evidence because its snapshot did not prove the CUDA runtime version. It is not admitted as terminal authorization evidence.
