# GeoReliab v4 GPU selection Attempt-04

Attempt-04 completed the resource gate and the three-sample GPU selection gate,
then issued an execution authorization without starting the MVE. The immutable
runtime evidence in this directory was copied byte-for-byte from:

```text
/srv/private/smli/GeoReliab/authorization-attempts/attempt-04/
```

The canonical terminal state is:

```text
V4_MVE_EXECUTION_AUTHORIZED
GPU_SELECTION_ATTEMPT=04
SELECTED_GPU_UUID=GPU-6ae218e6-3d51-b748-e308-1f0509e87886
SELECTED_GPU_PCI_BUS_ID=00000000:4D:00.0
SELECTED_GPU_INDEX=0
RECTIFIED_MEMBER_COUNT=960
NO_SCIENTIFIC_RESULT
```

The selected device is an `NVIDIA A100 80GB PCIe`. Three inventories, separated
by the frozen five-second interval, found a stable UUID/index/PCI mapping, zero
compute processes, zero utilization, 14 MiB used memory, MIG disabled, Default
compute mode, and healthy ECC state. The selection is bound by UUID and PCI bus
ID; index `0` records only the mapping observed during this attempt.

The authorization binds the 400-unit schedule, 960-member Rectified closure,
frozen dual-model set, scientific anchor, budgets, storage limits, runtime paths,
and atomic finalizer. It does not authorize fallback, device switching, matrix
expansion, UAVLight, v4.1, a third model, or a second corruption family.

No Torch probe, checkpoint load, model load, model forward, scientific metric,
dispatcher call, execution lock, GPU inference ledger, run directory, scientific
artifact, or MVE unit was created. Formal execution requires a later independent
user command and must fail closed if this exact device is no longer eligible.

The authorization tooling identity is commit
`05fc40640fd5dcbf64e20220cbe50049fa1968de`, tree
`a28db8d3aba73fc7b5a159fd71ef1d11d41acde9`. The evidence-archive commit that
contains this directory does not grant new or expanded execution authority.
