# Task 1 implementation report: Artifact contract v1.1 and route governance

## Status

Completed and verified.

## Files changed

- `georeliab_mve/contracts.py`
- `georeliab_mve/gates.py`
- `tests/test_contracts.py`
- `tests/test_gates.py`
- `tests/test_contracts_v11.py`
- `tests/test_gates_v11.py`

## Design choices

- v1.0 readers normalize all public artifact types to v1.1 in memory; writers emit v1.1 only.
- `ScientificProvenance` is the typed real/smoke provenance representation. It records project commit/tree, model source, the required lock/manifest SHA-256 values, and DUSt3R/CroCo commits for MASt3R.
- Bundle validation loads local NPZ payloads fail-closed, validates required fields, shapes, finite data, raw-confidence/mask consistency, 2 mm labels, digests, and artifact linkage.
- `BLOCKED` is non-terminal. Any non-terminal route blocks final selection; a GeoReliab pass with Geometry blocked emits `GEORELIAB_PASS_PENDING_GEOMETRY` and `BLOCKED_PENDING_GEOMETRY`.

## TDD evidence

RED command:

```text
python -m pytest -q tests/test_contracts_v11.py tests/test_gates_v11.py
```

RED result: expected collection failures for missing `validate_artifact_bundle` and `SelectedTrack.BLOCKED_PENDING_GEOMETRY`.

GREEN focused command/result:

```text
python -m pytest -q --basetemp C:\Users\SZ597\AppData\Local\Temp\georeliab-task1-pytest tests/test_contracts_v11.py tests/test_gates_v11.py
12 passed
```

Full-suite command/result:

```text
python -m pytest -q --basetemp C:\Users\SZ597\AppData\Local\Temp\georeliab-task1-pytest
75 passed
```

Additional verification:

```text
python -m py_compile georeliab_mve\contracts.py georeliab_mve\gates.py
git diff --check
```

Both passed.

## Self-review

- Confirmed v1.0 migration coverage for `RunManifest`, `PredictionArtifact`, and `AuditRecord`.
- Confirmed real/smoke provenance and MASt3R dependency provenance validation.
- Confirmed required payload keys, shape/filtering drift, digest mismatch, and cross-sample linkage rejection.
- Confirmed fixture/smoke evidence cannot pass a scientific gate and every non-terminal selector case stays blocked.

## Concerns

- Bundle validation intentionally supports local `file://` URIs only because it must inspect NPZ bytes and payload arrays; remote artifact fetching remains out of scope for Task 1.
- The sandboxed pytest process cannot access its inherited temporary directory, so final pytest evidence used an isolated elevated temporary directory.

## Commit

Implementation commit: `ed28f81`.
