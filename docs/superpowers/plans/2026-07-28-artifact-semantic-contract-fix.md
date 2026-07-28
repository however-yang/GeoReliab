# Artifact Semantic Contract Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the two deterministic P1 engineering blockers without changing any scientific protocol, split, corruption, threshold, model recipe, or scientific evidence.

**Architecture:** The frozen split/view manifest remains the single source of ordered DTU camera semantics. Materialization camera maps are validated by camera ID membership and then consumed in the frozen FPS sequence, so JSON object serialization order is irrelevant. Prepared artifacts continue recording the complete writer runtime fingerprint, but admission compares only the content-affecting writer recipe; deterministic re-decoding and byte digests remain the enforcement mechanism for runtime-dependent output differences.

**Tech Stack:** Python 3.10/3.14-compatible code, NumPy, pytest, JSON/NPY/NPZ artifact contracts.

## Global Constraints

- Base commit is `f73008172cfbac376fe3141e5fb0fdc0201f0494`.
- Do not modify `configs/`, the DTU split/view manifest, corruption parameters, scientific thresholds, gate logic, model inference recipes, checkpoints, or existing remote scientific evidence.
- Preserve fail-closed validation for missing/extra camera IDs, camera-source swaps, source-code recipe mismatches, algorithm-version mismatches, manifest digest mismatches, and decoded-content mismatches.
- DTU has no timestamp; do not invent one. Ordered identity is explicit as `view_id` plus its `fps_index` in the frozen split manifest.
- Writer runtime versions remain recorded in `producer.dependencies` for provenance, but consumer dependency versions are diagnostic and are not an admission equality condition.
- Add no dependencies.
- Follow test-driven development: each regression test must fail for the observed blocker before production code changes.
- Produce one Lore-format implementation commit after all tests pass.

---

### Task 1: Repair the two P1 artifact semantic contracts

**Files:**
- Modify: `georeliab_mve/audit.py`
- Modify: `georeliab_mve/preparation_round2.py`
- Test: `tests/test_audit_task3_round2.py`
- Test: `tests/test_prepared_writer_round4.py`
- Include: `docs/superpowers/plans/2026-07-28-artifact-semantic-contract-fix.md`

**Interfaces:**
- Consumes: `split_payload["views"][scene_id]` as the canonical ordered view sequence.
- Produces: `load_official_dtu_evidence(...)["view_ids"]` in frozen FPS order regardless of JSON map key order.
- Consumes: `producer` from `prepared-input-v2` and `tartanair-prepared-v2`.
- Produces: a strict producer-recipe validator that requires the writer version, source module/hash, and algorithm versions to match while treating `dependencies` as recorded provenance.

- [ ] **Step 1: Write the failing JSON reorder regression**

Add a test that rewrites a valid frozen materialization with the same camera entries in reverse JSON insertion order, updates only the materialization file serialization, and calls `load_official_dtu_evidence`. Assert that:

```python
assert evidence["view_ids"] == tuple(split_payload["views"]["26"])
assert evidence["gt_camera_centers"][:, 0].tolist() == pytest.approx(
    [float(view) for view in split_payload["views"]["26"]]
)
```

Also mutate the camera map to remove one expected ID and assert `AuditError` with a camera-membership mismatch. The production change caught by this test is reintroducing mapping iteration order as ordered identity or failing to reject missing IDs.

- [ ] **Step 2: Run the camera regression and verify RED**

Run:

```powershell
python -m pytest -q tests\test_audit_task3_round2.py -k "json_camera_order or camera_membership" --basetemp C:\tmp\georeliab-red-camera-contract
```

Expected: the reorder case fails with `materialized camera views do not match frozen FPS order`; the missing-ID case remains fail-closed.

- [ ] **Step 3: Implement minimal ordered camera identity**

In `load_official_dtu_evidence`, replace tuple comparison against `scene_record["cameras"]` iteration order with:

```python
camera_ids = {int(view_id) for view_id in scene_record["cameras"]}
if camera_ids != set(expected_views):
    raise AuditError("materialized camera IDs do not match frozen FPS views")
```

Continue iterating `expected_views` when reading assets and constructing camera centers/digests. Do not sort the semantic sequence and do not add a second ordering source.

- [ ] **Step 4: Run the camera regression and verify GREEN**

Run the command from Step 2 and then:

```powershell
python -m pytest -q tests\test_audit_task3_round2.py tests\test_runner_audit_task5.py --basetemp C:\tmp\georeliab-green-camera-contract
```

Expected: all selected tests pass.

- [ ] **Step 5: Write the failing cross-environment producer regression**

Create a valid `prepared-input-v2` fixture, deep-copy its producer evidence, and change only:

```python
payload["producer"]["dependencies"]["numpy"] = "2.2.6"
payload["producer"]["dependencies"]["pillow"] = "12.2.0"
```

Assert that `load_prepared_batch(..., expected_stage="smoke")` succeeds and returns the same record. Add parameterized negative cases changing `source_sha256`, `writer_version`, and one algorithm version; each must still raise `PreparationError` with a producer recipe mismatch. The production change caught by this test is restoring exact consumer/runtime equality or weakening content-affecting recipe validation.

- [ ] **Step 6: Run the producer regression and verify RED**

Run:

```powershell
python -m pytest -q tests\test_prepared_writer_round4.py -k "cross_environment or producer_recipe" --basetemp C:\tmp\georeliab-red-producer-contract
```

Expected: the dependency-only mismatch case fails with `prepared input producer/dependency recipe mismatch`.

- [ ] **Step 7: Implement strict recipe/runtime separation**

Add a private helper in `preparation_round2.py`:

```python
def _producer_recipe(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "writer_version": evidence.get("writer_version"),
        "source_module": evidence.get("source_module"),
        "source_sha256": evidence.get("source_sha256"),
        "algorithms": evidence.get("algorithms"),
    }
```

Validate that `producer` is a mapping, `dependencies` is a non-empty mapping retained in the payload, and `_producer_recipe(producer) == _producer_recipe(implementation_evidence())`. Apply the same rule to DTU and TartanAir prepared loaders. Do not remove or rewrite recorded dependency provenance. Keep all existing deterministic decode, source-asset, and digest checks unchanged.

- [ ] **Step 8: Run producer and related regressions GREEN**

Run:

```powershell
python -m pytest -q tests\test_prepared_writer_round4.py tests\test_preparation_round2.py tests\test_materialization_round3.py tests\test_runner_task5.py --basetemp C:\tmp\georeliab-green-producer-contract
```

Expected: all selected tests pass.

- [ ] **Step 9: Verify the whole project and invariants**

Run all test files, splitting only for timeout isolation, plus:

```powershell
python -m py_compile georeliab_mve\audit.py georeliab_mve\preparation_round2.py tests\test_audit_task3_round2.py tests\test_prepared_writer_round4.py
git diff --check
git diff -- configs
git status --short
```

Expected: 244 existing tests plus the new regressions pass; compile and diff checks pass; `git diff -- configs` is empty.

- [ ] **Step 10: Commit once using Lore format**

Commit the two implementation files, two regression-test files, and this plan with an intent-first Lore message. Record:

```text
Constraint: Scientific protocol, split, corruption parameters, thresholds, and evidence remain frozen
Rejected: Treat JSON object order as FPS order | canonical JSON serialization reorders numeric string keys
Rejected: Bind prepared artifacts to consumer dependency versions | two frozen model environments intentionally differ
Directive: Keep runtime dependencies as provenance; decoded bytes and content-affecting recipe fields remain fail-closed
Tested: Targeted RED/GREEN regressions, full pytest suite, py_compile, git diff --check
Not-tested: Remote A100 P0/P1 replay before deployment
```
