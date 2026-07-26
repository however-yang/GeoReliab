from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from georeliab_mve import runner
from georeliab_mve.contracts import (
    AuditRecord,
    PredictionArtifact,
    RunManifest,
    RunMode,
    SampleKey,
    ScientificValidity,
    validate_artifact_bundle,
)


def _sha(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _minimal_root(tmp_path: Path, *, dev_count: int = 10) -> Path:
    root = tmp_path / "GeoReliab"
    (root / "prepared").mkdir(parents=True)
    (root / "rendered" / "smoke").mkdir(parents=True)
    (root / "rendered" / "test").mkdir(parents=True)
    (root / "manifests").mkdir()
    (root / "evidence").mkdir()
    splits = {
        "dev": list(range(201, 201 + dev_count)),
        "test": list(runner.TEST_SCENES),
        "calibration": list(range(301, 311)),
        "reference-token": list(range(401, 406)),
    }
    views = {str(scene): [1, 3, 5, 7, 9, 11, 13, 15] for split in splits.values() for scene in split}
    split_payload = {"schema_version": "dtu-preparation-v1", "splits": splits, "views": views}
    split_path = _write_json(root / "manifests" / "split_view_manifest.json", split_payload)
    corruption_path = _write_json(root / "manifests" / "corruption_calibration.json", {"schema_version": "corruption-calibration-v1"})
    for stage, scenes in (("smoke", splits["dev"]), ("test", splits["test"])):
        records = []
        for scene in scenes:
            for view in views[str(scene)]:
                key = f"dtu/{'dev' if stage == 'smoke' else 'test'}/scan{scene:03d}/fps8/clean/0/0"
                records.append(
                    {
                        "scene_id": scene,
                        "view_id": view,
                        "sample_key": key,
                        "raw_source_sha256": _sha(f"raw-{scene}-{view}"),
                        "gt_digest": _sha(f"gt-{scene}"),
                    }
                )
                for condition, severity in runner.CONDITIONS:
                    png = root / "rendered" / stage / f"scan{scene:03d}_view{view:03d}_{condition}_s{severity}.png"
                    data = f"{stage}-{scene}-{view}-{condition}-{severity}".encode("ascii")
                    png.write_bytes(data)
                    meta = {
                        "rendered_png_sha256": runner.sha256_file(png),
                        "raw_source_sha256": _sha(f"raw-{scene}-{view}"),
                        "gt_digest": _sha(f"gt-{scene}"),
                    }
                    _write_json(png.with_suffix(".json"), meta)
        _write_json(
            root / "prepared" / f"render_inputs_{stage}.json",
            {
                "schema_version": "prepared-input-v2",
                "stage": stage,
                "split": "dev" if stage == "smoke" else "test",
                "record_count": len(records),
                "split_view_manifest_sha256": runner.sha256_file(split_path),
                "materialization_sha256": _sha("materialized"),
                "records": records,
            },
        )
    _write_json(root / "evidence" / "tartanair_native_fog_sanity.json", {"passed": True})
    _write_json(root / "manifests" / "corruption_calibration_qa.json", {"passed": True})
    _write_json(root / "manifests" / "test_render_lock.json", {"schema_version": "test-render-lock-v1"})
    return root


def _fake_prediction(manifest: RunManifest, sample_key: SampleKey, output_dir: Path, *, invalid: bool = False) -> PredictionArtifact:
    output_dir.mkdir(parents=True, exist_ok=True)
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    camera = np.repeat(np.eye(4)[None, :, :], 8, axis=0)
    intrinsics = np.repeat(np.eye(3)[None, :, :], 8, axis=0)
    pixels = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    view_id = np.array([1, 3], dtype=np.int64)
    conf = np.array([2.0, 3.0], dtype=np.float64)
    mask = np.array([True, True])
    if invalid:
        points = np.empty((0, 3), dtype=np.float64)
        pixels = np.empty((0, 2), dtype=np.float64)
        view_id = np.empty((0,), dtype=np.int64)
        conf = np.empty((0,), dtype=np.float64)
        mask = np.empty((0,), dtype=bool)
    geo = output_dir / "geometry.npz"
    native = output_dir / "confidence.npz"
    valid = output_dir / "valid.npz"
    np.savez(geo, points_world=points, camera_c2w=camera, intrinsics=intrinsics, pixel_xy=pixels, view_id=view_id)
    np.savez(native, raw_confidence=conf)
    np.savez(valid, valid_mask=mask)
    return PredictionArtifact(
        manifest.run_id,
        str(sample_key),
        geo.as_uri(),
        native.as_uri(),
        valid.as_uri(),
        None,
        1.0,
        128.0,
        invalid,
        {
            "geometry_prediction_uri": runner.sha256_file(geo),
            "native_confidence_uri": runner.sha256_file(native),
            "valid_mask_uri": runner.sha256_file(valid),
        },
    )


class FakeAdapter:
    def __init__(self, invalid: bool = False):
        self.invalid = invalid

    def predict_sample(self, manifest, sample_key, rendered_views):
        assert len(rendered_views) == 8
        return _fake_prediction(manifest, sample_key, Path(rendered_views[0].png_path).parent / "adapter", invalid=self.invalid)


def test_schedule_counts_and_stable_shards(tmp_path: Path):
    root = _minimal_root(tmp_path)
    preflight = runner.build_schedule(root, "preflight", model="all")
    smoke = runner.build_schedule(root, "smoke", model="all")
    test = runner.build_schedule(root, "test", model="all")
    assert len(preflight) == 8
    assert len(smoke) == 200
    assert len(test) == 400
    assert len({item.identity for item in test}) == 400
    shards = [runner.shard_schedule(test, index=i, total=3) for i in range(3)]
    union = {item.identity for shard in shards for item in shard}
    assert union == {item.identity for item in test}
    assert sum(len(shard) for shard in shards) == 400
    assert runner.build_schedule(root, "test", model="vggt")
    assert len(runner.full_schedule(root, "test")) == 400


def test_smoke_manifest_is_non_scientific(tmp_path: Path):
    root = _minimal_root(tmp_path)
    p1 = runner.build_schedule(root, "preflight", model="vggt")[0]
    item = runner.build_schedule(root, "smoke", model="vggt")[0]
    p3 = runner.build_schedule(root, "test", model="vggt")[0]
    specs = {"VGGT": runner.frozen_model_spec("VGGT", None)}
    p1_manifest = runner.make_manifest(p1, root, specs, device="cuda:0")
    smoke_manifest = runner.make_manifest(item, root, specs, device="cuda:0")
    p3_manifest = runner.make_manifest(p3, root, specs, device="cuda:0")
    assert p1_manifest.mode is RunMode.REAL
    assert p1_manifest.scientific_validity is ScientificValidity.SCIENTIFIC
    assert smoke_manifest.mode is RunMode.SMOKE
    assert smoke_manifest.scientific_validity is ScientificValidity.NON_SCIENTIFIC_SMOKE
    assert p3_manifest.mode is RunMode.REAL
    assert p3_manifest.split == "test"


def test_atomic_run_skip_partial_conflict_invalid_and_claim(tmp_path: Path):
    root = _minimal_root(tmp_path)
    out = tmp_path / "out"
    item = runner.build_schedule(root, "smoke", model="vggt")[0]
    context = runner.RunnerContext(root=root, output_root=out, config_path=None, device="cuda:0")
    result = runner.execute_item(context, item, adapter_factory=lambda _model, _ctx: FakeAdapter(), audit_factory=runner.fixture_audit_factory)
    assert result.state == "completed"
    bundle = runner.load_completed_bundle(result.bundle_dir)
    validate_artifact_bundle(*bundle)
    (result.bundle_dir / "stale.partial").write_text("x", encoding="utf-8")
    skipped = runner.execute_item(context, item, adapter_factory=lambda _model, _ctx: FakeAdapter(), audit_factory=runner.fixture_audit_factory)
    assert skipped.state == "skipped"
    (result.bundle_dir / "prediction_artifact.json").write_text("{}", encoding="utf-8")
    with pytest.raises(runner.RunnerError, match="provenance-conflict"):
        runner.execute_item(context, item, adapter_factory=lambda _model, _ctx: FakeAdapter(), audit_factory=runner.fixture_audit_factory)
    invalid_item = runner.build_schedule(root, "smoke", model="mast3r")[0]
    invalid = runner.execute_item(context, invalid_item, adapter_factory=lambda _model, _ctx: FakeAdapter(invalid=True), audit_factory=runner.fixture_audit_factory)
    assert invalid.invalid_prediction is True
    exception_item = runner.build_schedule(root, "smoke", model="vggt")[1]
    exception = runner.execute_item(
        context,
        exception_item,
        adapter_factory=lambda _model, _ctx: (_ for _ in ()).throw(RuntimeError("boom")),
        audit_factory=runner.fixture_audit_factory,
    )
    assert exception.invalid_prediction is True
    ledger = runner.read_stage_ledger(out, "smoke")
    assert ledger["counts"]["completed"] == 3
    assert ledger["counts"]["invalid"] == 2
    lock = runner.claim_item(out / "stage" / "smoke" / "claims", item, stale_seconds=3600)
    assert lock.acquired
    assert not runner.claim_item(out / "stage" / "smoke" / "claims", item, stale_seconds=3600).acquired
    old = lock.path
    old.write_text(json.dumps({"created_at": "1970-01-01T00:00:00Z"}), encoding="utf-8")
    assert runner.claim_item(out / "stage" / "smoke" / "claims", item, stale_seconds=0).acquired


def test_test_stage_freeze_and_budget_refuse_mutation(tmp_path: Path):
    root = _minimal_root(tmp_path)
    out = tmp_path / "out"
    context = runner.RunnerContext(root=root, output_root=out, config_path=None, device="cuda:0")
    freeze = runner.ensure_stage_freeze(context, "test", dry_run=False)
    assert freeze["stage"] == "test"
    split = root / "manifests" / "split_view_manifest.json"
    split.write_text(split.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(runner.RunnerError, match="changed after freeze"):
        runner.ensure_stage_freeze(context, "test", dry_run=False)
    blocked = runner.check_budget(
        context,
        "test",
        next_stage_gpu_hours=51.0,
        next_stage_bytes=1,
        completed_gpu_hours=0.0,
        completed_bytes=0,
    )
    assert blocked["status"] == "BLOCKED_RESOURCE_BUDGET"
    assert runner.refuse_shrunk_test_grid(runner.build_schedule(root, "test", model="vggt")) == "TEST_GRID_SHRINK_FORBIDDEN"


def test_repeatability_and_p5_short_circuit(tmp_path: Path):
    root = _minimal_root(tmp_path)
    p5 = runner.build_zero_update_schedule(root, {"status": "PASS"}, model="all")
    assert len(p5) == 20 * 2 * 3 * 4
    assert {item.subset for item in p5} == {(0, 4), (1, 5), (2, 6), (3, 7)}
    assert all(item.parent_identity for item in p5)
    assert {item.severity for item in p5} == {2}
    assert {item.condition for item in p5} == {"fog", "low-light-noise", "defocus"}
    ok = runner.evaluate_repeatability(
        {"input_digest": "a", "aggregate_error": 100.0, "rho": 0.3},
        {"input_digest": "a", "aggregate_error": 100.05, "rho": 0.304},
    )
    assert ok["passed"] is True
    bad = runner.evaluate_repeatability(
        {"input_digest": "a", "aggregate_error": 100.0, "rho": 0.3},
        {"input_digest": "a", "aggregate_error": 101.0, "rho": 0.306},
    )
    assert bad["reason_code"] == "PREFLIGHT_REPEATABILITY_FAILED"
    assert runner.zero_update_schedule_allowed({"status": "FAIL"})["status"] == "SHORT_CIRCUIT_P5"
