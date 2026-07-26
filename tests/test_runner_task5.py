from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import georeliab_mve.audit as audit_module
import georeliab_mve.gates as gates_module
from georeliab_mve import runner
from georeliab_mve.audit import load_stage_evidence_manifest
from georeliab_mve.cli import main
from georeliab_mve.contracts import (
    AuditRecord,
    PredictionArtifact,
    RunManifest,
    RunMode,
    SampleKey,
    ScientificValidity,
    validate_artifact_bundle,
)
from georeliab_mve.preparation_round2 import PreparedBatch
from georeliab_mve.prepared_inputs import implementation_evidence


def _sha(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _overlay(path: Path) -> Path:
    path.write_text(
        "[runtime]\n"
        f"root = '{path.parent.as_posix()}'\n"
        "vggt_source = '/src/vggt'\nmast3r_source = '/src/mast3r'\n"
        "dust3r_source = '/src/dust3r'\ncroco_source = '/src/croco'\n"
        "vggt_env = '/env/vggt'\nmast3r_env = '/env/mast3r'\n"
        "vggt_python = '3.10.20'\nvggt_torch = '2.3.1+cu121'\n"
        "mast3r_python = '3.10.20'\nmast3r_torch = '2.5.1+cu121'\n"
        "[resources]\n"
        "vggt_checkpoint = '/models/vggt.pt'\nvggt_checkpoint_sha256 = '" + "2" * 64 + "'\n"
        "vggt_source_commit = 'a288dd0f14786c93483e45524328726ab7b1b4ce'\n"
        "mast3r_checkpoint = '/models/mast3r.safetensors'\nmast3r_checkpoint_sha256 = '" + "3" * 64 + "'\n"
        "mast3r_config = '/models/mast3r.json'\nmast3r_config_sha256 = '" + "4" * 64 + "'\n"
        "mast3r_source_commit = 'f5209afc300cec36239a7ac992263f36847bbba0'\n"
        "dust3r_source_commit = '3cc8c88c413bb9e34c41db0e0eef99c2ee010b12'\n"
        "croco_source_commit = 'd7de0705845239092414480bd829228723bf20de'\n",
        encoding="utf-8",
    )
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
    materialization_path = _write_json(
        root / "manifests" / "frozen_materialization.json",
        {
            "schema_version": "test-frozen-materialization-placeholder",
            "split_view_manifest_sha256": runner.sha256_file(split_path),
        },
    )
    materialization_sha = runner.sha256_file(materialization_path)
    corruption_path = _write_json(root / "manifests" / "corruption_calibration.json", {"schema_version": "corruption-calibration-v1"})
    parameter_sha = runner.sha256_file(corruption_path)
    prepared_shas: dict[str, str] = {}
    render_index_records = []
    for stage, scenes in (("smoke", splits["dev"]), ("test", splits["test"])):
        records = []
        for scene in scenes:
            for view in views[str(scene)]:
                key = f"dtu/{'dev' if stage == 'smoke' else 'test'}/scan{scene}/views-0001/clean/0/0"
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
        prepared = _write_json(
            root / "prepared" / f"render_inputs_{stage}.json",
            {
                "schema_version": "prepared-input-v2",
                "stage": stage,
                "split": "dev" if stage == "smoke" else "test",
                "producer": implementation_evidence(),
                "record_count": len(records),
                "split_view_manifest_sha256": runner.sha256_file(split_path),
                "materialization_sha256": materialization_sha,
                "records": records,
            },
        )
        prepared_shas[stage] = runner.sha256_file(prepared)
    for scene in splits["test"]:
        for condition, severity in runner.CONDITIONS:
            views_payload = []
            for view in views[str(scene)]:
                png = root / "rendered" / "test" / f"scan{scene:03d}_view{view:03d}_{condition}_s{severity}.png"
                views_payload.append({"view_id": view, "path": str(png), "sha256": runner.sha256_file(png)})
            render_index_records.append({"scene": f"scan{scene}", "condition": condition, "severity": str(severity), "sample_key": f"dtu/test/scan{scene}/views-0001/{condition}/{severity}/0", "views": views_payload})
    tartan_prepared = _write_json(root / "prepared" / "tartanair_p000_pairs.json", {"schema_version": "tartanair-prepared-v2", "producer": implementation_evidence(), "record_count": 100, "records": [{"frame_id": f"{index:06d}"} for index in range(100)]})
    qa = _write_json(root / "manifests" / "corruption_calibration_qa.json", {"schema_version": "calibration-qa-v1", "passed": True, "checks": {"fog": True, "synthetic_fog": True, "low_light": True, "defocus": True, "gt": True, "cross_view": True}, "parameter_manifest_sha256": parameter_sha, "split_view_manifest_sha256": runner.sha256_file(split_path), "materialization_sha256": materialization_sha})
    _write_json(root / "manifests" / "test_render_lock.json", {"schema_version": "test-render-lock-v1", "stage": "test", "split": "test", "parameter_manifest_sha256": parameter_sha, "split_view_manifest_sha256": runner.sha256_file(split_path), "materialization_sha256": materialization_sha, "calibration_qa_sha256": runner.sha256_file(qa), "prepared_input_sha256": prepared_shas["test"]})
    _write_json(root / "evidence" / "tartanair_native_fog_sanity.json", {"reason_code": "TARTANAIR_NATIVE_FOG_SANITY", "passed": True, "evaluated_frames": 100, "negative_frames": 80, "correlations": [-0.1] * 80 + [0.0] * 20, "prepared_input_sha256": runner.sha256_file(tartan_prepared), "calibration_qa_sha256": runner.sha256_file(qa)})
    _write_json(root / "manifests" / "test_render_index.json", {"schema_version": "test-render-index-v1", "stage": "test", "split": "test", "prepared_input_sha256": prepared_shas["test"], "parameter_manifest_sha256": parameter_sha, "records": render_index_records})
    return root


def _fake_prediction(manifest: RunManifest, sample_key: SampleKey, output_dir: Path, *, invalid: bool = False) -> PredictionArtifact:
    output_dir.mkdir(parents=True, exist_ok=True)
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64)
    camera = np.repeat(np.eye(4)[None, :, :], 8, axis=0)
    camera[:, :3, 3] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.float64)
    intrinsics = np.repeat(np.eye(3)[None, :, :], 8, axis=0)
    pixels = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    view_id = np.array([1, 3, 5, 7], dtype=np.int64)
    conf = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    mask = np.array([True, True, True, True])
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


class SixViewAdapter:
    def __init__(self, *, invalid: bool = False):
        self.seen: list[tuple[int, ...]] = []
        self.invalid = invalid

    def predict_sample(self, manifest, sample_key, rendered_views):
        self.seen.append(tuple(view.view_id for view in rendered_views))
        assert len(rendered_views) == 6
        return _fake_prediction(
            manifest,
            sample_key,
            Path(rendered_views[0].png_path).parent / f"adapter6_{len(self.seen)}",
            invalid=self.invalid,
        )


class ContextOutputAdapter:
    def __init__(self, output_root: Path):
        self.output_root = output_root

    def predict_sample(self, manifest, sample_key, rendered_views):
        prediction = _fake_prediction(manifest, sample_key, self.output_root)
        _write_json(self.output_root / "adapter_prediction.json", {"geometry_prediction_uri": prediction.geometry_prediction_uri})
        return prediction


def _patch_task3_loader(monkeypatch):
    def load_prepared(path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        return PreparedBatch(
            stage=payload["stage"],
            split=payload["split"],
            split_view_manifest_sha256=payload["split_view_manifest_sha256"],
            materialization_sha256=payload["materialization_sha256"],
            records=tuple(() for _ in payload["records"]),
        )

    def load_tartan(path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [(str(index), np.empty((0,)), np.empty((0,))) for index in range(len(payload["records"]))]

    monkeypatch.setattr(audit_module, "_load_verified_prepared_batch", load_prepared)
    monkeypatch.setattr(audit_module, "_load_verified_tartanair_pairs", load_tartan)


def _materialize_zero_update_bundles(context: runner.RunnerContext, root: Path, freeze: dict):
    gate = {"schema_version": "native-phenomenon-gate-v1", "status": "PASS"}
    for item in runner.build_zero_update_schedule(root, gate, model="all"):
        bundle = runner._zero_update_bundle_dir(context.output_root, item)
        bundle.mkdir(parents=True, exist_ok=True)
        _parent_dir, parent_manifest, parent_prediction, _parent_audit = runner._parent_bundle_for_zero_update(context, item)
        kept = [index for index in range(8) if index not in tuple(item.subset)]
        subset_path = bundle / "subset_prediction.npz"
        np.savez(
            subset_path,
            points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, float(len(kept))], [0.0, 1.0, 0.5], [1.0, 1.0, 1.0]], dtype=np.float64),
            camera_centers=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.float64)[kept],
            view_ids=np.asarray(kept, dtype=np.int64),
            parent_model=np.asarray(parent_manifest.model),
            parent_sample_key=np.asarray(parent_prediction.sample_key),
            parent_project_commit=np.asarray(parent_manifest.provenance.project_commit),
            parent_run_id=np.asarray(parent_manifest.run_id),
        )
        result_path = _write_json(bundle / "zero_update_result.json", {"schema_version": "zero-update-result-v1", "identity": item.identity, "invalid_prediction": False, "reason_code": "OK", "subset_prediction_sha256": runner.sha256_file(subset_path)})
        stage_payload = runner._stage_item_payload_for_context(item, freeze)
        stage_payload.update(
            {
                "subset_prediction_sha256": runner.sha256_file(subset_path),
                "zero_update_result_sha256": runner.sha256_file(result_path),
            }
        )
        _write_json(bundle / "stage_item.json", stage_payload)


def _audit_factory(manifest: RunManifest, prediction: PredictionArtifact, output_dir: Path) -> AuditRecord:
    output_dir.mkdir(parents=True, exist_ok=True)
    dense = output_dir / "dense_audit.npz"
    gt = output_dir / "gt_points.npz"
    np.savez(gt, gt_points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64))
    if prediction.invalid_prediction:
        np.savez(
            dense,
            voxel_points=np.empty((0, 3)), raw_confidence=np.empty((0,)), risk=np.empty((0,)),
            gt_error=np.empty((0,)), failure_label=np.empty((0,), dtype=bool), provenance_count=np.empty((0,), dtype=np.int64),
        )
        return AuditRecord(
            manifest.run_id, prediction.sample_key, None, True, 1.0, 0.0, False, 0.0, True,
            {"dense_audit_uri": dense.as_uri(), "dense_audit_sha256": runner.sha256_file(dense), "gt_points_uri": gt.as_uri(), "gt_points_sha256": runner.sha256_file(gt), "audit_factory": "test-only"},
        )
    key = SampleKey.parse(prediction.sample_key)
    if key.condition == "clean":
        gt_error = np.array([0.5, 1.0, 3.0, 4.0], dtype=np.float64)
    elif key.severity == "1":
        gt_error = np.array([0.5, 3.0, 1.0, 4.0], dtype=np.float64)
    elif key.severity == "2":
        gt_error = np.array([3.0, 0.5, 4.0, 1.0], dtype=np.float64)
    else:
        gt_error = np.array([4.0, 3.0, 1.0, 0.5], dtype=np.float64)
    np.savez(
        dense,
        voxel_points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
        raw_confidence=np.array([2.0, 3.0, 4.0, 5.0]),
        risk=np.array([0.1, 0.2, 0.3, 0.4]),
        gt_error=gt_error,
        failure_label=gt_error > 2.0,
        provenance_count=np.ones(4, dtype=np.int64),
    )
    return AuditRecord(
        manifest.run_id, prediction.sample_key, float(np.median(gt_error)), bool(np.any(gt_error > 2.0)), 0.1, 1.0, True, 1.0, False,
        {"dense_audit_uri": dense.as_uri(), "dense_audit_sha256": runner.sha256_file(dense), "gt_points_uri": gt.as_uri(), "gt_points_sha256": runner.sha256_file(gt), "audit_factory": "test-only"},
    )


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
    result = runner.execute_item(context, item, adapter_factory=lambda _model, _ctx: FakeAdapter(), audit_factory=_audit_factory)
    assert result.state == "completed"
    bundle = runner.load_completed_bundle(result.bundle_dir)
    validate_artifact_bundle(*bundle)
    (result.bundle_dir / "stale.partial").write_text("x", encoding="utf-8")
    skipped = runner.execute_item(context, item, adapter_factory=lambda _model, _ctx: FakeAdapter(), audit_factory=_audit_factory)
    assert skipped.state == "skipped"
    (result.bundle_dir / "prediction_artifact.json").write_text("{}", encoding="utf-8")
    with pytest.raises(runner.RunnerError, match="provenance-conflict"):
        runner.execute_item(context, item, adapter_factory=lambda _model, _ctx: FakeAdapter(), audit_factory=_audit_factory)
    invalid_item = runner.build_schedule(root, "smoke", model="mast3r")[0]
    invalid = runner.execute_item(context, invalid_item, adapter_factory=lambda _model, _ctx: FakeAdapter(invalid=True), audit_factory=_audit_factory)
    assert invalid.invalid_prediction is True
    assert invalid.reason_code == "INVALID_PREDICTION"
    resumed_invalid = runner.execute_item(context, invalid_item, adapter_factory=lambda _model, _ctx: FakeAdapter(), audit_factory=_audit_factory)
    assert resumed_invalid.state == "skipped"
    assert resumed_invalid.invalid_prediction is True
    assert runner.stage_progress_counts(out, "smoke", runner.build_schedule(root, "smoke", model="all"))["invalid"] == 1
    exception_item = runner.build_schedule(root, "smoke", model="vggt")[1]
    exception = runner.execute_item(
        context,
        exception_item,
        adapter_factory=lambda _model, _ctx: (_ for _ in ()).throw(RuntimeError("boom")),
        audit_factory=_audit_factory,
    )
    assert exception.invalid_prediction is True
    assert exception.reason_code == "ADAPTER_EXCEPTION"
    ledger = runner.read_stage_ledger(out, "smoke")
    assert ledger["counts"]["completed"] == 3
    assert len(ledger["rows"]) >= 4
    assert ledger["counts"]["invalid"] == 2
    lock = runner.claim_item(out / "stage" / "smoke" / "claims", item, stale_seconds=3600)
    assert lock.acquired
    assert not runner.claim_item(out / "stage" / "smoke" / "claims", item, stale_seconds=3600).acquired
    old = lock.path
    old.write_text(json.dumps({"created_at": "1970-01-01T00:00:00Z"}), encoding="utf-8")
    assert runner.claim_item(out / "stage" / "smoke" / "claims", item, stale_seconds=0).acquired


def test_stage_orchestrator_blocks_budget_before_dispatch_and_cli_dry_run(tmp_path: Path, monkeypatch, capsys):
    root = _minimal_root(tmp_path)
    called = []

    monkeypatch.setattr(runner, "default_adapter_factory", lambda *_args: called.append(True))
    monkeypatch.setattr(runner, "_verify_stage_readiness", lambda *_args: None)
    context = runner.RunnerContext(root=root, output_root=tmp_path / "out", config_path=None, device="cuda:0")
    blocked = runner.run_stage(
        context,
        stage="test",
        model="all",
        shard="0/1",
        dry_run=False,
        adapter_factory=lambda *_args: called.append(True),
        audit_factory=_audit_factory,
        budget_override={"next_stage_gpu_hours": 51.0, "next_stage_bytes": 1},
    )
    assert blocked["status"] == "BLOCKED_RESOURCE_BUDGET"
    assert called == []
    assert not list((tmp_path / "out").rglob("*.partial"))
    code = runner.cli_main([
        "run-georeliab", "--stage", "smoke", "--model", "all", "--device", "cuda:0",
        "--shard", "0/1", "--config", str(root / "missing.toml"), "--output-root", str(root), "--dry-run",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["schedule_counts"]["scheduled"] == 200
    assert called == []
    assert runner.cli_main(["preflight-real", "--config", str(root / "missing.toml"), "--output-root", str(root), "--device", "cuda:0", "--dry-run"]) == 0
    assert main(["run-georeliab", "--stage", "smoke", "--model", "all", "--device", "cuda:0", "--shard", "0/1", "--config", str(root / "missing.toml"), "--output-root", str(root), "--dry-run"]) == 0


def test_fingerprint_skip_refusal_no_fixture_default_and_ledger_concurrency(tmp_path: Path):
    root = _minimal_root(tmp_path)
    out = tmp_path / "out"
    context = runner.RunnerContext(root=root, output_root=out, config_path=None, device="cuda:0")
    item = runner.build_schedule(root, "smoke", model="vggt")[0]
    with pytest.raises(runner.RunnerError, match="production audit factory"):
        runner.execute_item(context, item, adapter_factory=lambda _m, _c: FakeAdapter())
    result = runner.execute_item(context, item, adapter_factory=lambda _m, _c: FakeAdapter(), audit_factory=_audit_factory)
    stage_item = result.bundle_dir / "stage_item.json"
    changed_config = root / "changed_overlay.toml"
    changed_config.write_text(
        "[resources]\nvggt_checkpoint_sha256 = '" + "1" * 64 + "'\n"
        "[runtime]\nvggt_python='3.10.20'\nvggt_torch='2.3.1+cu121'\n",
        encoding="utf-8",
    )
    changed_context = runner.RunnerContext(root=root, output_root=out, config_path=changed_config, device="cuda:0")
    with pytest.raises(runner.RunnerError, match="provenance-conflict"):
        runner.execute_item(changed_context, item, adapter_factory=lambda _m, _c: FakeAdapter(), audit_factory=_audit_factory)
    rows = [threading.Thread(target=runner.append_ledger_row, args=(out, "smoke", {"state": "completed", "identity": str(i)})) for i in range(8)]
    for row in rows:
        row.start()
    for row in rows:
        row.join()
    ledger = runner.read_stage_ledger(out, "smoke")
    assert len(ledger["rows"]) >= 9

    stale_lock = out / "stage" / "smoke" / "ledger.jsonl.lock"
    stale_lock.write_text("interrupted", encoding="utf-8")
    stale_time = time.time() - runner.LEDGER_LOCK_STALE_SECONDS - 1
    os.utime(stale_lock, (stale_time, stale_time))
    runner.append_ledger_row(out, "smoke", {"state": "completed", "identity": "recovered"})
    assert not stale_lock.exists()
    assert runner.read_stage_ledger(out, "smoke")["rows"][-1]["identity"] == "recovered"


def test_test_stage_freeze_and_budget_refuse_mutation(tmp_path: Path):
    root = _minimal_root(tmp_path)
    out = tmp_path / "out"
    context = runner.RunnerContext(root=root, output_root=out, config_path=None, device="cuda:0")
    freeze = runner.ensure_stage_freeze(context, "test", dry_run=False)
    assert freeze["stage"] == "test"
    assert freeze["schedule_count"] == 400
    for key in (
        "project_commit", "project_tree", "model_specs", "prepared_input_sha256",
        "frozen_materialization_sha256", "prepared_materialization_sha256",
        "schedule_fingerprint", "render_digest_fingerprint", "test_render_lock_sha256",
        "test_render_index_sha256", "tartanair_prepared_input_sha256",
    ):
        assert key in freeze
    split = root / "manifests" / "split_view_manifest.json"
    split.write_text(split.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(runner.RunnerError, match="changed after freeze"):
        runner.ensure_stage_freeze(context, "test", dry_run=False)
    root2 = _minimal_root(tmp_path / "render_mutation")
    context2 = runner.RunnerContext(root=root2, output_root=tmp_path / "out2", config_path=None, device="cuda:0")
    runner.ensure_stage_freeze(context2, "test", dry_run=False)
    target_png = next((root2 / "rendered" / "test").glob("*_clean_s0.png"))
    target_png.write_bytes(target_png.read_bytes() + b"mutated")
    with pytest.raises(runner.RunnerError, match="changed after freeze"):
        runner.ensure_stage_freeze(context2, "test", dry_run=False)
    root3 = _minimal_root(tmp_path / "source_mutation")
    context3 = runner.RunnerContext(root=root3, output_root=tmp_path / "out3", config_path=None, device="cuda:0")
    runner.ensure_stage_freeze(context3, "test", dry_run=False)
    target_meta = next((root3 / "rendered" / "test").glob("*_clean_s0.json"))
    metadata = json.loads(target_meta.read_text(encoding="utf-8"))
    metadata["raw_source_sha256"] = "f" * 64
    target_meta.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(runner.RunnerError, match="changed after freeze"):
        runner.ensure_stage_freeze(context3, "test", dry_run=False)
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


def test_model_spec_records_mast3r_config_sha_and_preflight_device_default(tmp_path: Path, capsys):
    root = _minimal_root(tmp_path)
    config = _overlay(root / "overlay.toml")
    spec = runner.frozen_model_spec("MASt3R", config)
    assert spec.config_sha256 == "4" * 64
    item = runner.build_schedule(root, "test", model="mast3r")[0]
    manifest = runner.make_manifest(item, root, {"MASt3R": spec}, device="cuda:7")
    assert manifest.environment["config_sha256"] == "4" * 64
    assert runner.cli_main(["preflight-real", "--config", str(config), "--output-root", str(root), "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True


def test_default_adapter_factory_reuses_upstream_but_isolates_outputs_and_cache(tmp_path: Path, monkeypatch):
    from georeliab_mve import adapters as adapters_module

    root = _minimal_root(tmp_path)
    config = _overlay(root / "overlay.toml")
    runner._UPSTREAM_CACHE.clear()
    counts = {"vggt": 0, "mast3r": 0}

    class FakeVGGTUpstream:
        def __init__(self, runtime, *, device):
            counts["vggt"] += 1
            self.runtime = runtime
            self.device = device

    class FakeMASt3RUpstream:
        def __init__(self, runtime, *, device, cache_dir):
            counts["mast3r"] += 1
            self.runtime = runtime
            self.device = device
            self.cache_dir = cache_dir

    monkeypatch.setattr(adapters_module, "RealVGGTUpstream", FakeVGGTUpstream)
    monkeypatch.setattr(adapters_module, "RealMASt3RUpstream", FakeMASt3RUpstream)
    v1 = runner.default_adapter_factory("VGGT", runner.RunnerContext(root=root, output_root=tmp_path / "a", config_path=config, device="cuda:0"))
    v2 = runner.default_adapter_factory("VGGT", runner.RunnerContext(root=root, output_root=tmp_path / "b", config_path=config, device="cuda:0"))
    assert counts["vggt"] == 1
    assert v1.upstream is v2.upstream
    assert v1.output_root != v2.output_root
    m1 = runner.default_adapter_factory("MASt3R", runner.RunnerContext(root=root, output_root=tmp_path / "m1", config_path=config, device="cuda:0"))
    m2 = runner.default_adapter_factory("MASt3R", runner.RunnerContext(root=root, output_root=tmp_path / "m2", config_path=config, device="cuda:0"))
    assert counts["mast3r"] == 1
    assert m1.upstream is m2.upstream
    assert m1.cache_dir == tmp_path / "m1" / "mast3r_cache"
    assert m2.cache_dir == tmp_path / "m2" / "mast3r_cache"
    assert m2.upstream.cache_dir == m2.cache_dir
    runner._UPSTREAM_CACHE.clear()


def test_adapter_outputs_are_committed_inside_atomic_bundle(tmp_path: Path):
    root = _minimal_root(tmp_path)
    out = tmp_path / "out"
    context = runner.RunnerContext(root=root, output_root=out, config_path=None, device="cuda:0")
    item = runner.build_schedule(root, "smoke", model="vggt")[0]
    result = runner.execute_item(context, item, adapter_factory=lambda _model, ctx: ContextOutputAdapter(ctx.output_root), audit_factory=_audit_factory)
    assert result.state == "completed"
    assert (result.bundle_dir / "adapter" / "geometry.npz").exists()
    assert not (out / "adapter").exists()
    assert not list(out.rglob("*.partial"))
    for payload in result.bundle_dir.rglob("*.json"):
        assert ".partial" not in payload.read_text(encoding="utf-8")
    skipped = runner.execute_item(context, item, adapter_factory=lambda _model, ctx: ContextOutputAdapter(ctx.output_root), audit_factory=_audit_factory)
    ledger = runner.read_stage_ledger(out, "smoke")
    latest = ledger["rows"][-1]
    assert skipped.state == "skipped"
    assert latest["attempt"] == 2
    assert set(latest["artifact_digests"]) >= {"run_manifest.json", "prediction_artifact.json", "audit_record.json", "stage_item.json", "scene_summary.json"}


def test_repeatability_and_p5_short_circuit(tmp_path: Path):
    root = _minimal_root(tmp_path)
    p5_gate = {"schema_version": "native-phenomenon-gate-v1", "status": "PASS"}
    p5 = runner.build_zero_update_schedule(root, p5_gate, model="all")
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
    assert runner.zero_update_schedule_allowed({"status": "PASS"})["status"] == "BLOCKED_PENDING_EVIDENCE"
    assert runner.zero_update_schedule_allowed({"schema_version": "native-phenomenon-gate-v1", "status": "FAIL"})["status"] == "SHORT_CIRCUIT_P5"
    context = runner.RunnerContext(root=root, output_root=tmp_path / "out", config_path=None, device="cuda:0")
    blocked = runner.run_stage(
        context,
        stage="zero-update",
        model="vggt",
        shard="0/1",
        dry_run=False,
        native_gate={"schema_version": "native-phenomenon-gate-v1", "status": "PASS"},
        budget_override={"next_stage_gpu_hours": 0.0, "next_stage_bytes": 0},
    )
    assert blocked["status"] == "BLOCKED_PENDING_EVIDENCE"
    assert blocked["reason_code"] in {"NATIVE_CONFIDENCE_GATE_MISSING", "NATIVE_CONFIDENCE_GATE_INJECTION_FORBIDDEN"}


def test_native_gate_writer_binds_digest_and_is_immutable(tmp_path: Path, monkeypatch):
    root = _minimal_root(tmp_path)
    out = tmp_path / "out"
    context = runner.RunnerContext(root=root, output_root=out, config_path=None, device="cuda:0")
    freeze = runner.ensure_stage_freeze(context, "test", dry_run=False)
    evidence_path = out / "stage" / "test" / "stage_evidence_p3.json"
    _write_json(evidence_path, {"schema_version": "stage-evidence-v1", "bundle_index": []})

    class FakeEvidence:
        schedule_counts = {"scheduled": 400, "completed": 400, "missing": 0, "invalid": 0}

        def to_gate_input(self):
            return object()

    monkeypatch.setattr(audit_module, "load_stage_evidence_manifest", lambda _path: FakeEvidence())
    monkeypatch.setattr(gates_module, "evaluate_georeliab_gate", lambda _input: gates_module.GateDecision("georeliab", gates_module.GateStatus.FAIL, ("P5_DOWNSTREAM_SCHEDULE_COUNTS_INVALID",), {}, ScientificValidity.SCIENTIFIC))
    audit_output = {"stage_evidence_path": str(evidence_path), "stage_evidence_sha256": runner.sha256_file(evidence_path), "georeliab_gate": {"reason_codes": ["P5_DOWNSTREAM_SCHEDULE_COUNTS_INVALID"]}, "p5_skip_reason": None}
    gate_path = runner.write_native_phenomenon_gate_from_audit_output(context, audit_output)
    assert json.loads(gate_path.read_text(encoding="utf-8"))["status"] == "PASS"
    assert runner.load_native_phenomenon_gate(context)["status"] == "PASS"
    assert runner.write_native_phenomenon_gate_from_audit_output(context, audit_output) == gate_path
    tampered = json.loads(gate_path.read_text(encoding="utf-8"))
    tampered["status"] = "FAIL"
    gate_path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    assert runner.load_native_phenomenon_gate(context)["reason_code"] == "NATIVE_CONFIDENCE_GATE_DECISION_MISMATCH"
    with pytest.raises(runner.RunnerError, match="immutable native"):
        runner.write_native_phenomenon_gate_from_audit_output(context, audit_output)
    gate_path.unlink()
    bad = dict(audit_output)
    bad["stage_evidence_sha256"] = "0" * 64
    with pytest.raises(runner.RunnerError, match="not bound"):
        runner.write_native_phenomenon_gate_from_audit_output(context, bad)
    monkeypatch.setattr(gates_module, "evaluate_georeliab_gate", lambda _input: gates_module.GateDecision("georeliab", gates_module.GateStatus.FAIL, ("CONFIDENCE_FAILURE_GATE_NOT_MET",), {"p5_skip_reason": "P4_CONFIDENCE_PHENOMENON_GATE_FAILED"}, ScientificValidity.SCIENTIFIC))
    fail_output = {"stage_evidence_path": str(evidence_path), "stage_evidence_sha256": runner.sha256_file(evidence_path), "georeliab_gate": {"reason_codes": ["CONFIDENCE_FAILURE_GATE_NOT_MET"]}, "p5_skip_reason": "P4_CONFIDENCE_PHENOMENON_GATE_FAILED"}
    fail_path = runner.write_native_phenomenon_gate_from_audit_output(context, fail_output)
    assert json.loads(fail_path.read_text(encoding="utf-8"))["status"] == "FAIL"


def test_zero_update_executes_six_view_subsets_with_parent_linkage(tmp_path: Path):
    root = _minimal_root(tmp_path)
    out = tmp_path / "out"
    context = runner.RunnerContext(root=root, output_root=out, config_path=None, device="cuda:0")
    parent = next(item for item in runner.build_schedule(root, "test", model="vggt") if item.condition == "fog" and item.severity == 2)
    runner.execute_item(context, parent, adapter_factory=lambda _m, _c: FakeAdapter(), audit_factory=_audit_factory, stage_fingerprint=runner.ensure_stage_freeze(context, "test", dry_run=False))
    gate = {"schema_version": "native-phenomenon-gate-v1", "status": "PASS"}
    subsets = [item for item in runner.build_zero_update_schedule(root, gate, model="vggt") if item.parent_identity == parent.identity]
    adapter = SixViewAdapter()
    for item in subsets:
        runner.execute_zero_update_item(context, item, adapter_factory=lambda _m, _c: adapter, stage_fingerprint=runner.ensure_stage_freeze(context, "test", dry_run=True))
    assert len(subsets) == 4
    assert len(set(adapter.seen)) == 4
    assert all(len(view_ids) == 6 for view_ids in adapter.seen)
    artifact = next((out / "stage" / "zero-update" / "bundles" / "vggt").glob("*/subset_prediction.npz"))
    payload = np.load(artifact, allow_pickle=False)
    assert str(payload["parent_model"]) == "VGGT"
    assert str(payload["parent_sample_key"]) == str(parent.sample_key)
    assert str(payload["parent_run_id"])
    assert str(payload["parent_project_commit"])
    assert set(tuple(seen) for seen in adapter.seen) == {tuple(view.view_id for index, view in enumerate(parent.rendered_views) if index not in item.subset) for item in subsets}
    stage_item = json.loads((artifact.parent / "stage_item.json").read_text(encoding="utf-8"))
    assert tuple(payload["view_ids"].tolist()) == tuple(index for index in range(8) if index not in tuple(stage_item["subset"]))
    zero_result = json.loads((artifact.parent / "zero_update_result.json").read_text(encoding="utf-8"))
    assert zero_result["subset_prediction_sha256"] == runner.sha256_file(artifact)
    skipped = runner.execute_zero_update_item(context, subsets[0], adapter_factory=lambda _m, _c: adapter, stage_fingerprint=runner.ensure_stage_freeze(context, "test", dry_run=True))
    assert skipped.state == "skipped"

    invalid_parent = next(item for item in runner.build_schedule(root, "test", model="mast3r") if item.condition == "fog" and item.severity == 2)
    runner.execute_item(context, invalid_parent, adapter_factory=lambda _m, _c: FakeAdapter(), audit_factory=_audit_factory, stage_fingerprint=runner.ensure_stage_freeze(context, "test", dry_run=True))
    invalid_subset = next(item for item in runner.build_zero_update_schedule(root, gate, model="mast3r") if item.parent_identity == invalid_parent.identity)
    invalid_result = runner.execute_zero_update_item(context, invalid_subset, adapter_factory=lambda _m, _c: SixViewAdapter(invalid=True), stage_fingerprint=runner.ensure_stage_freeze(context, "test", dry_run=True))
    assert invalid_result.invalid_prediction is True
    assert invalid_result.reason_code == "INVALID_PREDICTION"
    invalid_skip = runner.execute_zero_update_item(context, invalid_subset, adapter_factory=lambda _m, _c: SixViewAdapter(), stage_fingerprint=runner.ensure_stage_freeze(context, "test", dry_run=True))
    assert invalid_skip.state == "skipped"
    assert invalid_skip.invalid_prediction is True
    assert invalid_skip.reason_code == "INVALID_PREDICTION"


def test_full_stage_evidence_chain_reaches_task3_loader_and_freezes_p3(tmp_path: Path, monkeypatch):
    _patch_task3_loader(monkeypatch)
    monkeypatch.setattr(runner, "_project_identity", lambda: ("a" * 40, "b" * 40))
    root = _minimal_root(tmp_path)
    out = tmp_path / "out"
    context = runner.RunnerContext(root=root, output_root=out, config_path=None, device="cuda:0")
    freeze = runner.ensure_stage_freeze(context, "test", dry_run=False)
    for item in runner.full_schedule(root, "test"):
        runner.execute_item(context, item, adapter_factory=lambda _model, ctx: ContextOutputAdapter(ctx.output_root), audit_factory=_audit_factory, stage_fingerprint=freeze)
    p3 = runner.build_stage_evidence_manifest(context, p3_only=True)
    assert p3 is not None
    p3_payload = json.loads(p3.read_text(encoding="utf-8"))
    assert len(p3_payload["bundle_index"]) == 400
    p3_evidence = load_stage_evidence_manifest(p3)
    assert dict(p3_evidence.schedule_counts) == {"scheduled": 400, "completed": 400, "missing": 0, "invalid": 0}
    assert len(p3_evidence.conditions) == 6
    downstream = runner.build_downstream_evidence(context)
    assert len(downstream) == 6
    _materialize_zero_update_bundles(context, root, freeze)
    zero = runner.build_zero_update_evidence(context)
    assert zero is not None and len(zero) == 6
    for path in zero:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert len(payload["subset_artifacts"]) == 20
        assert all(len(rows) == 4 for rows in payload["subset_artifacts"].values())
    final = runner.build_stage_evidence_manifest(context, p3_only=False)
    assert final is not None
    final_payload = json.loads(final.read_text(encoding="utf-8"))
    assert len(final_payload["downstream_index"]) == 6
    assert len(final_payload["zero_update_index"]) == 6
    final_evidence = load_stage_evidence_manifest(final)
    assert dict(final_evidence.statistics["downstream_schedule_counts"]) == {"scheduled": 6, "completed": 6, "missing": 0}
    assert len(final_evidence.downstream_harm) == 6
    assert len(final_evidence.zero_update) == 6
    assert runner.build_stage_evidence_manifest(context, p3_only=True) == p3
    first_summary = Path(p3_payload["bundle_index"][0]["scene_summary_path"])
    tampered = json.loads(first_summary.read_text(encoding="utf-8"))
    tampered["tamper"] = True
    first_summary.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    with pytest.raises(runner.RunnerError, match="immutable P3"):
        runner.build_stage_evidence_manifest(context, p3_only=True)


def test_preflight_real_runs_two_independent_repeats_and_blocks_threshold(tmp_path: Path, monkeypatch):
    root = _minimal_root(tmp_path)
    context = runner.RunnerContext(root=root, output_root=tmp_path / "out", config_path=None, device="cuda:0")
    monkeypatch.setattr(runner, "_verify_stage_readiness", lambda *_args: None)
    payload = runner.run_preflight_real(
        context,
        dry_run=False,
        adapter_factory=lambda _m, _c: FakeAdapter(),
        audit_factory=_audit_factory,
        budget_override={"next_stage_gpu_hours": 0.0, "next_stage_bytes": 0},
    )
    assert payload["status"] == "OK"
    assert len(payload["repeats"]) == 2
    assert all(repeat["schedule_counts"]["completed"] == 8 for repeat in payload["repeats"])
    assert all(repeat["schedule_counts"]["skipped"] == 0 for repeat in payload["repeats"])
    assert (tmp_path / "out" / "preflight-real" / "repeat-a" / "stage" / "preflight" / "bundles").exists()
    assert (tmp_path / "out" / "preflight-real" / "repeat-b" / "stage" / "preflight" / "bundles").exists()

    snapshots = iter([
        {"input_digest": "same", "aggregate_error": 1.0, "rho": 0.5},
        {"input_digest": "same", "aggregate_error": 1.01, "rho": 0.5},
    ])
    monkeypatch.setattr(runner, "_preflight_repeat_snapshot", lambda _root: next(snapshots))
    blocked = runner.run_preflight_real(
        runner.RunnerContext(root=root, output_root=tmp_path / "out_block", config_path=None, device="cuda:0"),
        dry_run=False,
        adapter_factory=lambda _m, _c: FakeAdapter(),
        audit_factory=_audit_factory,
        budget_override={"next_stage_gpu_hours": 0.0, "next_stage_bytes": 0},
    )
    assert blocked["status"] == "PREFLIGHT_REPEATABILITY_FAILED"


def test_budget_observes_nested_preflight_repeats(tmp_path: Path):
    root = _minimal_root(tmp_path)
    out = tmp_path / "out"
    ledger = out / "preflight-real" / "repeat-a" / "stage" / "preflight" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps({"state": "completed", "item_identity": "a", "runtime_seconds": 360.0, "artifact_bytes": 123}) + "\n", encoding="utf-8")
    context = runner.RunnerContext(root=root, output_root=out, config_path=None, device="cuda:0")
    budget = runner.estimate_stage_budget(context, "smoke", runner.build_schedule(root, "smoke", model="vggt"))
    assert budget["status"] == "OK"
    assert budget["remaining_gpu_hours"] < runner.GPU_HOUR_LIMIT


def test_cli_model_all_uses_frozen_env_workers_and_dry_run_does_not(tmp_path: Path, monkeypatch, capsys):
    root = _minimal_root(tmp_path)
    config = _overlay(root / "overlay.toml")
    calls = []
    call_kwargs = []

    def fake_run(command, **kwargs):
        if "--summary-json" not in command:
            return subprocess.CompletedProcess(command, 0, stdout="a" * 40 + "\n", stderr="")
        calls.append(command)
        call_kwargs.append(kwargs)
        summary_path = Path(command[command.index("--summary-json") + 1])
        _write_json(summary_path, {"status": "OK", "schedule_counts": {"scheduled": 200}})
        return subprocess.CompletedProcess(command, 0, stdout="upstream log noise\n{not-json}\n", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    code = runner.cli_main([
        "run-georeliab", "--stage", "smoke", "--model", "all", "--device", "cuda:0",
        "--shard", "0/1", "--config", str(config), "--output-root", str(root),
    ])
    assert code == 0
    assert len(calls) == 2
    assert str(calls[0][0]).replace("\\", "/").endswith("/env/vggt/bin/python")
    assert str(calls[1][0]).replace("\\", "/").endswith("/env/mast3r/bin/python")
    assert all("GeoReliab" in item["cwd"] for item in call_kwargs)
    assert all("PYTHONPATH" in item["env"] for item in call_kwargs)
    payload = json.loads(capsys.readouterr().out)
    assert payload["model_isolation"] == "per-frozen-env-python"
    for worker in payload["workers"]:
        assert Path(worker["worker_stdout_path"]).read_text(encoding="utf-8").startswith("upstream log noise")
        assert runner.sha256_file(Path(worker["worker_stdout_path"])) == worker["worker_stdout_sha256"]
        assert Path(worker["worker_stderr_path"]).exists()
    first_summary_paths = [Path(call[call.index("--summary-json") + 1]).name for call in calls]
    assert all("0of1" in name for name in first_summary_paths)
    calls.clear()
    assert runner.cli_main([
        "run-georeliab", "--stage", "smoke", "--model", "all", "--device", "cuda:0",
        "--shard", "1/2", "--config", str(config), "--output-root", str(root),
    ]) == 0
    second_summary_paths = [Path(call[call.index("--summary-json") + 1]).name for call in calls]
    assert all("1of2" in name for name in second_summary_paths)
    assert set(first_summary_paths).isdisjoint(second_summary_paths)
    calls.clear()
    assert runner.cli_main([
        "run-georeliab", "--stage", "smoke", "--model", "all", "--device", "cuda:0",
        "--shard", "0/1", "--config", str(config), "--output-root", str(root), "--dry-run",
    ]) == 0
    assert calls == []


def test_real_output_policy_requires_exact_non_home_overlay_root(tmp_path: Path):
    root = tmp_path / "runtime"
    root.mkdir()
    config = _overlay(root / "overlay.toml")
    runner._verify_output_root_policy(runner.RunnerContext(root=root, output_root=root, config_path=config, device="cuda:0"))
    with pytest.raises(runner.RunnerError, match="must equal"):
        runner._verify_output_root_policy(runner.RunnerContext(root=root, output_root=tmp_path / "other", config_path=config, device="cuda:0"))
    home_config = tmp_path / "home.toml"
    home_config.write_text("[runtime]\nroot='/home/smli/GeoReliab'\n", encoding="utf-8")
    with pytest.raises(runner.RunnerError, match="below /home"):
        runner._verify_output_root_policy(runner.RunnerContext(root=root, output_root=root, config_path=home_config, device="cuda:0"))
