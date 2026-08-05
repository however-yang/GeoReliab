from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from georeliab_mve.adapters import RenderedView
from georeliab_mve.artifact_storage import write_deterministic_npz
from georeliab_mve.contracts import (
    PredictionArtifact,
    read_json_artifact,
    RunManifest,
    RunMode,
    SampleKey,
    ScientificProvenance,
    ScientificValidity,
)
from georeliab_mve.v4_counterfactuals import (
    SCIENTIFIC_MODELS,
    SCIENTIFIC_STATES,
    TEST_SCENE_IDS,
    ScientificExecutionUnit,
    canonical_json_sha256,
)
from georeliab_mve.v4_metrics import NativeWarningCalibration
from georeliab_mve.v4_records import read_task_audit_record
from georeliab_mve.v4_science_lock import (
    V4_PROTOCOL_ID,
    V4_PROTOCOL_PROVENANCE_SCHEMA_VERSION,
    V4_PROTOCOL_SHA256,
    V4_PROTOCOL_VERSION,
)

import georeliab_mve.v4_attempt05_runtime as runtime


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unit(model_id: str = "VGGT", scene_id: int | None = None, state_id: str = "L3") -> ScientificExecutionUnit:
    scene = TEST_SCENE_IDS[0] if scene_id is None else scene_id
    provenance = {
        "schema_version": V4_PROTOCOL_PROVENANCE_SCHEMA_VERSION,
        "protocol_id": V4_PROTOCOL_ID,
        "protocol_version": V4_PROTOCOL_VERSION,
        "protocol_sha256": V4_PROTOCOL_SHA256,
    }
    payload = {
        "schema_version": "georeliab-v4-scientific-execution-unit-1.0",
        "protocol_provenance": provenance,
        "dataset": "DTU",
        "model_id": model_id,
        "scene_id": scene,
        "state_id": state_id,
        "state_identity_sha256": canonical_json_sha256({"state": [scene, state_id]}),
        "pair_identity_sha256": None if state_id == "L3" else canonical_json_sha256({"pair": [scene, state_id]}),
    }
    return ScientificExecutionUnit.from_dict(
        {**payload, "execution_unit_sha256": canonical_json_sha256(payload)}
    )


def _manifest(unit: ScientificExecutionUnit) -> RunManifest:
    return RunManifest(
        run_id=f"attempt05-{unit.model_id.lower()}-{unit.scene_id}-{unit.state_id}",
        mode=RunMode.REAL,
        scientific_validity=ScientificValidity.SCIENTIFIC,
        model=unit.model_id,
        checkpoint_hash="a" * 64,
        dataset="dtu",
        split="test",
        seed=0,
        intervention_version="none",
        corruption_version="georeliab-c-v1",
        environment={"device": "cuda:0", "python": "3.11", "torch": "frozen"},
        rgb_digest="rgb",
        prompt_digest="fixed-empty-prompt",
        decoder_digest="fixed-native-decoder",
        provenance=ScientificProvenance(
            project_commit="b" * 40,
            project_tree="c" * 40,
            model_source_commit="d" * 40,
            environment_lock_sha256="e" * 64,
            corruption_manifest_sha256="f" * 64,
            split_view_manifest_sha256="1" * 64,
            dust3r_source_commit="2" * 40 if unit.model_id == "MASt3R" else None,
            croco_source_commit="3" * 40 if unit.model_id == "MASt3R" else None,
        ),
    )


def _calibration(model_id: str = "VGGT") -> NativeWarningCalibration:
    scenes = tuple(scene for scene in range(1, 50) if scene not in TEST_SCENE_IDS)[:20]
    scores = tuple(float(index + 1) for index in range(20))
    split_sha = "4" * 64
    inventory_sha = "5" * 64
    payload = {
        "schema_version": "georeliab-v4-native-warning-calibration-1.0",
        "protocol_provenance": {
            "schema_version": V4_PROTOCOL_PROVENANCE_SCHEMA_VERSION,
            "protocol_id": V4_PROTOCOL_ID,
            "protocol_version": V4_PROTOCOL_VERSION,
            "protocol_sha256": V4_PROTOCOL_SHA256,
        },
        "model_id": model_id,
        "state_id": "L3",
        "scene_ids": list(scenes),
        "sorted_warning_scores": list(scores),
        "quantile_probability": 0.9,
        "quantile_method": "linear",
        "alarm_threshold": 18.1,
        "split_schema_version": "georeliab-v4-splits-1.0",
        "split_fingerprint_sha256": split_sha,
        "inventory_sha256": inventory_sha,
    }
    return NativeWarningCalibration(
        model_id=model_id,
        scene_ids=scenes,
        sorted_warning_scores=scores,
        alarm_threshold=18.1,
        split_fingerprint_sha256=split_sha,
        inventory_sha256=inventory_sha,
        calibration_identifier=canonical_json_sha256(payload),
    )



def _camera_stack() -> np.ndarray:
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], 8, axis=0)
    c2w[:, :3, 3] = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    return c2w

def _views(tmp_path: Path) -> tuple[RenderedView, ...]:
    rows = []
    for view_id in range(1, 9):
        path = tmp_path / f"{view_id}.png"
        path.write_bytes(b"png" + bytes([view_id]))
        rows.append(RenderedView(view_id=view_id, png_path=path, png_sha256=_sha_file(path)))
    return tuple(rows)


def _write_prediction(unit: ScientificExecutionUnit, tmp_path: Path, *, invalid: bool = False, finite: bool = True) -> PredictionArtifact:
    out = tmp_path / "adapter"
    out.mkdir(parents=True)
    points = np.stack([np.arange(8, dtype=np.float64), np.zeros(8), np.zeros(8)], axis=1)
    c2w = _camera_stack()
    conf = np.linspace(2.0, 9.0, 8, dtype=np.float64)
    if not finite:
        conf[1] = np.nan
    geom = out / "geometry.npz"
    confidence = out / "confidence.npz"
    mask = out / "mask.npz"
    write_deterministic_npz(
        geom,
        {
            "points_world": points,
            "camera_c2w": c2w,
            "intrinsics": np.repeat(np.eye(3, dtype=np.float64)[None], 8, axis=0),
            "pixel_xy": np.zeros((8, 2), dtype=np.float64),
            "view_id": np.arange(1, 9, dtype=np.int64),
        },
    )
    write_deterministic_npz(confidence, {"raw_confidence": conf})
    valid_mask = np.ones(8, dtype=bool)
    if not finite:
        valid_mask[1] = False
    write_deterministic_npz(mask, {"valid_mask": valid_mask})
    return PredictionArtifact(
        run_id=_manifest(unit).run_id,
        sample_key=str(SampleKey("dtu", "test", f"scan{unit.scene_id:03d}", "views-0001", unit.state_id, "0", "0")),
        geometry_prediction_uri=geom.resolve().as_uri(),
        native_confidence_uri=confidence.resolve().as_uri(),
        valid_mask_uri=mask.resolve().as_uri(),
        hook_location=None,
        runtime_seconds=1.25,
        peak_memory_mb=128.0,
        invalid_prediction=invalid,
        payload_digests={
            "geometry_prediction_uri": _sha_file(geom),
            "native_confidence_uri": _sha_file(confidence),
            "valid_mask_uri": _sha_file(mask),
        },
    )


class FakeAdapter:
    def __init__(self, prediction: PredictionArtifact) -> None:
        self.prediction = prediction
        self.calls = []

    def predict_sample(self, manifest, sample_key, rendered_views):
        self.calls.append((manifest, sample_key, tuple(rendered_views)))
        return self.prediction


def test_cpu_input_closure_counts_full_attempt05_population() -> None:
    schedule = tuple(
        _unit(model, scene, state)
        for model in SCIENTIFIC_MODELS
        for scene in TEST_SCENE_IDS
        for state in SCIENTIFIC_STATES
    )
    assert runtime.build_cpu_input_closure(
        calibration_l3_units=tuple(object() for _ in range(40)),
        model_independent_states=tuple(object() for _ in range(200)),
        scientific_units=schedule,
        rectified_bindings=tuple(object() for _ in range(960)),
    ) == {
        "calibration_l3_units": 40,
        "model_independent_states": 200,
        "scientific_units": 400,
        "schedule_units": 400,
        "rectified_non_l3_members": 960,
        "test_l3_units": 40,
        "non_l3_scientific_units": 360,
        "fog_bindings_to_l3": 120,
    }


def test_dtu_projection_rq_decomposition_preserves_view_order_and_det_r() -> None:
    projection = np.array(
        [[1000.0, 0.0, 320.0, -640.0], [0.0, 900.0, 240.0, 120.0], [0.0, 0.0, 1.0, 2.0]],
        dtype=np.float64,
    )
    by_view = {
        view_id: projection
        + np.array([[0, 0, 0, view_id * 0.01], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64)
        for view_id in range(1, 9)
    }
    decomposed = runtime.decompose_ordered_dtu_projections(by_view, ordered_view_ids=(8, 7, 6, 5, 4, 3, 2, 1))
    assert len(decomposed) == 8
    assert decomposed[0].view_id == 8
    for item in decomposed:
        rotation = np.array(item.world_to_camera_rotation)
        assert np.isclose(np.linalg.det(rotation), 1.0)
        assert item.max_reprojection_abs_error < 1e-7
        assert np.allclose(np.array(item.camera_to_world)[:3, :3], rotation.T)


def test_execute_unit_calls_adapter_directly_and_writes_valid_task_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    unit = _unit()
    manifest = _manifest(unit)
    prediction = _write_prediction(unit, tmp_path)
    adapter = FakeAdapter(prediction)
    called = {"runner": False}
    monkeypatch.setattr("georeliab_mve.runner.execute_item", lambda *a, **k: called.__setitem__("runner", True))

    result = runtime.execute_attempt05_unit(
        unit=unit,
        manifest=manifest,
        sample_key=SampleKey.parse(prediction.sample_key),
        rendered_views=_views(tmp_path),
        adapter=adapter,
        calibration=_calibration(unit.model_id),
        output_dir=tmp_path / "unit",
        gt_points=np.stack([np.arange(8, dtype=np.float64), np.zeros(8), np.zeros(8)], axis=1),
        gt_camera_c2w=_camera_stack(),
        observability_mask=np.ones(8, dtype=bool),
        gt_dtu_camera_c2w=_camera_stack(),
    )

    assert called["runner"] is False
    assert len(adapter.calls) == 1
    assert result.status == "VALID_COMPLETE"
    record = read_task_audit_record(result.record_path)
    assert record.valid is True
    assert record.reason_code == "VALID"
    assert record.ordered_view_ids == tuple(range(1, 9))
    stored_prediction = read_json_artifact(result.record_path.parent / "prediction_artifact.json", PredictionArtifact)
    assert ".partial" not in stored_prediction.geometry_prediction_uri
    assert Path(runtime._file_uri_path(stored_prediction.geometry_prediction_uri, "geometry")).parent == result.record_path.parent
    with zipfile.ZipFile(runtime._file_uri_path(stored_prediction.geometry_prediction_uri, "geometry")) as archive:
        assert archive.infolist()
        assert all(item.compress_type == zipfile.ZIP_DEFLATED for item in archive.infolist())


def test_adapter_exception_and_partial_fail_closed(tmp_path: Path) -> None:
    class RaisingAdapter:
        def predict_sample(self, *_args):
            raise RuntimeError("boom")

    unit = _unit()
    partial = tmp_path / "unit" / "task_audit_record.json.partial"
    partial.parent.mkdir()
    partial.write_text("{}", encoding="utf-8")
    with pytest.raises(runtime.Attempt05RuntimeError, match="V4_ATTEMPT05_PARTIAL_EXISTS"):
        runtime.execute_attempt05_unit(
            unit=unit,
            manifest=_manifest(unit),
            sample_key=SampleKey("dtu", "test", f"scan{unit.scene_id:03d}", "views-0001", "L3", "0", "0"),
            rendered_views=_views(tmp_path),
            adapter=RaisingAdapter(),
            calibration=_calibration(unit.model_id),
            output_dir=tmp_path / "unit",
            gt_points=np.zeros((1, 3)),
            gt_camera_c2w=_camera_stack(),
            observability_mask=np.ones(1, dtype=bool),
            gt_dtu_camera_c2w=_camera_stack(),
        )

    clean_out = tmp_path / "clean-unit"
    with pytest.raises(runtime.Attempt05RuntimeError, match="V4_ATTEMPT05_ADAPTER_EXCEPTION"):
        runtime.execute_attempt05_unit(
            unit=unit,
            manifest=_manifest(unit),
            sample_key=SampleKey("dtu", "test", f"scan{unit.scene_id:03d}", "views-0001", "L3", "0", "0"),
            rendered_views=_views(tmp_path),
            adapter=RaisingAdapter(),
            calibration=_calibration(unit.model_id),
            output_dir=clean_out,
            gt_points=np.zeros((1, 3)),
            gt_camera_c2w=_camera_stack(),
            observability_mask=np.ones(1, dtype=bool),
            gt_dtu_camera_c2w=_camera_stack(),
        )
    assert not (clean_out / "task_audit_record.json").exists()


def test_invalid_prediction_uses_frozen_sentinel_without_native_warning(tmp_path: Path) -> None:
    unit = _unit()
    finite_prediction = _write_prediction(unit, tmp_path / "finite", invalid=True)
    result = runtime.execute_attempt05_unit(
        unit=unit,
        manifest=_manifest(unit),
        sample_key=SampleKey.parse(finite_prediction.sample_key),
        rendered_views=_views(tmp_path),
        adapter=FakeAdapter(finite_prediction),
        calibration=_calibration(unit.model_id),
        output_dir=tmp_path / "invalid-finite",
        gt_points=np.zeros((1, 3)),
        gt_camera_c2w=_camera_stack(),
        observability_mask=np.ones(1, dtype=bool),
        gt_dtu_camera_c2w=_camera_stack(),
    )
    assert read_task_audit_record(result.record_path).valid is False

    class ForbiddenAdapter:
        def predict_sample(self, *_args):
            raise AssertionError("invalid resume must not invoke adapter")

    resumed = runtime.execute_attempt05_unit(
        unit=unit,
        manifest=_manifest(unit),
        sample_key=SampleKey.parse(finite_prediction.sample_key),
        rendered_views=_views(tmp_path),
        adapter=ForbiddenAdapter(),
        calibration=_calibration(unit.model_id),
        output_dir=tmp_path / "invalid-finite",
        gt_points=np.zeros((1, 3)),
        gt_camera_c2w=_camera_stack(),
        observability_mask=np.ones(1, dtype=bool),
        gt_dtu_camera_c2w=_camera_stack(),
        resume=True,
    )
    assert resumed.status == "RESUMED_INVALID_FAILURE_RECORDED"
    assert resumed.record.valid is False

    nonfinite_prediction = _write_prediction(unit, tmp_path / "nonfinite", invalid=True, finite=False)
    nonfinite = runtime.execute_attempt05_unit(
        unit=unit,
        manifest=_manifest(unit),
        sample_key=SampleKey.parse(nonfinite_prediction.sample_key),
        rendered_views=_views(tmp_path),
        adapter=FakeAdapter(nonfinite_prediction),
        calibration=_calibration(unit.model_id),
        output_dir=tmp_path / "invalid-nonfinite",
        gt_points=np.zeros((1, 3)),
        gt_camera_c2w=_camera_stack(),
        observability_mask=np.ones(1, dtype=bool),
        gt_dtu_camera_c2w=_camera_stack(),
    )
    assert nonfinite.record.native_warning_score == 1e12
    assert nonfinite.record.valid is False


def test_resume_missing_output_executes_existing_valid_skips(tmp_path: Path) -> None:
    unit = _unit()
    prediction = _write_prediction(unit, tmp_path)
    output_dir = tmp_path / "unit"
    missing_out = tmp_path / "resume-missing"
    missing_adapter = FakeAdapter(prediction)
    executed = runtime.execute_attempt05_unit(
        unit=unit,
        manifest=_manifest(unit),
        sample_key=SampleKey.parse(prediction.sample_key),
        rendered_views=_views(tmp_path),
        adapter=missing_adapter,
        calibration=_calibration(unit.model_id),
        output_dir=missing_out,
        gt_points=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        gt_camera_c2w=_camera_stack(),
        observability_mask=np.ones(8, dtype=bool),
        gt_dtu_camera_c2w=_camera_stack(),
        resume=True,
    )
    assert executed.status == "VALID_COMPLETE"
    assert len(missing_adapter.calls) == 1

    first = runtime.execute_attempt05_unit(
        unit=unit,
        manifest=_manifest(unit),
        sample_key=SampleKey.parse(prediction.sample_key),
        rendered_views=_views(tmp_path),
        adapter=FakeAdapter(prediction),
        calibration=_calibration(unit.model_id),
        output_dir=output_dir,
        gt_points=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        gt_camera_c2w=_camera_stack(),
        observability_mask=np.ones(8, dtype=bool),
        gt_dtu_camera_c2w=_camera_stack(),
    )
    before = _sha_file(first.record_path)

    class ForbiddenAdapter:
        def predict_sample(self, *_args):
            raise AssertionError("resume must not invoke adapter")

    resumed = runtime.execute_attempt05_unit(
        unit=unit,
        manifest=_manifest(unit),
        sample_key=SampleKey.parse(prediction.sample_key),
        rendered_views=_views(tmp_path),
        adapter=ForbiddenAdapter(),
        calibration=_calibration(unit.model_id),
        output_dir=output_dir,
        gt_points=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        gt_camera_c2w=_camera_stack(),
        observability_mask=np.ones(8, dtype=bool),
        gt_dtu_camera_c2w=_camera_stack(),
        resume=True,
    )
    assert resumed.status == "RESUMED_VALID_COMPLETE"
    assert _sha_file(resumed.record_path) == before


def test_calibration_l3_bridge_writes_warning_evidence_without_task_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit = _unit()
    manifest = _manifest(unit)
    prediction = _write_prediction(unit, tmp_path)
    adapter = FakeAdapter(prediction)
    output_dir = tmp_path / "calibration"
    called = {"runner": False}
    monkeypatch.setattr("georeliab_mve.runner.execute_item", lambda *a, **k: called.__setitem__("runner", True))

    result = runtime.execute_attempt05_calibration_l3(
        manifest=manifest,
        sample_key=SampleKey.parse(prediction.sample_key),
        model_id=unit.model_id,
        scene_id=unit.scene_id,
        rendered_views=_views(tmp_path),
        adapter=adapter,
        output_dir=output_dir,
    )

    assert called["runner"] is False
    assert len(adapter.calls) == 1
    assert result.status == "CALIBRATION_WARNING_RECORDED"
    assert result.evidence_path == output_dir / "native_warning_evidence.json"
    assert np.isfinite(result.warning_score)
    assert not (output_dir / "task_audit_record.json").exists()
    assert not (output_dir / "audit_record.json").exists()
    assert (output_dir / "run_manifest.json").is_file()
    stored_prediction = read_json_artifact(output_dir / "prediction_artifact.json", PredictionArtifact)
    assert ".partial" not in stored_prediction.geometry_prediction_uri
    assert Path(runtime._file_uri_path(stored_prediction.geometry_prediction_uri, "geometry")).parent == output_dir
    evidence = json.loads(result.evidence_path.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == runtime.CALIBRATION_WARNING_EVIDENCE_SCHEMA
    assert evidence["calibration_only"] is True
    assert evidence["scientific_record_created"] is False
    assert evidence["state_id"] == "L3"
    assert evidence["ordered_view_ids"] == list(range(1, 9))
    assert evidence["warning_score"] == result.warning_score
    assert evidence["evidence_sha256"] == result.evidence_sha256
    assert evidence["prediction_payload_digests"] == dict(stored_prediction.payload_digests)
    with zipfile.ZipFile(runtime._file_uri_path(stored_prediction.native_confidence_uri, "confidence")) as archive:
        assert archive.infolist()
        assert all(item.compress_type == zipfile.ZIP_DEFLATED for item in archive.infolist())


def test_calibration_l3_resume_is_read_only_and_validates_bundle(tmp_path: Path) -> None:
    unit = _unit()
    prediction = _write_prediction(unit, tmp_path)
    output_dir = tmp_path / "calibration"
    first = runtime.execute_attempt05_calibration_l3(
        manifest=_manifest(unit),
        sample_key=SampleKey.parse(prediction.sample_key),
        model_id=unit.model_id,
        scene_id=unit.scene_id,
        rendered_views=_views(tmp_path),
        adapter=FakeAdapter(prediction),
        output_dir=output_dir,
    )
    before = _sha_file(first.evidence_path)

    class ForbiddenAdapter:
        def predict_sample(self, *_args):
            raise AssertionError("calibration resume must not invoke adapter")

    with pytest.raises(runtime.Attempt05RuntimeError, match="V4_ATTEMPT05_CALIBRATION_ALREADY_EXISTS"):
        runtime.execute_attempt05_calibration_l3(
            manifest=_manifest(unit),
            sample_key=SampleKey.parse(prediction.sample_key),
            model_id=unit.model_id,
            scene_id=unit.scene_id,
            rendered_views=_views(tmp_path),
            adapter=ForbiddenAdapter(),
            output_dir=output_dir,
        )

    resumed = runtime.execute_attempt05_calibration_l3(
        manifest=_manifest(unit),
        sample_key=SampleKey.parse(prediction.sample_key),
        model_id=unit.model_id,
        scene_id=unit.scene_id,
        rendered_views=_views(tmp_path),
        adapter=ForbiddenAdapter(),
        output_dir=output_dir,
        resume=True,
    )

    assert resumed.status == "CALIBRATION_RESUMED_VALID"
    assert resumed.warning_score == first.warning_score
    assert resumed.evidence_sha256 == first.evidence_sha256
    assert _sha_file(resumed.evidence_path) == before
    assert not (output_dir / "task_audit_record.json").exists()
    assert not (output_dir / "audit_record.json").exists()


def test_calibration_l3_fail_closed_on_partial_exception_and_nonfinite_warning(tmp_path: Path) -> None:
    unit = _unit()
    manifest = _manifest(unit)
    sample_key = SampleKey("dtu", "test", f"scan{unit.scene_id:03d}", "views-0001", "L3", "0", "0")
    partial_out = tmp_path / "partial-calibration"
    partial_out.with_name(partial_out.name + ".partial").mkdir()
    with pytest.raises(runtime.Attempt05RuntimeError, match="V4_ATTEMPT05_PARTIAL_EXISTS"):
        runtime.execute_attempt05_calibration_l3(
            manifest=manifest,
            sample_key=sample_key,
            model_id=unit.model_id,
            scene_id=unit.scene_id,
            rendered_views=_views(tmp_path),
            adapter=FakeAdapter(_write_prediction(unit, tmp_path / "partial-pred")),
            output_dir=partial_out,
        )

    class RaisingAdapter:
        def predict_sample(self, *_args):
            raise RuntimeError("boom")

    exception_out = tmp_path / "exception-calibration"
    with pytest.raises(runtime.Attempt05RuntimeError, match="V4_ATTEMPT05_ADAPTER_EXCEPTION"):
        runtime.execute_attempt05_calibration_l3(
            manifest=manifest,
            sample_key=sample_key,
            model_id=unit.model_id,
            scene_id=unit.scene_id,
            rendered_views=_views(tmp_path),
            adapter=RaisingAdapter(),
            output_dir=exception_out,
        )
    assert not (exception_out / "native_warning_evidence.json").exists()
    assert exception_out.with_name(exception_out.name + ".partial").exists()

    invalid_prediction = _write_prediction(unit, tmp_path / "invalid-calibration", invalid=True, finite=False)
    invalid_out = tmp_path / "invalid-calibration-out"
    with pytest.raises(runtime.Attempt05RuntimeError, match="V4_ATTEMPT05_CALIBRATION_INVALID_PREDICTION"):
        runtime.execute_attempt05_calibration_l3(
            manifest=manifest,
            sample_key=SampleKey.parse(invalid_prediction.sample_key),
            model_id=unit.model_id,
            scene_id=unit.scene_id,
            rendered_views=_views(tmp_path),
            adapter=FakeAdapter(invalid_prediction),
            output_dir=invalid_out,
        )
    assert not (invalid_out / "native_warning_evidence.json").exists()

    nonfinite_prediction = _write_prediction(unit, tmp_path / "nonfinite-calibration", finite=False)
    nonfinite_out = tmp_path / "nonfinite-calibration-out"
    with pytest.raises(runtime.Attempt05RuntimeError, match="V4_MVE_BLOCKED_NATIVE_WARNING_UNAVAILABLE"):
        runtime.execute_attempt05_calibration_l3(
            manifest=manifest,
            sample_key=SampleKey.parse(nonfinite_prediction.sample_key),
            model_id=unit.model_id,
            scene_id=unit.scene_id,
            rendered_views=_views(tmp_path),
            adapter=FakeAdapter(nonfinite_prediction),
            output_dir=nonfinite_out,
        )
    assert not (nonfinite_out / "native_warning_evidence.json").exists()
    assert not (nonfinite_out / "task_audit_record.json").exists()



@pytest.mark.parametrize(
    ("failure_stage", "reason_code"),
    [
        ("audit", "V4_ATTEMPT05_GT_ARRAY_AUDIT_FAILED"),
        ("point", "V4_ATTEMPT05_POINT_METRIC_FAILED"),
        ("pose", "V4_ATTEMPT05_POSE_METRIC_FAILED"),
        ("record", "V4_ATTEMPT05_TASK_RECORD_BUILD_FAILED"),
    ],
)
def test_prediction_record_failures_preserve_split_failure_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    reason_code: str,
) -> None:
    unit = _unit()
    prediction = _write_prediction(unit, tmp_path / "prediction")
    original_audit = runtime.audit_prediction_arrays
    original_point = runtime.compute_point_task_metrics
    original_pose = runtime.compute_relative_pose_metrics
    original_record = runtime.build_task_audit_record

    def fail(*_args, **_kwargs):
        raise ValueError(f"injected-{failure_stage}")

    if failure_stage == "audit":
        monkeypatch.setattr(runtime, "audit_prediction_arrays", fail)
    elif failure_stage == "point":
        monkeypatch.setattr(runtime, "compute_point_task_metrics", fail)
    elif failure_stage == "pose":
        monkeypatch.setattr(runtime, "compute_relative_pose_metrics", fail)
    else:
        monkeypatch.setattr(runtime, "build_task_audit_record", fail)

    with pytest.raises(runtime.Attempt05RuntimeError) as captured:
        runtime.execute_attempt05_unit(
            unit=unit,
            manifest=_manifest(unit),
            sample_key=SampleKey.parse(prediction.sample_key),
            rendered_views=_views(tmp_path),
            adapter=FakeAdapter(prediction),
            calibration=_calibration(unit.model_id),
            output_dir=tmp_path / f"out-{failure_stage}",
            gt_points=np.stack([np.arange(8, dtype=np.float64), np.zeros(8), np.zeros(8)], axis=1),
            gt_camera_c2w=_camera_stack(),
            observability_mask=np.ones(8, dtype=bool),
            gt_dtu_camera_c2w=_camera_stack(),
        )
    error = captured.value
    assert str(error) == reason_code
    assert error.failure_envelope is not None
    assert error.failure_envelope.reason_code == reason_code
    assert error.failure_envelope.stage in {
        "gt_array_audit",
        "point_metrics",
        "pose_metrics",
        "task_record_build",
    }
    assert f"injected-{failure_stage}" in error.failure_envelope.traceback_text
    assert original_audit is not None and original_point is not None
    assert original_pose is not None and original_record is not None
