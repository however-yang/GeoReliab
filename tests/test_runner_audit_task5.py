from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import georeliab_mve.runner_audit as runner_audit
from georeliab_mve.contracts import (
    PredictionArtifact,
    RunManifest,
    RunMode,
    ScientificProvenance,
    ScientificValidity,
    validate_artifact_bundle,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance() -> ScientificProvenance:
    return ScientificProvenance(
        project_commit="1" * 40,
        project_tree="2" * 40,
        model_source_commit="3" * 40,
        environment_lock_sha256="4" * 64,
        corruption_manifest_sha256="5" * 64,
        split_view_manifest_sha256="6" * 64,
    )


def _manifest(*, mode: RunMode, split: str) -> RunManifest:
    validity = (
        ScientificValidity.NON_SCIENTIFIC_SMOKE
        if mode is RunMode.SMOKE
        else ScientificValidity.SCIENTIFIC
    )
    return RunManifest(
        run_id=f"{mode.value}-{split}-audit",
        mode=mode,
        scientific_validity=validity,
        model="VGGT",
        checkpoint_hash="7" * 64,
        dataset="dtu",
        split=split,
        seed=0,
        intervention_version="none",
        corruption_version="georeliab-c-v1",
        environment={"python": "3.10.20", "torch": "2.3.1+cu121"},
        rgb_digest="rgb",
        prompt_digest="prompt",
        decoder_digest="decoder",
        provenance=_provenance(),
    )


def _prediction(
    manifest: RunManifest, output_dir: Path, *, invalid: bool = False
) -> PredictionArtifact:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_key = f"dtu/{manifest.split}/scan201/fps8/clean/0/0"
    camera_c2w = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 8, axis=0)
    camera_c2w[:, :3, 3] = np.array(
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
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    pixel_xy = np.array([[10.0, 10.0], [11.0, 11.0]], dtype=np.float64)
    view_id = np.array([1, 3], dtype=np.int64)
    raw_confidence = np.array([2.0, 4.0], dtype=np.float64)
    valid_mask = np.array([True, True], dtype=bool)
    if invalid:
        points = np.empty((0, 3), dtype=np.float64)
        pixel_xy = np.empty((0, 2), dtype=np.float64)
        view_id = np.empty((0,), dtype=np.int64)
        raw_confidence = np.empty((0,), dtype=np.float64)
        valid_mask = np.empty((0,), dtype=bool)
    geometry = output_dir / "geometry.npz"
    confidence = output_dir / "confidence.npz"
    mask = output_dir / "valid_mask.npz"
    np.savez(
        geometry,
        points_world=points,
        camera_c2w=camera_c2w,
        intrinsics=np.repeat(np.eye(3, dtype=np.float64)[None, :, :], 8, axis=0),
        pixel_xy=pixel_xy,
        view_id=view_id,
    )
    np.savez(confidence, raw_confidence=raw_confidence)
    np.savez(mask, valid_mask=valid_mask)
    return PredictionArtifact(
        run_id=manifest.run_id,
        sample_key=sample_key,
        geometry_prediction_uri=geometry.resolve().as_uri(),
        native_confidence_uri=confidence.resolve().as_uri(),
        valid_mask_uri=mask.resolve().as_uri(),
        hook_location=None,
        runtime_seconds=1.0,
        peak_memory_mb=128.0,
        invalid_prediction=invalid,
        payload_digests={
            "geometry_prediction_uri": _sha256_file(geometry),
            "native_confidence_uri": _sha256_file(confidence),
            "valid_mask_uri": _sha256_file(mask),
        },
    )


def _fake_evidence(*, sample_key: str, frozen_materialization: Path, split_manifest: Path):
    assert frozen_materialization.name == "frozen_materialization.json"
    assert split_manifest.name == "split_view_manifest.json"
    assert sample_key.startswith("dtu/")
    camera_centers = np.array(
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
    return {
        "gt_points": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64),
        "gt_camera_centers": camera_centers,
        "obs_mask": np.ones((3, 3, 3), dtype=bool),
        "obs_bb": np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], dtype=np.float64),
        "obs_res": 1.0,
    }


def test_runner_audit_writes_dense_bundle_and_gt_provenance(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(runner_audit, "load_official_dtu_evidence", _fake_evidence)
    manifest = _manifest(mode=RunMode.REAL, split="dev")
    prediction = _prediction(manifest, tmp_path / "prediction")

    audit = runner_audit.audit_prediction_with_frozen_dtu(
        root=tmp_path / "root",
        manifest=manifest,
        prediction=prediction,
        output_dir=tmp_path / "audit",
    )

    validate_artifact_bundle(manifest, prediction, audit)
    dense_path = Path(audit.metadata["dense_audit_uri"].replace("file:///", ""))
    with np.load(dense_path, allow_pickle=False) as dense:
        assert tuple(dense.files) == (
            "voxel_points",
            "raw_confidence",
            "risk",
            "gt_error",
            "failure_label",
            "provenance_count",
        )
        assert dense["voxel_points"].shape == (2, 3)
        np.testing.assert_allclose(dense["raw_confidence"], [2.0, 4.0])
        assert dense["failure_label"].tolist() == [False, False]
    gt_path = Path(audit.metadata["gt_points_uri"].replace("file:///", ""))
    assert audit.metadata["gt_points_sha256"] == _sha256_file(gt_path)
    assert audit.metadata["dense_audit_sha256"] == _sha256_file(dense_path)
    summary = json.loads(audit.metadata["audit_summary"])
    evidence = json.loads(audit.metadata["official_dtu_evidence"])
    assert summary["invalid_prediction"] is False
    assert evidence["gt_point_count"] == 2
    assert evidence["gt_camera_count"] == 8
    assert audit.invalid_prediction is False
    assert audit.accepted is True


def test_runner_audit_keeps_invalid_prediction_as_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(runner_audit, "load_official_dtu_evidence", _fake_evidence)
    manifest = _manifest(mode=RunMode.REAL, split="dev")
    prediction = _prediction(manifest, tmp_path / "prediction", invalid=True)

    audit = runner_audit.audit_prediction_with_frozen_dtu(
        root=tmp_path / "root",
        manifest=manifest,
        prediction=prediction,
        output_dir=tmp_path / "audit",
    )

    validate_artifact_bundle(manifest, prediction, audit)
    dense_path = Path(audit.metadata["dense_audit_uri"].replace("file:///", ""))
    with np.load(dense_path, allow_pickle=False) as dense:
        assert dense["voxel_points"].shape == (0, 3)
        assert dense["gt_error"].shape == (0,)
        assert dense["failure_label"].shape == (0,)
    summary = json.loads(audit.metadata["audit_summary"])
    assert summary["invalid_prediction"] is True
    assert summary["fscore_2mm"] == 0.0
    assert audit.invalid_prediction is True
    assert audit.gt_error == 1e12
    assert audit.selection_score == 1e12
    assert audit.failure_label is True
    assert audit.accepted is False
    assert audit.downstream_outcome == 0.0


def test_runner_audit_preserves_smoke_scientific_validity(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(runner_audit, "load_official_dtu_evidence", _fake_evidence)
    manifest = _manifest(mode=RunMode.SMOKE, split="dev")
    prediction = _prediction(manifest, tmp_path / "prediction")

    audit = runner_audit.audit_prediction_with_frozen_dtu(
        root=tmp_path / "root",
        manifest=manifest,
        prediction=prediction,
        output_dir=tmp_path / "audit",
    )

    validate_artifact_bundle(manifest, prediction, audit)
    assert manifest.scientific_validity is ScientificValidity.NON_SCIENTIFIC_SMOKE
    assert audit.metadata["gt_points_uri"]
