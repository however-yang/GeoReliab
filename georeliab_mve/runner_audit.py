"""Runner-facing production audit binding for frozen DTU evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping
from urllib.parse import unquote, urlparse

import numpy as np

from .audit import (
    AuditError,
    audit_prediction_arrays,
    load_official_dtu_evidence,
    model_risk_from_confidence,
)
from .contracts import AuditRecord, PredictionArtifact, RunManifest, validate_artifact_bundle


_DENSE_AUDIT_KEYS = (
    "voxel_points",
    "raw_confidence",
    "risk",
    "gt_error",
    "failure_label",
    "provenance_count",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_file_uri_path(uri: str, field_name: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise AuditError(f"{field_name} must be a local file URI")
    path_text = unquote(parsed.path)
    if len(path_text) >= 3 and path_text[0] == "/" and path_text[2] == ":":
        path_text = path_text[1:]
    return Path(path_text)


def _load_npz_uri(uri: str, field_name: str) -> dict[str, np.ndarray]:
    path = _local_file_uri_path(uri, field_name)
    try:
        with np.load(path, allow_pickle=False) as payload:
            return {name: payload[name] for name in payload.files}
    except (OSError, ValueError) as exc:
        raise AuditError(f"cannot read {field_name} NPZ payload: {exc}") from exc


def _atomic_savez(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.savez(handle, **dict(payload))
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    partial.replace(path)


def _dense_payload(result, *, invalid_prediction: bool) -> dict[str, np.ndarray]:
    if invalid_prediction:
        return {
            "voxel_points": np.empty((0, 3), dtype=np.float64),
            "raw_confidence": np.empty((0,), dtype=np.float64),
            "risk": np.empty((0,), dtype=np.float64),
            "gt_error": np.empty((0,), dtype=np.float64),
            "failure_label": np.empty((0,), dtype=bool),
            "provenance_count": np.empty((0,), dtype=np.int64),
        }
    return {key: np.asarray(getattr(result, key)) for key in _DENSE_AUDIT_KEYS}


def _json_metadata(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _evidence_summary(evidence: Mapping[str, object]) -> dict[str, object]:
    provenance = evidence.get("provenance")
    return {
        "scene": evidence.get("scene"),
        "scene_id": evidence.get("scene_id"),
        "split": evidence.get("split"),
        "view_ids": list(evidence.get("view_ids", ())),
        "gt_point_count": int(len(np.asarray(evidence["gt_points"]))),
        "gt_camera_count": int(len(np.asarray(evidence["gt_camera_centers"]))),
        "obs_mask_shape": list(np.asarray(evidence["obs_mask"]).shape),
        "obs_res": float(evidence["obs_res"]),
        "provenance": provenance if isinstance(provenance, Mapping) else {},
    }


def _audit_metadata(dense_path: Path, gt_points_path: Path, result, evidence: Mapping[str, object]) -> dict[str, str]:
    return {
        "dense_audit_uri": dense_path.resolve().as_uri(),
        "dense_audit_sha256": _sha256_file(dense_path),
        "gt_points_uri": gt_points_path.resolve().as_uri(),
        "gt_points_sha256": _sha256_file(gt_points_path),
        "audit_summary": _json_metadata(result.summary),
        "official_dtu_evidence": _json_metadata(_evidence_summary(evidence)),
    }

def audit_prediction_with_frozen_dtu(
    *,
    root: Path,
    manifest: RunManifest,
    prediction: PredictionArtifact,
    output_dir: Path,
) -> AuditRecord:
    """Audit one runner prediction against the verified frozen DTU materialization."""

    evidence = load_official_dtu_evidence(
        sample_key=prediction.sample_key,
        frozen_materialization=root / "manifests" / "frozen_materialization.json",
        split_manifest=root / "manifests" / "split_view_manifest.json",
    )
    if prediction.invalid_prediction:
        result = audit_prediction_arrays(
            points_world=np.empty((0, 3), dtype=np.float64),
            pred_camera_centers=np.empty((0, 3), dtype=np.float64),
            gt_camera_centers=np.empty((0, 3), dtype=np.float64),
            raw_confidence=np.empty((0,), dtype=np.float64),
            risk=np.empty((0,), dtype=np.float64),
            valid_mask=np.empty((0,), dtype=bool),
            gt_points=np.empty((0, 3), dtype=np.float64),
            observability_mask=np.empty((0,), dtype=bool),
            invalid_prediction=True,
        )
    else:
        geometry = _load_npz_uri(
            prediction.geometry_prediction_uri, "geometry_prediction_uri"
        )
        confidence = _load_npz_uri(
            prediction.native_confidence_uri, "native_confidence_uri"
        )
        mask = _load_npz_uri(prediction.valid_mask_uri, "valid_mask_uri")
        raw_confidence = np.asarray(confidence["raw_confidence"])
        camera_c2w = np.asarray(geometry["camera_c2w"], dtype=np.float64)
        result = audit_prediction_arrays(
            points_world=geometry["points_world"],
            pred_camera_centers=camera_c2w[:, :3, 3],
            gt_camera_centers=evidence["gt_camera_centers"],
            raw_confidence=raw_confidence,
            risk=model_risk_from_confidence(manifest.model, raw_confidence),
            valid_mask=mask["valid_mask"],
            gt_points=evidence["gt_points"],
            observability_mask=evidence["obs_mask"],
            observability_bb=evidence["obs_bb"],
            observability_res=float(evidence["obs_res"]),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    dense_path = output_dir / "dense_audit.npz"
    gt_points_path = output_dir / "gt_points.npz"
    _atomic_savez(dense_path, _dense_payload(result, invalid_prediction=prediction.invalid_prediction))
    _atomic_savez(gt_points_path, {"gt_points": np.asarray(evidence["gt_points"])})
    metadata = _audit_metadata(dense_path, gt_points_path, result, evidence)

    if prediction.invalid_prediction:
        audit = AuditRecord(
            run_id=manifest.run_id,
            sample_key=prediction.sample_key,
            gt_error=1e12,
            failure_label=True,
            selection_score=1e12,
            coverage=0.0,
            accepted=False,
            downstream_outcome=0.0,
            invalid_prediction=True,
            metadata=metadata,
        )
    else:
        audit = AuditRecord(
            run_id=manifest.run_id,
            sample_key=prediction.sample_key,
            gt_error=float(np.median(result.gt_error)),
            failure_label=bool(np.any(result.failure_label)),
            selection_score=float(np.median(result.risk)),
            coverage=1.0,
            accepted=True,
            downstream_outcome=float(result.summary["fscore_2mm"]),
            invalid_prediction=False,
            metadata=metadata,
        )
    validate_artifact_bundle(manifest, prediction, audit)
    return audit
