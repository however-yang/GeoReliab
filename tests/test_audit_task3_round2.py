from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from georeliab_mve.audit import (
    AuditError,
    dtu_observability_mask_for_points,
    model_risk_from_confidence,
    load_official_dtu_evidence,
)
from georeliab_mve.cli import main
from georeliab_mve.contracts import (
    AuditRecord,
    PredictionArtifact,
    RunManifest,
    RunMode,
    ScientificProvenance,
    ScientificValidity,
    read_json_artifact,
    validate_artifact_bundle,
    write_json_artifact,
)
from georeliab_mve.gates import (
    DownstreamHarmEvidence,
    GateStatus,
    GeoReliabConditionEvidence,
    GeoReliabGateInput,
    ZeroUpdateEvidence,
    evaluate_georeliab_gate,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(model: str = 'VGGT', *, invalid=False) -> tuple[RunManifest, PredictionArtifact, Path, Path, Path]:
    raise RuntimeError('test helper must be monkeypatched per tmp_path')


def _provenance() -> ScientificProvenance:
    return ScientificProvenance(
        project_commit='a' * 40,
        project_tree='b' * 40,
        model_source_commit='c' * 40,
        environment_lock_sha256='d' * 64,
        corruption_manifest_sha256='e' * 64,
        split_view_manifest_sha256='f' * 64,
    )


def _condition(model: str, corruption: str) -> GeoReliabConditionEvidence:
    scenes = tuple(f's{i:02d}' for i in range(20))
    return GeoReliabConditionEvidence(
        model=model,
        corruption=corruption,
        clean_rho=0.8,
        severity_rhos=(0.6, 0.4, 0.2),
        failure_auroc=0.70,
        corruption_severity_monotonic=True,
        cross_view_consistent=True,
        gt_geometry_invariant=True,
        relative_decline_ci_lower=0.60,
        scene_ids=scenes,
        scene_count=20,
        n_resamples=10_000,
        relative_decline_raw_p=0.001,
        relative_decline_adjusted_p=0.006,
        relative_decline_holm_rejected=True,
        branch_reason_codes=('RELATIVE_DECLINE_HOLM_SUPPORTED',),
    )


def _gate_input(**overrides) -> GeoReliabGateInput:
    conditions = tuple(
        _condition(model, corruption)
        for model in ('VGGT', 'MASt3R')
        for corruption in ('fog', 'low-light-noise', 'defocus')
    )
    fields = {
        'scientific_validity': ScientificValidity.SCIENTIFIC,
        'required_models_ready': ('VGGT', 'MASt3R'),
        'required_datasets_ready': True,
        'tartanair_native_fog_sanity': True,
        'conditions': conditions,
        'downstream_harm': tuple(
            DownstreamHarmEvidence(model, f'{corruption}-s2', -0.02 if corruption in ('fog', 'defocus') else 0.01, -0.01 if corruption in ('fog', 'defocus') else 0.02)
            for model in ('VGGT', 'MASt3R')
            for corruption in ('fog', 'low-light-noise', 'defocus')
        ),
        'zero_update': tuple(
            ZeroUpdateEvidence(model, f'{corruption}-s2', 0.10 if model == 'VGGT' and corruption == 'fog' else 0.0, 0.01 if model == 'VGGT' and corruption == 'fog' else -0.01)
            for model in ('VGGT', 'MASt3R')
            for corruption in ('fog', 'low-light-noise', 'defocus')
        ),
        'split': 'test',
        'schedule_counts': {'scheduled': 400, 'completed': 400, 'missing': 0, 'invalid': 0},
        'downstream_schedule_counts': {'scheduled': 6, 'completed': 6, 'missing': 0},
    }
    fields.update(overrides)
    return GeoReliabGateInput(**fields)


def test_gate_rejects_hand_authored_or_incomplete_p3_evidence():
    assert evaluate_georeliab_gate(_gate_input()).status is GateStatus.PASS
    assert evaluate_georeliab_gate(_gate_input(schedule_counts={})).reason_codes == ('P3_SCHEDULE_COUNTS_INVALID',)
    assert evaluate_georeliab_gate(_gate_input(schedule_counts={'scheduled': 399, 'completed': 399, 'missing': 0, 'invalid': 0})).reason_codes == ('P3_SCHEDULE_COUNTS_INVALID',)
    duplicate = _gate_input().conditions + (_condition('VGGT', 'fog'),)
    assert 'CONDITION_GRID_INVALID' in evaluate_georeliab_gate(_gate_input(conditions=duplicate)).reason_codes
    assert evaluate_georeliab_gate(_gate_input(schedule_counts={'scheduled': 400, 'completed': 400, 'missing': 0, 'invalid': 3})).reason_codes == ('INVALID_PROVENANCE_COUNTS_UNBOUND',)
    assert evaluate_georeliab_gate(_gate_input(downstream_harm=_gate_input().downstream_harm[:2])).reason_codes == ('DOWNSTREAM_EXECUTION_GRID_INVALID',)
    assert evaluate_georeliab_gate(_gate_input(zero_update=_gate_input().zero_update[:1])).reason_codes == ('ZERO_UPDATE_EXECUTION_GRID_INVALID',)
    missing_holm = tuple(
        GeoReliabConditionEvidence(
            **{**item.to_dict(), 'relative_decline_adjusted_p': None, 'relative_decline_holm_rejected': False}
        )
        for item in _gate_input().conditions
    )
    assert 'CONFIDENCE_FAILURE_GATE_NOT_MET' in evaluate_georeliab_gate(_gate_input(conditions=missing_holm)).reason_codes


def test_non_cubic_dtu_obsmask_uses_y_x_z_axes():
    volume = np.zeros((2, 4, 3), dtype=bool)
    volume[1, 3, 2] = True
    points = np.array([[3.0, 1.0, 2.0], [1.0, 3.0, 2.0]])
    mask = dtu_observability_mask_for_points(
        points,
        obs_mask=volume,
        bb=np.array([[0.0, 0.0, 0.0], [3.0, 1.0, 2.0]]),
        res=1.0,
    )
    assert mask.tolist() == [True, False]


def test_model_specific_risk_conversion_is_explicit():
    np.testing.assert_allclose(
        model_risk_from_confidence('VGGT', np.array([1.5])),
        -np.log(np.array([0.5])),
    )
    np.testing.assert_allclose(
        model_risk_from_confidence('MASt3R', np.array([0.5])),
        -np.log(np.array([0.5])),
    )
    with pytest.raises(AuditError, match='unknown model'):
        model_risk_from_confidence('Other', np.array([0.5]))


def _write_bundle_inputs(tmp_path: Path, *, invalid_prediction: bool = False, model='VGGT'):
    manifest = RunManifest(
        run_id='RUN001',
        mode=RunMode.REAL,
        scientific_validity=ScientificValidity.SCIENTIFIC,
        model=model,
        checkpoint_hash='1' * 64,
        dataset='dtu',
        split='test',
        seed=0,
        intervention_version='none',
        corruption_version='clean',
        environment={'python': '3.10.20'},
        rgb_digest='rgb',
        prompt_digest='prompt',
        decoder_digest='decoder',
        provenance=_provenance(),
    )
    geometry = tmp_path / 'geometry.npz'
    confidence = tmp_path / 'confidence.npz'
    valid = tmp_path / 'valid.npz'
    c2w = np.repeat(np.eye(4)[None, :, :], 4, axis=0)
    c2w[:, :3, 3] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    np.savez(
        geometry,
        points_world=np.array([[1.0, 1.0, 1.0], [1.05, 1.05, 1.05], [3.0, 0.0, 0.0]]),
        camera_c2w=c2w,
        intrinsics=np.repeat(np.eye(3)[None, :, :], 4, axis=0),
        pixel_xy=np.array([[0, 0], [1, 1], [2, 2]]),
        view_id=np.array([0, 0, 1]),
    )
    np.savez(confidence, raw_confidence=np.array([1.2, 1.9, 1.1]))
    np.savez(valid, valid_mask=np.array([True, True, True]))
    prediction = PredictionArtifact(
        run_id='RUN001',
        sample_key='dtu/test/scan001/views-0001/clean/0/0',
        geometry_prediction_uri=geometry.as_uri(),
        native_confidence_uri=confidence.as_uri(),
        valid_mask_uri=valid.as_uri(),
        hook_location=None,
        runtime_seconds=1.0,
        peak_memory_mb=10.0,
        invalid_prediction=invalid_prediction,
        payload_digests={
            'geometry_prediction_uri': _sha256(geometry),
            'native_confidence_uri': _sha256(confidence),
            'valid_mask_uri': _sha256(valid),
        },
    )
    manifest_path = tmp_path / 'manifest.json'
    prediction_path = tmp_path / 'prediction.json'
    write_json_artifact(manifest_path, manifest)
    write_json_artifact(prediction_path, prediction)
    return manifest, prediction, manifest_path, prediction_path


def test_invalid_bundle_writes_contract_valid_empty_dense(tmp_path: Path):
    manifest, prediction, manifest_path, prediction_path = _write_bundle_inputs(tmp_path, invalid_prediction=True)
    gt_points = tmp_path / 'gt_points.npy'
    gt_cameras = tmp_path / 'gt_cameras.npy'
    obs_mask = tmp_path / 'obs_mask.npy'
    np.save(gt_points, np.array([[1.0, 1.0, 1.0], [3.0, 0.0, 0.0]]))
    np.save(gt_cameras, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float))
    np.save(obs_mask, np.ones((8, 8, 8), dtype=bool))
    out = tmp_path / 'out'
    assert main([
        'audit-georeliab', '--manifest', str(manifest_path), '--prediction', str(prediction_path),
        '--gt-points', str(gt_points), '--gt-cameras', str(gt_cameras), '--obs-mask', str(obs_mask),
        '--obs-bb', '0,0,0,7,7,7', '--obs-res', '1.0', '--output-dir', str(out),
    ]) == 0
    audit = read_json_artifact(out / 'audit_record.json', AuditRecord)
    validate_artifact_bundle(manifest, prediction, audit)
    dense = np.load(out / 'dense_audit.npz')
    assert dense['voxel_points'].shape == (0, 3)
    assert audit.failure_label is True
    assert audit.accepted is False
    assert audit.downstream_outcome == 0.0


def test_stage_manifest_rejects_hand_authored_aggregate(tmp_path: Path):
    stage = tmp_path / 'stage.json'
    stage.write_text(json.dumps({'conditions': []}), encoding='utf-8')
    assert main(['audit-georeliab', '--stage-evidence', str(stage), '--output', str(tmp_path / 'out.json')]) == 2


def test_official_loader_binds_manifest_digests_and_sample_scene(tmp_path: Path):
    raw = tmp_path / 'scan001.ply'
    raw.write_bytes(b'ply\n')
    frozen = tmp_path / 'frozen_materialization.json'
    frozen.write_text(json.dumps({'scenes': {'scan001': {'ply_sha256': _sha256(raw), 'ply_path': str(raw)}}}), encoding='utf-8')
    split = tmp_path / 'split.json'
    split.write_text(json.dumps({'test': ['scan001']}), encoding='utf-8')
    evidence = load_official_dtu_evidence(
        sample_key='dtu/test/scan001/views-0001/clean/0/0',
        frozen_materialization=frozen,
        split_manifest=split,
    )
    assert evidence['scene'] == 'scan001'
    raw.write_bytes(b'tamper')
    with pytest.raises(AuditError, match='digest'):
        load_official_dtu_evidence(
            sample_key='dtu/test/scan001/views-0001/clean/0/0',
            frozen_materialization=frozen,
            split_manifest=split,
        )
