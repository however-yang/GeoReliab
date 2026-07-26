from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zlib

import numpy as np
import pytest
import scipy.io

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
from georeliab_mve.preparation import TEST_SCENES


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(model: str = 'VGGT', *, invalid=False) -> tuple[RunManifest, PredictionArtifact, Path, Path, Path]:
    raise RuntimeError('test helper must be monkeypatched per tmp_path')


def _provenance(*, split_sha: str = 'f' * 64) -> ScientificProvenance:
    return ScientificProvenance(
        project_commit='a' * 40,
        project_tree='b' * 40,
        model_source_commit='c' * 40,
        environment_lock_sha256='d' * 64,
        corruption_manifest_sha256='e' * 64,
        split_view_manifest_sha256=split_sha,
        dust3r_source_commit='1' * 40,
        croco_source_commit='2' * 40,
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
    assert evaluate_georeliab_gate(_gate_input(schedule_counts={'scheduled': 400, 'completed': 400, 'missing': 0, 'invalid': 3})).reason_codes == ('INVALID_OUTPUT_IN_VERIFIED_CONDITION',)
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


def _write_stage_bundle_grid(tmp_path: Path) -> tuple[Path, list[dict]]:
    root = tmp_path / 'stage'
    root.mkdir(parents=True)
    split_manifest = tmp_path / 'split_view_manifest.json'
    split_manifest.write_text(json.dumps({'schema_version': 'dtu-preparation-v1', 'splits': {'test': list(TEST_SCENES)}}), encoding='utf-8')
    split_sha = _sha256(split_manifest)
    corruption_qa = tmp_path / 'corruption_calibration_qa.json'
    render_lock = tmp_path / 'test_render_lock.json'
    tartan_sanity = tmp_path / 'tartanair_native_fog_sanity.json'
    stage_scenes = [f'scan{scene}' for scene in TEST_SCENES]
    parameter_sha = '9' * 64
    materialization_sha = '8' * 64
    corruption_qa.write_text(json.dumps({'schema_version': 'calibration-qa-v1', 'passed': True, 'checks': {'fog': True, 'synthetic_fog': True, 'low_light': True, 'defocus': True, 'gt': True, 'cross_view': True}, 'parameter_manifest_sha256': parameter_sha, 'split_view_manifest_sha256': split_sha, 'materialization_sha256': materialization_sha}), encoding='utf-8')
    qa_sha = _sha256(corruption_qa)
    render_lock.write_text(json.dumps({'schema_version': 'test-render-lock-v1', 'stage': 'test', 'split': 'test', 'split_view_manifest_sha256': split_sha, 'materialization_sha256': materialization_sha, 'parameter_manifest_sha256': parameter_sha, 'calibration_qa_sha256': qa_sha, 'prepared_input_sha256': '7' * 64}), encoding='utf-8')
    tartan_sanity.write_text(json.dumps({'reason_code': 'TARTANAIR_NATIVE_FOG_SANITY', 'passed': True, 'negative_frames': 80, 'evaluated_frames': 100, 'correlations': ([-0.1] * 80 + [0.0] * 20), 'prepared_input_sha256': '6' * 64, 'calibration_qa_sha256': qa_sha}), encoding='utf-8')
    entries = []
    risk = np.array([0.0, 1.0, 2.0, 3.0])
    errors = {
        ('clean', '0'): np.array([0.0, 1.0, 2.0, 3.0]),
        ('fog', '1'): np.array([0.0, 1.0, 3.0, 2.0]),
        ('fog', '2'): np.array([3.0, 2.0, 1.0, 0.0]),
        ('fog', '3'): np.array([3.0, 2.0, 1.0, 0.0]),
        ('low-light-noise', '1'): np.array([0.0, 1.0, 3.0, 2.0]),
        ('low-light-noise', '2'): np.array([3.0, 2.0, 1.0, 0.0]),
        ('low-light-noise', '3'): np.array([3.0, 2.0, 1.0, 0.0]),
        ('defocus', '1'): np.array([0.0, 1.0, 3.0, 2.0]),
        ('defocus', '2'): np.array([3.0, 2.0, 1.0, 0.0]),
        ('defocus', '3'): np.array([3.0, 2.0, 1.0, 0.0]),
    }
    geometry = root / 'geometry.npz'
    confidence = root / 'confidence.npz'
    valid = root / 'valid.npz'
    c2w = np.repeat(np.eye(4)[None, :, :], 8, axis=0)
    c2w[:, :3, 3] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=float)
    full_points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
    np.savez(geometry, points_world=full_points, camera_c2w=c2w, intrinsics=np.repeat(np.eye(3)[None, :, :], 8, axis=0), pixel_xy=np.array([[0, 0], [1, 0], [2, 0], [3, 0]]), view_id=np.array([0, 1, 2, 3]))
    np.savez(confidence, raw_confidence=np.array([1.1, 1.2, 1.3, 1.4]))
    np.savez(valid, valid_mask=np.ones(4, dtype=bool))
    payload_digests = {'geometry_prediction_uri': _sha256(geometry), 'native_confidence_uri': _sha256(confidence), 'valid_mask_uri': _sha256(valid)}
    for model in ('VGGT', 'MASt3R'):
        for scene in stage_scenes:
            for condition, severity in [('clean', '0'), ('fog', '1'), ('fog', '2'), ('fog', '3'), ('low-light-noise', '1'), ('low-light-noise', '2'), ('low-light-noise', '3'), ('defocus', '1'), ('defocus', '2'), ('defocus', '3')]:
                sample_key = f'dtu/test/{scene}/views-0001/{condition}/{severity}/0'
                stem = f'{model}_{scene}_{condition}_{severity}'.replace('-', '_')
                manifest = RunManifest(run_id=f'RUN{len(entries):04d}', mode=RunMode.REAL, scientific_validity=ScientificValidity.SCIENTIFIC, model=model, checkpoint_hash='1' * 64, dataset='dtu', split='test', seed=0, intervention_version='none', corruption_version='clean', environment={'python': '3.10.20'}, rgb_digest='rgb', prompt_digest='prompt', decoder_digest='decoder', provenance=_provenance(split_sha=split_sha))
                prediction = PredictionArtifact(run_id=manifest.run_id, sample_key=sample_key, geometry_prediction_uri=geometry.as_uri(), native_confidence_uri=confidence.as_uri(), valid_mask_uri=valid.as_uri(), hook_location=None, runtime_seconds=1.0, peak_memory_mb=10.0, payload_digests=payload_digests)
                dense_path = root / f'{stem}_dense.npz'
                gt_path = root / f'{stem}_gt.npz'
                np.savez(gt_path, gt_points=np.array([[0, 0, 0], [3, 0, 0]], dtype=float))
                gt_error = errors[(condition, severity)]
                np.savez(dense_path, voxel_points=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float), raw_confidence=np.array([1.1, 1.2, 1.3, 1.4]), risk=risk, gt_error=gt_error, failure_label=gt_error > 2.0, provenance_count=np.ones(4, dtype=np.int64))
                audit = AuditRecord(run_id=manifest.run_id, sample_key=sample_key, gt_error=float(np.median(gt_error)), failure_label=bool(np.any(gt_error > 2.0)), selection_score=float(np.median(risk)), coverage=1.0, accepted=True, downstream_outcome=0.5, metadata={'dense_audit_uri': dense_path.resolve().as_uri(), 'dense_audit_sha256': _sha256(dense_path), 'gt_points_uri': gt_path.resolve().as_uri(), 'gt_points_sha256': _sha256(gt_path)})
                manifest_path = root / f'{stem}_manifest.json'
                prediction_path = root / f'{stem}_prediction.json'
                audit_path = root / f'{stem}_audit.json'
                summary_path = root / f'{stem}_summary.json'
                write_json_artifact(manifest_path, manifest)
                write_json_artifact(prediction_path, prediction)
                write_json_artifact(audit_path, audit)
                summary_path.write_text(json.dumps({'schema_version': 'validated-scene-summary-v1', 'producer': 'audit-georeliab', 'sample_key': sample_key, 'audit_sha256': _sha256(audit_path), 'corruption_severity_monotonic': True, 'cross_view_consistent': True, 'gt_geometry_invariant': True}), encoding='utf-8')
                entries.append({'manifest_path': str(manifest_path), 'manifest_sha256': _sha256(manifest_path), 'prediction_path': str(prediction_path), 'prediction_sha256': _sha256(prediction_path), 'audit_path': str(audit_path), 'audit_sha256': _sha256(audit_path), 'scene_summary_path': str(summary_path), 'scene_summary_sha256': _sha256(summary_path)})
    stage_payload = {
        'schema_version': 'stage-evidence-v1',
        'split_view_manifest_path': str(split_manifest),
        'split_view_manifest_sha256': split_sha,
        'corruption_calibration_qa_path': str(corruption_qa),
        'corruption_calibration_qa_sha256': _sha256(corruption_qa),
        'test_render_lock_path': str(render_lock),
        'test_render_lock_sha256': _sha256(render_lock),
        'tartanair_native_fog_sanity_path': str(tartan_sanity),
        'tartanair_native_fog_sanity_sha256': _sha256(tartan_sanity),
        'parameter_manifest_sha256': parameter_sha,
        'materialization_sha256': materialization_sha,
        'bundle_index': entries,
        'downstream_index': [],
        'zero_update_index': [],
    }
    severity2_keys = {
        (model, f'{corruption}-s2'): [f'dtu/test/{scene}/views-0001/{corruption}/2/0' for scene in stage_scenes]
        for model in ('VGGT', 'MASt3R') for corruption in ('fog', 'low-light-noise', 'defocus')
    }
    for row_type in ('downstream', 'zero_update'):
        for model in ('VGGT', 'MASt3R'):
            for corruption in ('fog', 'low-light-noise', 'defocus'):
                condition = f'{corruption}-s2'
                if row_type == 'downstream':
                    gt_inputs = {}
                    for scene in stage_scenes:
                        source_stem = f'{model}_{scene}_{corruption}_2'.replace('-', '_')
                        path = root / f'{source_stem}_gt.npz'
                        gt_inputs[scene] = {'input_path': str(path), 'input_sha256': _sha256(path)}
                    payload = {'schema_version': 'downstream-harm-v1', 'model': model, 'condition': condition, 'coverage': [0.9, 0.7, 0.5, 0.3], 'random_mask_count': 100, 'n_resamples': 10000, 'gt_inputs': gt_inputs, 'source_sample_keys': severity2_keys[(model, condition)]}
                else:
                    subset_artifacts = {}
                    for scene in stage_scenes:
                        subset_artifacts[scene] = []
                        for index in range(4):
                            artifact = root / f'subset_{model}_{condition}_{scene}_{index}.npz'.replace('-', '_')
                            view_ids = np.array([view for view in range(8) if view not in ([0, 4], [1, 5], [2, 6], [3, 7])[index]], dtype=np.int64)
                            subset_points = full_points.copy()
                            if model == 'VGGT' and corruption == 'fog':
                                subset_points[0, 2] += 10.0
                            else:
                                subset_points[:, 2] += np.array([0.0, 1.0, 2.0, 3.0])
                            parent_sample = f'dtu/test/{scene}/views-0001/{corruption}/2/0'
                            np.savez(artifact, points=subset_points, camera_centers=c2w[view_ids, :3, 3], view_ids=view_ids, parent_model=np.asarray(model), parent_sample_key=np.asarray(parent_sample), parent_project_commit=np.asarray('a' * 40))
                            subset_artifacts[scene].append({'artifact_path': str(artifact), 'artifact_sha256': _sha256(artifact)})
                    payload = {'schema_version': 'zero-update-v1', 'model': model, 'condition': condition, 'n_resamples': 10000, 'omitted_view_pairs': [[0, 4], [1, 5], [2, 6], [3, 7]], 'subset_artifacts': subset_artifacts, 'source_sample_keys': severity2_keys[(model, condition)]}
                path = root / f'{row_type}_{model}_{condition}.json'.replace('-', '_')
                path.write_text(json.dumps(payload), encoding='utf-8')
                stage_payload[f'{row_type}_index'].append({'evidence_path': str(path), 'evidence_sha256': _sha256(path)})
    stage_path = tmp_path / 'stage_evidence.json'
    stage_path.write_text(json.dumps(stage_payload), encoding='utf-8')
    return stage_path, entries


def test_stage_manifest_derives_gate_from_400_validated_bundles(tmp_path: Path):
    stage, _ = _write_stage_bundle_grid(tmp_path)
    output = tmp_path / 'out.json'
    assert main(['audit-georeliab', '--stage-evidence', str(stage), '--output', str(output)]) == 0
    written = json.loads(output.read_text(encoding='utf-8'))
    assert written['georeliab_gate']['status'] == 'PASS'
    assert written['gate_input']['schedule_counts'] == {'scheduled': 400, 'completed': 400, 'missing': 0, 'invalid': 0}
    assert len(written['gate_input']['conditions']) == 6


def test_stage_manifest_rejects_missing_duplicate_crosslink_and_aggregate_bypass(tmp_path: Path):
    stage, entries = _write_stage_bundle_grid(tmp_path)
    payload = json.loads(stage.read_text(encoding='utf-8'))
    missing = tmp_path / 'missing.json'
    missing.write_text(json.dumps({**payload, 'bundle_index': entries[:-1]}), encoding='utf-8')
    assert main(['audit-georeliab', '--stage-evidence', str(missing), '--output', str(tmp_path / 'missing_out.json')]) == 2
    duplicate = tmp_path / 'duplicate.json'
    duplicate.write_text(json.dumps({**payload, 'bundle_index': entries[:-1] + [entries[0]]}), encoding='utf-8')
    assert main(['audit-georeliab', '--stage-evidence', str(duplicate), '--output', str(tmp_path / 'dup_out.json')]) == 2
    first_summary = Path(entries[0]['scene_summary_path'])
    first_summary.write_text(json.dumps({'sample_key': 'dtu/test/other/views-0001/clean/0/0', 'corruption_severity_monotonic': True, 'cross_view_consistent': True, 'gt_geometry_invariant': True}), encoding='utf-8')
    cross = tmp_path / 'cross.json'
    cross_entries = [dict(item) for item in entries]
    cross_entries[0]['scene_summary_sha256'] = _sha256(first_summary)
    cross.write_text(json.dumps({**payload, 'bundle_index': cross_entries}), encoding='utf-8')
    assert main(['audit-georeliab', '--stage-evidence', str(cross), '--output', str(tmp_path / 'cross_out.json')]) == 2
    stage2, entries2 = _write_stage_bundle_grid(tmp_path / 'aggregate')
    payload2 = json.loads(stage2.read_text(encoding='utf-8'))
    injected = tmp_path / 'injected.json'
    injected.write_text(json.dumps({**payload2, 'conditions': [_condition('VGGT', 'fog').to_dict()]}), encoding='utf-8')
    assert main(['audit-georeliab', '--stage-evidence', str(injected), '--output', str(tmp_path / 'inject_out.json')]) == 2


def _asset(path: Path, member: str) -> dict:
    data = path.read_bytes()
    return {
        'path': str(path),
        'member': member,
        'raw_sha256': hashlib.sha256(data).hexdigest(),
        'uncompressed_size': len(data),
        'crc32': f'{zlib.crc32(data) & 0xFFFFFFFF:08x}',
    }


def _write_binary_ply(path: Path, xyz: np.ndarray) -> None:
    rows = np.zeros(
        len(xyz),
        dtype=np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ]),
    )
    rows['x'], rows['y'], rows['z'] = xyz.T
    header = (
        'ply\r\nformat binary_little_endian 1.0\r\n'
        f'element vertex {len(rows)}\r\n'
        'property float x\r\nproperty float y\r\nproperty float z\r\n'
        'property float nx\r\nproperty float ny\r\nproperty float nz\r\n'
        'property uchar red\r\nproperty uchar green\r\nproperty uchar blue\r\n'
        'end_header\r\n'
    ).encode('ascii')
    path.write_bytes(header + rows.tobytes())


def _write_realistic_dtu_materialization(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    root = tmp_path / 'data'
    materialized = root / 'materialized'
    manifests = root / 'manifests'
    materialized.mkdir(parents=True)
    manifests.mkdir(parents=True)
    inventory = manifests / 'inventory.json'
    inventory.write_text('{"ok": true}', encoding='utf-8')
    split = {
        'schema_version': 'dtu-preparation-v1',
        'splits': {'dev': list(range(1, 11)), 'calibration': list(range(11, 21)), 'reference-token': list(range(21, 26)), 'test': list(range(26, 46))},
        'views': {str(scene): list(range(1, 9)) for scene in range(1, 46)},
    }
    split_path = manifests / 'split_view_manifest.json'
    split_path.write_text(json.dumps(split), encoding='utf-8')
    ply = materialized / 'scan026.ply'
    _write_binary_ply(ply, np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32))
    mask = materialized / 'ObsMask26_10.mat'
    scipy.io.savemat(mask, {'ObsMask': np.ones((2, 3, 4), dtype=bool), 'BB': np.array([[0, 0, 0], [1, 2, 3]], dtype=float), 'Res': np.array([[1.0]])})
    rgb = materialized / 'rgb.png'
    rgb.write_bytes(b'rgb')
    tartan = materialized / 'tartan.bin'
    tartan.write_bytes(b'tartan')
    cameras = {}
    for view in range(1, 9):
        path = materialized / f'pos_{view:03d}.txt'
        path.write_text(f'1 0 0 {-view}\n0 1 0 0\n0 0 1 0\n', encoding='ascii')
        cameras[view] = path
    rows = []
    for split_name, scenes in split['splits'].items():
        for scene in scenes:
            rows.append({
                'scene_id': scene,
                'split': split_name,
                'points': _asset(ply, f'Points/stl/stl{scene:03d}_total.ply'),
                'mask': _asset(mask, f'MVS Data/ObsMask/ObsMask{scene}_10.mat'),
                'rgb': {str(view): _asset(rgb, f'Rectified/scan{scene}/rect_{view:03d}_3_r5000.png') for view in range(1, 9)},
                'cameras': {str(view): _asset(cameras[view], f'MVS Data/Calibration/cal18/pos_{view:03d}.txt') for view in range(1, 9)},
            })
    commit = 'a' * 40
    materialization = {
        'schema_version': 'frozen-materialization-v1',
        'split_view_manifest_path': str(split_path),
        'split_view_manifest_sha256': _sha256(split_path),
        'dtu_inventory_provenance_path': str(inventory),
        'dtu_inventory_provenance_sha256': _sha256(inventory),
        'archives': {'tartanair-image': {'url': f'https://hf/resolve/{commit}/image.zip'}, 'tartanair-depth': {'url': f'https://hf/resolve/{commit}/depth.zip'}},
        'dtu': rows,
        'tartanair': {'source_commit': commit, 'pairs': [
            {'frame_id': f'{i:06d}', 'rgb': _asset(tartan, f'GreatMarsh/Data_easy/P000/image_lcam_front/{i:06d}_lcam_front.png'), 'depth': _asset(tartan, f'GreatMarsh/Data_easy/P000/depth_lcam_front/{i:06d}_lcam_front_depth.png')}
            for i in range(100)
        ]},
    }
    frozen = manifests / 'frozen_materialization.json'
    frozen.write_text(json.dumps(materialization), encoding='utf-8')
    return frozen, split_path, {'ply': ply, 'mask': mask, 'camera1': cameras[1]}


def test_official_loader_binds_real_materialization_and_exact_sources(tmp_path: Path):
    frozen, split, paths = _write_realistic_dtu_materialization(tmp_path)
    evidence = load_official_dtu_evidence(
        sample_key='dtu/test/scan026/views-0001/clean/0/0',
        frozen_materialization=frozen,
        split_manifest=split,
    )
    assert evidence['scene_id'] == 26
    assert evidence['gt_points'].shape == (3, 3)
    assert evidence['gt_camera_centers'].shape == (8, 3)
    assert evidence['gt_camera_centers'][0].tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert evidence['obs_mask'].shape == (2, 3, 4)
    paths['ply'].write_text('tamper', encoding='ascii')
    with pytest.raises(AuditError, match='materialization verification failed'):
        load_official_dtu_evidence(sample_key='dtu/test/scan026/views-0001/clean/0/0', frozen_materialization=frozen, split_manifest=split)


def test_official_loader_rejects_mask_camera_and_split_view_mismatch(tmp_path: Path):
    frozen, split, paths = _write_realistic_dtu_materialization(tmp_path)
    paths['mask'].write_bytes(b'tamper')
    with pytest.raises(AuditError, match='materialization verification failed'):
        load_official_dtu_evidence(sample_key='dtu/test/scan026/views-0001/clean/0/0', frozen_materialization=frozen, split_manifest=split)
    frozen, split, paths = _write_realistic_dtu_materialization(tmp_path / 'camera')
    paths['camera1'].write_text('1 0 0 -9\n0 1 0 0\n0 0 1 0\n', encoding='ascii')
    with pytest.raises(AuditError, match='materialization verification failed'):
        load_official_dtu_evidence(sample_key='dtu/test/scan026/views-0001/clean/0/0', frozen_materialization=frozen, split_manifest=split)
    frozen, split, _ = _write_realistic_dtu_materialization(tmp_path / 'split')
    with pytest.raises(AuditError, match='scene/split'):
        load_official_dtu_evidence(sample_key='dtu/dev/scan026/views-0001/clean/0/0', frozen_materialization=frozen, split_manifest=split)
