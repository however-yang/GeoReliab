from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from georeliab_mve.contracts import (
    PredictionArtifact,
    RunManifest,
    RunMode,
    ScientificProvenance,
    ScientificValidity,
    validate_artifact_bundle,
    write_json_artifact,
)
from georeliab_mve.audit import (
    AuditError,
    align_subset_to_full_prediction,
    audit_prediction_arrays,
    build_condition_evidence,
    build_georeliab_evidence,
    compute_zero_update_disagreement_risk,
    coverage_auc,
    dtu_observability_mask_for_points,
    evaluate_zero_update_gain,
    fit_diagnostic_calibration,
    fscore_at_threshold,
    holm_primary_comparisons,
    native_vs_random_harm,
    scene_auroc_branch,
    umeyama_sim3,
)
from georeliab_mve.gates import evaluate_georeliab_gate
from georeliab_mve.metrics import binary_auroc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _proper_transform():
    angle = np.deg2rad(30.0)
    rot = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return 2.5, rot, np.array([0.3, -0.4, 1.2])


def test_umeyama_recovers_known_proper_sim3_and_rejects_reflection_or_degenerate():
    source = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.5, 0.5, 1.0]]
    )
    scale, rot, trans = _proper_transform()
    target = scale * (source @ rot.T) + trans
    estimated = umeyama_sim3(source, target)
    np.testing.assert_allclose(estimated.apply(source), target, atol=1e-10)
    assert np.linalg.det(estimated.rotation) > 0

    reflected = target.copy()
    reflected[:, 0] *= -1
    with pytest.raises(AuditError, match='reflection'):
        umeyama_sim3(source, reflected)
    with pytest.raises(AuditError, match='degenerate'):
        umeyama_sim3(np.zeros((4, 3)), target)


def test_geometry_audit_uses_camera_alignment_not_gt_surface_and_preserves_confidence():
    pred_centers = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    scale, rot, trans = _proper_transform()
    gt_centers = scale * (pred_centers @ rot.T) + trans
    pred_points = np.array(
        [[1.0, 1.0, 1.0], [1.03, 1.03, 1.03], [3.0, 0.0, 0.0]]
    )
    aligned_points = scale * (pred_points @ rot.T) + trans
    raw_conf = np.array([0.2, 0.9, 0.1])
    risk = np.array([4.0, 1.0, 9.0])
    valid_mask = np.array([True, True, True])
    result = audit_prediction_arrays(
        points_world=pred_points,
        pred_camera_centers=pred_centers,
        gt_camera_centers=gt_centers,
        raw_confidence=raw_conf,
        risk=risk,
        valid_mask=valid_mask,
        gt_points=aligned_points + np.array([0.5, 0.0, 0.0]),
        observability_mask=np.array([True, True, True]),
        voxel_size_mm=0.2,
    )
    assert result.summary['model_valid_count'] == 3
    assert result.voxel_points.shape[0] == 2
    assert set(result.provenance_count.tolist()) == {1, 2}
    assert 0.55 in result.raw_confidence
    assert result.labels_1mm.shape == result.failure_label.shape
    assert np.array_equal(result.failure_label, result.gt_error > 2.0)

    shifted_surface = aligned_points + np.array([100.0, 0.0, 0.0])
    shifted = audit_prediction_arrays(
        points_world=pred_points,
        pred_camera_centers=pred_centers,
        gt_camera_centers=gt_centers,
        raw_confidence=raw_conf,
        risk=risk,
        valid_mask=valid_mask,
        gt_points=shifted_surface,
        observability_mask=np.array([True, True, True]),
        voxel_size_mm=0.2,
    )
    np.testing.assert_allclose(shifted.aligned_camera_centers, gt_centers, atol=1e-10)


def test_invalid_output_is_accounted_as_failure_without_silent_drop():
    invalid = audit_prediction_arrays(
        points_world=np.empty((0, 3)),
        pred_camera_centers=np.empty((0, 3)),
        gt_camera_centers=np.empty((0, 3)),
        raw_confidence=np.empty((0,)),
        risk=np.empty((0,)),
        valid_mask=np.empty((0,), dtype=bool),
        gt_points=np.empty((0, 3)),
        observability_mask=np.empty((0,), dtype=bool),
        invalid_prediction=True,
    )
    assert invalid.summary['invalid_prediction'] is True
    assert invalid.summary['fscore_2mm'] == 0.0
    assert invalid.summary['invalid_failure_count'] == 1


def test_dtu_observability_volume_masks_points_in_native_mm_coordinates():
    volume = np.zeros((3, 3, 3), dtype=bool)
    volume[1, 1, 1] = True
    points = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [-1.0, 0.0, 0.0]])
    mask = dtu_observability_mask_for_points(
        points,
        obs_mask=volume,
        bb=np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
        res=1.0,
    )
    assert mask.tolist() == [True, False, False]


def test_directional_rho_ramp_auroc_eligibility_bootstrap_and_holm():
    scenes = [f's{i:02d}' for i in range(20)]
    clean = {scene: {'risk': [1, 2, 3, 4], 'error': [1, 2, 3, 4]} for scene in scenes}
    s1 = {scene: {'risk': [1, 2, 3, 4], 'error': [1, 2, 4, 3]} for scene in scenes}
    s2 = {scene: {'risk': [1, 2, 3, 4], 'error': [1, 3, 2, 4]} for scene in scenes}
    s3 = {
        scene: {'risk': [4, 3, 2, 1], 'error': [1, 2, 3, 4], 'failure': [0, 0, 1, 1]}
        for scene in scenes
    }
    evidence = build_condition_evidence(
        model='VGGT',
        corruption='fog',
        scene_condition_records={'clean': clean, '1': s1, '2': s2, '3': s3},
        corruption_qa=True,
        cross_view_consistent=True,
        gt_geometry_invariant=True,
        n_resamples=10_000,
        seed=5,
    )
    assert evidence.clean_rho == pytest.approx(1.0)
    assert evidence.severity_rhos[-1] == pytest.approx(-1.0)
    assert evidence.relative_decline_ci_lower > 0.5
    assert evidence.failure_auroc_ci_upper < 0.65

    with pytest.raises(AuditError, match='identical frozen scene keys'):
        build_condition_evidence(
            model='VGGT',
            corruption='fog',
            scene_condition_records={'clean': clean, '1': s1, '2': s2, '3': dict(list(s3.items())[:-1])},
            corruption_qa=True,
            cross_view_consistent=True,
            gt_geometry_invariant=True,
        )

    branch = scene_auroc_branch(
        {f's{i:02d}': ([0, 1], [0.1, 0.9]) for i in range(15)},
        n_resamples=10_000,
    )
    assert branch.eligible is False
    assert 'AUROC_ELIGIBLE_SCENES_LT_16' in branch.reason_codes

    adjusted = holm_primary_comparisons({'a': 0.01, 'b': 0.03, 'c': 0.2})
    assert adjusted['a'].adjusted_p == pytest.approx(0.03)
    assert adjusted['b'].adjusted_p == pytest.approx(0.06)


def test_diagnostic_calibration_is_calibration_only():
    mapping = fit_diagnostic_calibration(
        split='calibration',
        risks=[0.0, 1.0, 2.0, 3.0],
        failures=[0, 0, 1, 1],
    )
    assert mapping.apply([0.5, 2.5])[0] < mapping.apply([0.5, 2.5])[1]
    with pytest.raises(AuditError, match='calibration-clean'):
        fit_diagnostic_calibration(split='test', risks=[0, 1], failures=[0, 1])


def test_downstream_harm_and_zero_update_use_scene_pairs_and_predicted_alignment():
    errors = np.array([1.0, 3.0, 1.0, 3.0])
    risk = np.array([0.1, 0.2, 0.8, 0.9])
    pred_points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [10.0, 10.0, 0.0]])
    gt_points = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [5.0, 5.0, 0.0], [6.0, 6.0, 0.0]])
    gt_to_pred = np.array([1.0, 3.0, 3.0, 3.0])
    assert fscore_at_threshold(errors, np.ones(4, dtype=bool), gt_to_selected_errors=gt_to_pred) == pytest.approx(1 / 3)
    auc = coverage_auc(errors, risk, [0.5, 1.0], pred_points=pred_points, gt_points=gt_points)
    assert 0.0 <= auc <= 1.0
    harm = native_vs_random_harm(
        {'s1': (errors, risk, pred_points, gt_points), 's2': (errors, risk, pred_points, gt_points)},
        coverages=(0.5, 1.0),
        n_random_masks=100,
        seed=2,
        n_resamples=10_000,
    )
    assert harm.n_scenes == 2
    assert harm.n_random_masks == 100
    mask_a = np.array([True, True, False, False])
    mask_b = np.array([False, False, True, True])
    from georeliab_mve.audit import gt_to_selected_errors

    assert not np.array_equal(
        gt_to_selected_errors(pred_points, gt_points, mask_a),
        gt_to_selected_errors(pred_points, gt_points, mask_b),
    )

    full_cameras = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 1]], dtype=float
    )
    scale, rot, trans = _proper_transform()
    view_ids = [0, 1, 2, 3, 5, 6]
    subset_cameras = (full_cameras[view_ids] - trans) @ rot / scale
    full_points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    subset_points = (full_points - trans) @ rot / scale
    aligned = align_subset_to_full_prediction(
        subset_points, subset_cameras, full_cameras, subset_view_ids=view_ids
    )
    np.testing.assert_allclose(aligned, full_points, atol=1e-10)
    gain = evaluate_zero_update_gain(
        {
            's1': {
                'native_risk': [0.1, 0.2, 0.8, 0.9],
                'zero_update_risk': [0.1, 0.8, 0.2, 0.9],
                'failure': [0, 1, 0, 1],
            },
            's2': {
                'native_risk': [0.1, 0.2, 0.8, 0.9],
                'zero_update_risk': [0.1, 0.8, 0.2, 0.9],
                'failure': [0, 1, 0, 1],
            },
        },
        n_resamples=10_000,
    )
    assert gain.auroc_gain > 0

    subset_specs = []
    for view_ids_for_subset, offset in zip(
        ([1, 2, 3, 5, 6, 7], [0, 2, 3, 4, 6, 7], [0, 1, 3, 4, 5, 7], [0, 1, 2, 4, 5, 6]),
        (0.1, 0.5, 0.2, 0.4),
        strict=True,
    ):
        subset_cam = (full_cameras[view_ids_for_subset] - trans) @ rot / scale
        shifted_subset_points = (full_points + np.array([[0.0, 0.0, offset], [0.0, 0.0, offset * 2]]) - trans) @ rot / scale
        subset_specs.append(
            {
                'points': shifted_subset_points,
                'camera_centers': subset_cam,
                'view_ids': view_ids_for_subset,
            }
        )
    disagreement = compute_zero_update_disagreement_risk(
        full_points=full_points,
        full_camera_centers=full_cameras,
        subset_predictions=subset_specs,
    )
    assert disagreement.shape == (2,)
    assert disagreement[1] > disagreement[0]
    bad_specs = [dict(item) for item in subset_specs]
    bad_specs[0]['view_ids'] = [0, 1, 2, 3, 4, 5]
    with pytest.raises(AuditError, match='frozen omitted views'):
        compute_zero_update_disagreement_risk(
            full_points=full_points,
            full_camera_centers=full_cameras,
            subset_predictions=bad_specs,
        )


def test_georeliab_evidence_rejects_incomplete_p3_before_p4():
    payload = build_georeliab_evidence(
        condition_evidence=[],
        downstream_harm=[],
        zero_update=[],
        required_models_ready=('VGGT', 'MASt3R'),
        required_datasets_ready=True,
        tartanair_native_fog_sanity=True,
        run_mode='real',
        split='test',
        statistics={'primary_p_values': {'fog': 0.01, 'defocus': 0.04}},
    )
    gate = evaluate_georeliab_gate(payload.to_gate_input())
    assert gate.status.value == 'FAIL'
    assert payload.p5_skip_reason is None
    assert payload.statistics['holm_primary']['fog']['adjusted_p'] == pytest.approx(0.02)
    assert gate.reason_codes == ('P3_SCHEDULE_COUNTS_INVALID',)


def test_audit_cli_rejects_unbound_aggregate_evidence(tmp_path: Path):
    evidence = {
        'run_mode': 'smoke',
        'split': 'test',
        'evidence_schema_version': '1.1',
        'conditions': [],
    }
    source = tmp_path / 'evidence.json'
    source.write_text(json.dumps(evidence), encoding='utf-8')
    from georeliab_mve.cli import main

    assert main(['audit-georeliab', '--input', str(source), '--output', str(tmp_path / 'out.json')]) == 2
    evidence['run_mode'] = 'real'
    source.write_text(json.dumps(evidence), encoding='utf-8')
    assert main(['audit-georeliab', '--input', str(source), '--output', str(tmp_path / 'out.json')]) == 2
    assert not (tmp_path / 'out.json').exists()


def test_audit_cli_bundle_mode_writes_dense_npz_and_validated_audit(tmp_path: Path):
    from georeliab_mve.cli import main

    sample_key = 'dtu/test/scan001/views-0001/clean/0/0'
    provenance = ScientificProvenance(
        project_commit='a' * 40,
        project_tree='b' * 40,
        model_source_commit='c' * 40,
        environment_lock_sha256='d' * 64,
        corruption_manifest_sha256='e' * 64,
        split_view_manifest_sha256='f' * 64,
    )
    manifest = RunManifest(
        run_id='RUN001',
        mode=RunMode.REAL,
        scientific_validity=ScientificValidity.SCIENTIFIC,
        model='VGGT',
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
        provenance=provenance,
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
    np.savez(confidence, raw_confidence=np.array([0.2, 0.9, 0.1]))
    np.savez(valid, valid_mask=np.array([True, True, True]))
    prediction = PredictionArtifact(
        run_id='RUN001',
        sample_key=sample_key,
        geometry_prediction_uri=geometry.as_uri(),
        native_confidence_uri=confidence.as_uri(),
        valid_mask_uri=valid.as_uri(),
        hook_location=None,
        runtime_seconds=1.0,
        peak_memory_mb=10.0,
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
    gt_points = tmp_path / 'gt_points.npy'
    gt_cameras = tmp_path / 'gt_cameras.npy'
    obs_mask = tmp_path / 'obs_mask.npy'
    np.save(gt_points, np.array([[1.0, 1.0, 1.0], [3.0, 0.0, 0.0], [20.0, 20.0, 20.0]]))
    np.save(gt_cameras, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float))
    volume = np.ones((8, 8, 8), dtype=bool)
    np.save(obs_mask, volume)
    out_dir = tmp_path / 'bundle_out'
    assert main(
        [
            'audit-georeliab',
            '--manifest', str(manifest_path),
            '--prediction', str(prediction_path),
            '--gt-points', str(gt_points),
            '--gt-cameras', str(gt_cameras),
            '--obs-mask', str(obs_mask),
            '--obs-bb', '0,0,0,7,7,7',
            '--obs-res', '1.0',
            '--output-dir', str(out_dir),
        ]
    ) == 0
    assert (out_dir / 'dense_audit.npz').exists()
    audit = json.loads((out_dir / 'audit_record.json').read_text(encoding='utf-8'))
    assert audit['metadata']['dense_audit_uri'].startswith('file:')
    from georeliab_mve.contracts import read_json_artifact, AuditRecord

    validate_artifact_bundle(
        manifest,
        prediction,
        read_json_artifact(out_dir / 'audit_record.json', AuditRecord),
    )
