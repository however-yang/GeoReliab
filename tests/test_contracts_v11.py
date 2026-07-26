from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from georeliab_mve.contracts import (
    AuditRecord,
    ContractError,
    PredictionArtifact,
    RunManifest,
    RunMode,
    ScientificValidity,
    read_json_artifact,
    validate_artifact_bundle,
    write_json_artifact,
)


SHA256 = 'a' * 64
GIT_HASH = 'b' * 40


def provenance(*, mast3r=False):
    from georeliab_mve.contracts import ScientificProvenance

    fields = dict(
        project_commit=GIT_HASH,
        project_tree='c' * 40,
        model_source_commit='d' * 40,
        environment_lock_sha256=SHA256,
        corruption_manifest_sha256=SHA256,
        split_view_manifest_sha256=SHA256,
    )
    if mast3r:
        fields.update(dust3r_source_commit='e' * 40, croco_source_commit='f' * 40)
    return ScientificProvenance(**fields)


def fixture_manifest(*, schema_version='1.1'):
    return RunManifest(
        run_id='DRYRUN-001',
        mode=RunMode.FIXTURE,
        scientific_validity=ScientificValidity.NON_SCIENTIFIC_FIXTURE,
        model='fixture-model',
        checkpoint_hash='not-real',
        dataset='fixture',
        split='test',
        seed=0,
        intervention_version='geometry-v1',
        corruption_version='georeliab-v1',
        environment={'python': '3.14.3'},
        rgb_digest='rgb-fixed',
        prompt_digest='prompt-fixed',
        decoder_digest='decoder-fixed',
        schema_version=schema_version,
    )


def test_v1_0_reader_upgrades_to_v1_1_writer(tmp_path):
    payload = fixture_manifest().to_dict()
    payload['schema_version'] = '1.0'
    source = tmp_path / 'v1.0.json'
    source.write_text(json.dumps(payload), encoding='utf-8')

    loaded = read_json_artifact(source, RunManifest)
    assert loaded.schema_version == '1.1'

    upgraded = tmp_path / 'v1.1.json'
    write_json_artifact(upgraded, loaded)
    assert json.loads(upgraded.read_text(encoding='utf-8'))['schema_version'] == '1.1'


@pytest.mark.parametrize('artifact_type', (PredictionArtifact, AuditRecord))
def test_v1_0_non_manifest_readers_upgrade_to_v1_1(artifact_type, tmp_path):
    key = 'fixture/test/s1/v1/fog/3/0'
    if artifact_type is PredictionArtifact:
        artifact = PredictionArtifact(
            run_id='DRYRUN-001',
            sample_key=key,
            geometry_prediction_uri='fixture://geometry',
            native_confidence_uri='fixture://confidence',
            valid_mask_uri='fixture://mask',
            hook_location=None,
            runtime_seconds=0.0,
            peak_memory_mb=0.0,
        )
    else:
        artifact = AuditRecord(
            run_id='DRYRUN-001',
            sample_key=key,
            gt_error=0.0,
            failure_label=False,
            selection_score=0.0,
            coverage=1.0,
            accepted=True,
            downstream_outcome=0.0,
        )
    payload = artifact.to_dict()
    payload['schema_version'] = '1.0'
    source = tmp_path / f'{artifact_type.__name__}.json'
    source.write_text(json.dumps(payload), encoding='utf-8')
    assert read_json_artifact(source, artifact_type).schema_version == '1.1'


@pytest.mark.parametrize('mode', (RunMode.REAL, RunMode.SMOKE))
def test_real_and_smoke_require_checkpoint_and_full_provenance(mode):
    payload = fixture_manifest().to_dict()
    payload.update(
        mode=mode.value,
        scientific_validity=(
            'SCIENTIFIC' if mode is RunMode.REAL else 'NON_SCIENTIFIC_SMOKE'
        ),
        checkpoint_hash=SHA256,
    )
    with pytest.raises(ContractError, match='provenance'):
        RunManifest.from_dict(payload)

    payload['provenance'] = provenance().to_dict()
    assert RunManifest.from_dict(payload).mode is mode


def test_mast3r_provenance_requires_dust3r_and_croco():
    payload = fixture_manifest().to_dict()
    payload.update(
        mode='real',
        scientific_validity='SCIENTIFIC',
        model='MASt3R',
        checkpoint_hash=SHA256,
        provenance=provenance().to_dict(),
    )
    with pytest.raises(ContractError, match='MASt3R'):
        RunManifest.from_dict(payload)


def _bundle_records(tmp_path, *, digest=''):
    geometry = tmp_path / 'geometry.npz'
    confidence = tmp_path / 'confidence.npz'
    valid_mask = tmp_path / 'valid_mask.npz'
    dense_audit = tmp_path / 'dense_audit.npz'
    np.savez(
        geometry,
        points_world=np.ones((2, 3)),
        camera_c2w=np.eye(4)[None, ...],
        intrinsics=np.eye(3)[None, ...],
        pixel_xy=np.array([[1.0, 2.0], [3.0, 4.0]]),
        view_id=np.array([0, 0]),
    )
    np.savez(confidence, raw_confidence=np.array([0.2, 0.4]))
    np.savez(valid_mask, valid_mask=np.array([True, True]))
    np.savez(
        dense_audit,
        voxel_points=np.ones((2, 3)),
        raw_confidence=np.array([0.2, 0.4]),
        risk=np.array([1.0, 2.0]),
        gt_error=np.array([0.001, 0.003]),
        failure_label=np.array([False, True]),
        provenance_count=np.array([1, 1]),
    )
    manifest = fixture_manifest()
    key = 'fixture/test/s1/v1/fog/3/0'
    prediction = PredictionArtifact(
        run_id=manifest.run_id,
        sample_key=key,
        geometry_prediction_uri=geometry.as_uri(),
        native_confidence_uri=confidence.as_uri(),
        valid_mask_uri=valid_mask.as_uri(),
        hook_location=None,
        runtime_seconds=0.0,
        peak_memory_mb=0.0,
        payload_digests={
            'geometry_prediction_uri': digest,
            'native_confidence_uri': digest,
            'valid_mask_uri': digest,
        },
    )
    audit = AuditRecord(
        run_id=manifest.run_id,
        sample_key=key,
        gt_error=0.003,
        failure_label=True,
        selection_score=0.0,
        coverage=1.0,
        accepted=False,
        downstream_outcome=0.0,
        metadata={'dense_audit_uri': dense_audit.as_uri()},
    )
    return manifest, prediction, audit


def test_bundle_validator_rejects_payload_and_linkage_failures(tmp_path):
    manifest, prediction, audit = _bundle_records(tmp_path)
    geometry = prediction.geometry_prediction_uri.removeprefix('file:///')
    np.savez(geometry, points_world=np.ones((2, 3)))
    with pytest.raises(ContractError, match='camera_c2w'):
        validate_artifact_bundle(manifest, prediction, audit)

    manifest, prediction, audit = _bundle_records(tmp_path)
    mask = prediction.valid_mask_uri.removeprefix('file:///')
    np.savez(mask, valid_mask=np.array([True, False]))
    with pytest.raises(ContractError, match='filtering drift'):
        validate_artifact_bundle(manifest, prediction, audit)

    manifest, prediction, audit = _bundle_records(tmp_path, digest=SHA256)
    with pytest.raises(ContractError, match='digest mismatch'):
        validate_artifact_bundle(manifest, prediction, audit)

    audit = AuditRecord(
        **{**audit.to_dict(), 'sample_key': 'fixture/test/s2/v1/fog/3/0'}
    )
    with pytest.raises(ContractError, match='sample_key'):
        validate_artifact_bundle(manifest, prediction, audit)
