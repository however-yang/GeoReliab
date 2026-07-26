from __future__ import annotations

import json

import pytest

from georeliab_mve.contracts import (
    AuditRecord,
    ContractError,
    PredictionArtifact,
    RunManifest,
    RunMode,
    SampleKey,
    ScientificValidity,
    read_json_artifact,
    validate_artifact_linkage,
    write_json_artifact,
)


def fixture_manifest() -> RunManifest:
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
    )


def test_artifact_schema_round_trip(tmp_path):
    manifest = fixture_manifest()
    path = tmp_path / 'manifest.json'
    write_json_artifact(path, manifest)
    loaded = read_json_artifact(path, RunManifest)
    assert loaded == manifest
    assert json.loads(path.read_text(encoding='utf-8'))['schema_version'] == '1.0'


def test_prediction_and_audit_round_trip(tmp_path):
    key = 'dtu/test/scene-001/views-01/fog/3/0'
    prediction = PredictionArtifact(
        run_id='RUN-001',
        sample_key=key,
        geometry_prediction_uri='file:///prediction.bin',
        native_confidence_uri='file:///confidence.bin',
        valid_mask_uri='file:///mask.bin',
        hook_location=None,
        runtime_seconds=1.0,
        peak_memory_mb=128.0,
    )
    audit = AuditRecord(
        run_id='RUN-001',
        sample_key=key,
        gt_error=0.2,
        failure_label=True,
        selection_score=0.9,
        coverage=0.5,
        accepted=False,
        downstream_outcome=-0.1,
    )
    for name, artifact, artifact_type in (
        ('prediction.json', prediction, PredictionArtifact),
        ('audit.json', audit, AuditRecord),
    ):
        path = tmp_path / name
        write_json_artifact(path, artifact)
        assert read_json_artifact(path, artifact_type) == artifact


@pytest.mark.parametrize(
    'value',
    (
        'too/few/parts',
        'dtu/test/scene/view/fog/3/0/extra',
        'dtu/test/scene with space/view/fog/3/0',
        '/test/scene/view/fog/3/0',
    ),
)
def test_sample_key_rejects_invalid_wire_format(value):
    with pytest.raises(ContractError):
        SampleKey.parse(value)


def test_invalid_prediction_must_be_retained_as_failure():
    with pytest.raises(ContractError, match='counted as failures'):
        AuditRecord(
            run_id='RUN-001',
            sample_key='dtu/test/s1/v1/fog/3/0',
            gt_error=None,
            failure_label=False,
            selection_score=0.0,
            coverage=0.0,
            accepted=False,
            downstream_outcome=None,
            invalid_prediction=True,
        )


def test_real_manifest_requires_checkpoint_sha256():
    payload = fixture_manifest().to_dict()
    payload['mode'] = 'real'
    payload['scientific_validity'] = 'SCIENTIFIC'
    with pytest.raises(ContractError, match='SHA-256'):
        RunManifest.from_dict(payload)


def test_artifact_linkage_rejects_cross_run_records():
    manifest = fixture_manifest()
    key = 'dtu/test/s1/v1/fog/3/0'
    prediction = PredictionArtifact(
        run_id='OTHER-RUN',
        sample_key=key,
        geometry_prediction_uri='fixture://geometry',
        native_confidence_uri='fixture://confidence',
        valid_mask_uri='fixture://mask',
        hook_location=None,
        runtime_seconds=0.0,
        peak_memory_mb=0.0,
    )
    audit = AuditRecord(
        run_id=manifest.run_id,
        sample_key=key,
        gt_error=0.0,
        failure_label=False,
        selection_score=0.0,
        coverage=1.0,
        accepted=True,
        downstream_outcome=0.0,
    )
    with pytest.raises(ContractError, match='run_id'):
        validate_artifact_linkage(manifest, prediction, audit)


def linked_records(
    manifest,
    sample_key,
    *,
    prediction_invalid=False,
    audit_invalid=False,
):
    prediction = PredictionArtifact(
        run_id=manifest.run_id,
        sample_key=sample_key,
        geometry_prediction_uri="fixture://geometry",
        native_confidence_uri="fixture://confidence",
        valid_mask_uri="fixture://mask",
        hook_location=None,
        runtime_seconds=0.0,
        peak_memory_mb=0.0,
        invalid_prediction=prediction_invalid,
    )
    audit = AuditRecord(
        run_id=manifest.run_id,
        sample_key=sample_key,
        gt_error=None if audit_invalid else 0.0,
        failure_label=audit_invalid,
        selection_score=0.0,
        coverage=0.0 if audit_invalid else 1.0,
        accepted=not audit_invalid,
        downstream_outcome=None if audit_invalid else 0.0,
        invalid_prediction=audit_invalid,
    )
    return prediction, audit


def test_artifact_linkage_rejects_invalidity_mismatch():
    manifest = fixture_manifest()
    prediction, audit = linked_records(
        manifest,
        "fixture/test/s1/v1/fog/3/0",
        prediction_invalid=True,
        audit_invalid=False,
    )
    with pytest.raises(ContractError, match="invalid_prediction mismatch"):
        validate_artifact_linkage(manifest, prediction, audit)


def test_artifact_linkage_rejects_cross_dataset_sample():
    manifest = fixture_manifest()
    prediction, audit = linked_records(
        manifest,
        "dtu/test/s1/v1/fog/3/0",
    )
    with pytest.raises(ContractError, match="dataset/split"):
        validate_artifact_linkage(manifest, prediction, audit)


def test_artifact_linkage_accepts_consistent_invalid_failure():
    manifest = fixture_manifest()
    prediction, audit = linked_records(
        manifest,
        "fixture/test/s1/v1/fog/3/0",
        prediction_invalid=True,
        audit_invalid=True,
    )
    validate_artifact_linkage(manifest, prediction, audit)


def test_artifact_linkage_rejects_seed_mismatch():
    manifest = fixture_manifest()
    prediction, audit = linked_records(
        manifest,
        "fixture/test/s1/v1/fog/3/1",
    )
    with pytest.raises(ContractError, match="seed"):
        validate_artifact_linkage(manifest, prediction, audit)
