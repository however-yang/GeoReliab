from __future__ import annotations

import json
from pathlib import Path

from georeliab_mve.cli import _geometry_input, _georeliab_input, main, run_dry_run
from georeliab_mve.contracts import RunMode
from georeliab_mve.protocol import ProtocolConfig, ProtocolError
from georeliab_mve.readiness import (
    ResourceGroupRequirement,
    ResourceSpec,
    assess_readiness,
)


PROTOCOL = Path('configs/dual_mve_protocol.toml')


def test_readiness_materializes_generator(tmp_path):
    first = tmp_path / 'first.bin'
    second = tmp_path / 'second.bin'
    first.write_bytes(b'first')
    second.write_bytes(b'second')
    resources = (
        ResourceSpec(name, 'checkpoint', str(path))
        for name, path in (('first', first), ('second', second))
    )
    report = assess_readiness(resources, mode=RunMode.REAL)
    assert report.ready is True
    assert len(report.checks) == 2


def test_missing_required_resource_fails_closed():
    report = assess_readiness(
        [ResourceSpec('missing', 'dataset', None)], mode=RunMode.REAL
    )
    assert report.ready is False
    assert report.checks[0].ready is False


def test_candidate_group_accepts_any_two_of_three(tmp_path):
    first = tmp_path / 'first.bin'
    third = tmp_path / 'third.bin'
    first.write_bytes(b'first')
    third.write_bytes(b'third')
    resources = (
        ResourceSpec('first', 'checkpoint', str(first), required=False),
        ResourceSpec('second', 'checkpoint', None, required=False),
        ResourceSpec('third', 'checkpoint', str(third), required=False),
    )
    report = assess_readiness(
        resources,
        mode=RunMode.REAL,
        requirements=(
            ResourceGroupRequirement('models', ('first', 'second', 'third'), 2),
        ),
    )
    assert report.ready is True
    assert report.group_checks[0].ready_members == ('first', 'third')


def test_default_protocol_is_frozen_and_initially_blocked():
    protocol = ProtocolConfig.load(PROTOCOL)
    report = assess_readiness(
        protocol.resources,
        mode=RunMode.REAL,
        requirements=protocol.resource_groups,
    )
    assert protocol.internal_freeze == '2026-09-15'
    assert report.ready is False
    assert {group.minimum_ready for group in report.group_checks} == {2}


def test_protocol_rejects_fake_geometry_readiness_members(tmp_path):
    payload = PROTOCOL.read_text(encoding='utf-8').replace(
        "members = ['spatial-mllm-checkpoint', 'spatialstack-checkpoint', "
        "'guide-checkpoint']",
        "members = ['fake-a', 'fake-b', 'fake-c']",
    )
    drifted = tmp_path / 'fake-geometry-members.toml'
    drifted.write_text(payload, encoding='utf-8')
    try:
        ProtocolConfig.load(drifted)
    except ProtocolError as exc:
        assert 'Geometry readiness' in str(exc)
    else:
        raise AssertionError('fake Geometry readiness members were accepted')


def test_protocol_rejects_fake_georeliab_readiness_members(tmp_path):
    payload = PROTOCOL.read_text(encoding='utf-8').replace(
        "members = ['vggt-checkpoint', 'mast3r-checkpoint']",
        "members = ['fake-v', 'fake-m']",
    )
    drifted = tmp_path / 'fake-georeliab-members.toml'
    drifted.write_text(payload, encoding='utf-8')
    try:
        ProtocolConfig.load(drifted)
    except ProtocolError as exc:
        assert 'GeoReliab MVE readiness' in str(exc)
    else:
        raise AssertionError('fake GeoReliab readiness members were accepted')


def test_fixture_dry_run_is_explicitly_non_scientific(tmp_path):
    output = tmp_path / 'dry-run'
    summary = run_dry_run(PROTOCOL, output)
    assert summary['scientific_validity'] == 'NON_SCIENTIFIC_FIXTURE'
    assert summary['selection'] == 'BLOCKED_NON_SCIENTIFIC_FIXTURE'
    assert 'NON-SCIENTIFIC' in summary['notice']
    selection = json.loads((output / 'selection.json').read_text(encoding='utf-8'))
    assert selection['selected_track'] == 'BLOCKED_NON_SCIENTIFIC_FIXTURE'
    assert (output / 'run_manifest.json').exists()
    assert (output / 'statistics.json').exists()
    manifest = json.loads((output / 'run_manifest.json').read_text(encoding='utf-8'))
    prediction = json.loads(
        (output / 'prediction_artifact.json').read_text(encoding='utf-8')
    )
    assert prediction['run_id'] == manifest['run_id']


def test_cli_dry_run_and_artifact_validation(tmp_path):
    output = tmp_path / 'cli-dry-run'
    assert main(['dry-run', '--protocol', str(PROTOCOL), '--output-dir', str(output)]) == 0
    assert main(
        [
            'validate-artifact',
            '--type',
            'manifest',
            str(output / 'run_manifest.json'),
        ]
    ) == 0


def test_cli_readiness_returns_blocked_exit_code():
    assert main(['readiness', '--protocol', str(PROTOCOL)]) == 2


def test_gate_json_parser_rejects_string_booleans_and_string_lists():
    payload = {
        'scientific_validity': 'SCIENTIFIC',
        'run_mode': 'real',
        'evidence_schema_version': '1.1',
        'reproducible_checkpoints': ['Spatial-MLLM', 'SpatialStack'],
        'hookable_models': ['Spatial-MLLM', 'SpatialStack'],
        'required_datasets_ready': 'false',
        'fixed_inputs_verified': True,
        'zeroing_effective': False,
        'matched_intervention_effective': True,
        'evidence': [],
    }
    try:
        _geometry_input(payload)
    except ValueError as exc:
        assert 'JSON boolean' in str(exc)
    else:
        raise AssertionError('string boolean was accepted')
    payload['required_datasets_ready'] = True
    payload['reproducible_checkpoints'] = 'Spatial-MLLM'
    try:
        _geometry_input(payload)
    except ValueError as exc:
        assert 'JSON array' in str(exc)
    else:
        raise AssertionError('string list was accepted')
    payload['reproducible_checkpoints'] = ['Spatial-MLLM', 'SpatialStack']
    payload['evidence'] = [
        {
            'model': 'Spatial-MLLM',
            'benchmark': 'VSI-Bench',
            'sample_class': 'geometry-required',
            'stratum': 'distance',
            'delta_geom': 0.1,
            'ci_lower': 0.01,
            'recovery': 0.4,
            'equivalent_by_tost': 'false',
            'post_fusion_changed': True,
            'semantic_control_unchanged': True,
        }
    ]
    try:
        _geometry_input(payload)
    except ValueError as exc:
        assert 'must be boolean' in str(exc)
    else:
        raise AssertionError('string evidence boolean was accepted')


def test_gate_json_parser_requires_current_real_evidence_metadata():
    geometry = {
        'scientific_validity': 'SCIENTIFIC',
        'reproducible_checkpoints': [],
        'hookable_models': [],
        'required_datasets_ready': False,
        'fixed_inputs_verified': False,
        'zeroing_effective': False,
        'matched_intervention_effective': False,
        'evidence': [],
    }
    for key, value in (
        ('run_mode', None),
        ('run_mode', 'fixture'),
        ('run_mode', 'smoke'),
        ('evidence_schema_version', None),
        ('evidence_schema_version', '1.0'),
    ):
        candidate = dict(geometry)
        if value is not None:
            candidate[key] = value
        try:
            _geometry_input(candidate)
        except ValueError:
            pass
        else:
            raise AssertionError(f'{key}={value!r} was accepted')
    georeliab = {
        'scientific_validity': 'SCIENTIFIC',
        'run_mode': 'real',
        'evidence_schema_version': '1.1',
        'required_models_ready': [],
        'required_datasets_ready': False,
        'tartanair_native_fog_sanity': False,
        'conditions': [],
        'downstream_harm': [],
        'zero_update': [],
    }
    assert _georeliab_input(georeliab).run_mode is RunMode.REAL


def test_cli_rejects_hand_authored_georeliab_selection_input(tmp_path):
    scenes = [f's{i:02d}' for i in range(20)]
    geometry = {
        'scientific_validity': 'SCIENTIFIC',
        'run_mode': 'real',
        'evidence_schema_version': '1.1',
        'reproducible_checkpoints': ['Spatial-MLLM', 'SpatialStack'],
        'hookable_models': ['Spatial-MLLM', 'SpatialStack'],
        'required_datasets_ready': True,
        'zeroing_effective': False,
        'fixed_inputs_verified': True,
        'matched_intervention_effective': True,
        'evidence': [],
    }
    conditions = [
        {
            'model': model,
            'corruption': corruption,
            'clean_rho': 0.8,
            'severity_rhos': [0.6, 0.4, 0.2],
            'failure_auroc': 0.7,
            'relative_decline_ci_lower': 0.6,
            'scene_ids': scenes,
            'scene_count': 20,
            'n_resamples': 10000,
            'relative_decline_raw_p': 0.001,
            'relative_decline_adjusted_p': 0.006,
            'relative_decline_holm_rejected': True,
            'corruption_severity_monotonic': True,
            'cross_view_consistent': True,
            'gt_geometry_invariant': True,
        }
        for model in ('VGGT', 'MASt3R')
        for corruption in ('fog', 'low-light-noise', 'defocus')
    ]
    georeliab = {
        'scientific_validity': 'SCIENTIFIC',
        'run_mode': 'real',
        'evidence_schema_version': '1.1',
        'required_models_ready': ['VGGT', 'MASt3R'],
        'required_datasets_ready': True,
        'tartanair_native_fog_sanity': True,
        'conditions': conditions,
        'downstream_harm': [
            {
                'model': model,
                'condition': f'{corruption}-s2',
                'effect_vs_random': -0.02 if corruption in ('fog', 'defocus') else 0.01,
                'ci_upper': -0.01 if corruption in ('fog', 'defocus') else 0.02,
            }
            for model in ('VGGT', 'MASt3R')
            for corruption in ('fog', 'low-light-noise', 'defocus')
        ],
        'zero_update': [
            {
                'model': model,
                'condition': f'{corruption}-s2',
                'auroc_gain': 0.10 if model == 'VGGT' and corruption == 'fog' else 0.0,
                'ci_lower': 0.01 if model == 'VGGT' and corruption == 'fog' else -0.01,
            }
            for model in ('VGGT', 'MASt3R')
            for corruption in ('fog', 'low-light-noise', 'defocus')
        ],
        'schedule_counts': {'scheduled': 400, 'completed': 400, 'missing': 0, 'invalid': 0},
        'downstream_schedule_counts': {'scheduled': 6, 'completed': 6, 'missing': 0},
    }
    geometry_path = tmp_path / 'geometry.json'
    georeliab_path = tmp_path / 'georeliab.json'
    output_path = tmp_path / 'decision.json'
    geometry_path.write_text(json.dumps(geometry), encoding='utf-8')
    georeliab_path.write_text(json.dumps(georeliab), encoding='utf-8')
    exit_code = main(
        [
            'evaluate-gates',
            '--geometry',
            str(geometry_path),
            '--georeliab',
            str(georeliab_path),
            '--output',
            str(output_path),
        ]
    )
    assert exit_code == 2
    assert not output_path.exists()
