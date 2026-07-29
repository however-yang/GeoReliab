'''Command-line entry points for protocol checks, dry-run, and gate evaluation.'''

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

from .contracts import (
    AuditRecord,
    PredictionArtifact,
    RunManifest,
    RunMode,
    ScientificValidity,
    SampleKey,
    read_json_artifact,
    validate_artifact_linkage,
    write_json_artifact,
)
from .gates import (
    DownstreamHarmEvidence,
    GeometryEvidence,
    GeometryGateInput,
    GeoReliabConditionEvidence,
    GeoReliabGateInput,
    SelectedTrack,
    ZeroUpdateEvidence,
    evaluate_geometry_gate,
    evaluate_georeliab_gate,
    select_track,
)
from .protocol import ProtocolConfig
from .readiness import assess_readiness
from .prepare_cli import PREPARE_OPERATIONS, run_prepare_operation
from .splits import validate_scene_disjoint
from .statistics import holm_adjust, paired_scene_bootstrap, tost_equivalence
from .audit import (
    AuditError,
    load_georeliab_evidence_input,
    load_stage_evidence_manifest,
    write_dense_audit_bundle,
)
from .storage_audit import (
    StorageAuditError,
    apply_storage_plan,
    storage_audit,
)


DEFAULT_PROTOCOL = Path('configs/dual_mve_protocol.toml')


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path} must contain a JSON object')
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _require_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload[key]
    if type(value) is not bool:
        raise ValueError(f'{key} must be a JSON boolean')
    return value


def _require_string_list(
    payload: Mapping[str, Any], key: str
) -> tuple[str, ...]:
    value = payload[key]
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f'{key} must be a JSON array of non-empty strings')
    return tuple(value)


def _require_records(
    payload: Mapping[str, Any], key: str
) -> list[dict[str, Any]]:
    value = payload[key]
    if not isinstance(value, list) or any(
        not isinstance(item, dict) for item in value
    ):
        raise ValueError(f'{key} must be a JSON array of objects')
    return value


def _require_scientific_gate_metadata(
    payload: Mapping[str, Any]
) -> tuple[RunMode, str]:
    try:
        run_mode = RunMode(payload['run_mode'])
        schema_version = payload['evidence_schema_version']
    except KeyError as exc:
        raise ValueError(f'missing required scientific gate field: {exc.args[0]}') from exc
    if run_mode is not RunMode.REAL:
        raise ValueError('scientific gate JSON run_mode must be real')
    if schema_version != '1.1':
        raise ValueError('scientific gate JSON requires evidence_schema_version 1.1')
    return run_mode, schema_version


def _geometry_input(payload: Mapping[str, Any]) -> GeometryGateInput:
    run_mode, evidence_schema_version = _require_scientific_gate_metadata(payload)
    return GeometryGateInput(
        scientific_validity=ScientificValidity(payload['scientific_validity']),
        run_mode=run_mode,
        evidence_schema_version=evidence_schema_version,
        reproducible_checkpoints=_require_string_list(
            payload, 'reproducible_checkpoints'
        ),
        hookable_models=_require_string_list(payload, 'hookable_models'),
        required_datasets_ready=_require_bool(payload, 'required_datasets_ready'),
        fixed_inputs_verified=_require_bool(payload, 'fixed_inputs_verified'),
        zeroing_effective=_require_bool(payload, 'zeroing_effective'),
        matched_intervention_effective=_require_bool(
            payload, 'matched_intervention_effective'
        ),
        evidence=tuple(
            GeometryEvidence(**item)
            for item in _require_records(payload, 'evidence')
        ),
    )


def _georeliab_input(payload: Mapping[str, Any]) -> GeoReliabGateInput:
    run_mode, evidence_schema_version = _require_scientific_gate_metadata(payload)
    return GeoReliabGateInput(
        scientific_validity=ScientificValidity(payload['scientific_validity']),
        run_mode=run_mode,
        evidence_schema_version=evidence_schema_version,
        required_models_ready=_require_string_list(
            payload, 'required_models_ready'
        ),
        required_datasets_ready=_require_bool(payload, 'required_datasets_ready'),
        tartanair_native_fog_sanity=_require_bool(
            payload, 'tartanair_native_fog_sanity'
        ),
        conditions=tuple(
            GeoReliabConditionEvidence(**item)
            for item in _require_records(payload, 'conditions')
        ),
        downstream_harm=tuple(
            DownstreamHarmEvidence(**item)
            for item in _require_records(payload, 'downstream_harm')
        ),
        zero_update=tuple(
            ZeroUpdateEvidence(**item)
            for item in _require_records(payload, 'zero_update')
        ),
        split=payload.get('split', 'test'),
        schedule_counts=dict(payload.get('schedule_counts', {})),
        downstream_schedule_counts=dict(payload.get('downstream_schedule_counts', {})),
    )


def run_dry_run(protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    '''Exercise every local boundary without producing scientific evidence.'''

    protocol = ProtocolConfig.load(protocol_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        'dev': ['fixture-dev'],
        'reference-token': ['fixture-reference'],
        'calibration': ['fixture-calibration'],
        'test': ['fixture-test'],
    }
    split_report = validate_scene_disjoint(splits)
    sample_key = SampleKey(
        dataset='fixture',
        split='test',
        scene='fixture-test',
        view_set='views-0001',
        condition='clean',
        severity='0',
        seed='0',
    )
    manifest = RunManifest(
        run_id='DRYRUN-001',
        mode=RunMode.FIXTURE,
        scientific_validity=ScientificValidity.NON_SCIENTIFIC_FIXTURE,
        model='fixture-model',
        checkpoint_hash='fixture-not-a-checkpoint',
        dataset='fixture',
        split='test',
        seed=0,
        intervention_version='geometry-v1',
        corruption_version='georeliab-c-v1',
        environment={
            'python': platform.python_version(),
            'platform': platform.platform(),
        },
        rgb_digest='fixture-rgb-fixed',
        prompt_digest='fixture-prompt-fixed',
        decoder_digest='fixture-decoder-fixed',
    )
    prediction = PredictionArtifact(
        run_id=manifest.run_id,
        sample_key=str(sample_key),
        geometry_prediction_uri='fixture://geometry/0001',
        native_confidence_uri='fixture://confidence/0001',
        valid_mask_uri='fixture://mask/0001',
        hook_location='fixture.fusion.layer0',
        runtime_seconds=0.0,
        peak_memory_mb=0.0,
    )
    audit = AuditRecord(
        run_id=manifest.run_id,
        sample_key=str(sample_key),
        gt_error=0.1,
        failure_label=False,
        selection_score=0.8,
        coverage=1.0,
        accepted=True,
        downstream_outcome=0.5,
        metadata={'fixture': 'true'},
    )
    validate_artifact_linkage(manifest, prediction, audit)
    readiness = assess_readiness(
        protocol.resources,
        mode=RunMode.FIXTURE,
        requirements=protocol.resource_groups,
    )

    bootstrap = paired_scene_bootstrap(
        {'s1': 0.4, 's2': 0.5, 's3': 0.6, 's4': 0.7},
        {'s1': 0.3, 's2': 0.4, 's3': 0.5, 's4': 0.6},
        n_resamples=10_000,
        seed=0,
    )
    tost = tost_equivalence(
        {'s1': 0.001, 's2': -0.001, 's3': 0.002, 's4': -0.002}
    )
    holm = holm_adjust({'geometry': 0.01, 'georeliab': 0.04})

    validity = ScientificValidity.NON_SCIENTIFIC_FIXTURE
    geometry = evaluate_geometry_gate(
        GeometryGateInput(
            scientific_validity=validity,
            reproducible_checkpoints=('fixture-a', 'fixture-b'),
            hookable_models=('fixture-a', 'fixture-b'),
            required_datasets_ready=True,
            fixed_inputs_verified=True,
            zeroing_effective=True,
            matched_intervention_effective=True,
            evidence=(),
        )
    )
    georeliab = evaluate_georeliab_gate(
        GeoReliabGateInput(
            scientific_validity=validity,
            required_models_ready=('fixture-a', 'fixture-b'),
            required_datasets_ready=True,
            tartanair_native_fog_sanity=True,
            conditions=(),
            downstream_harm=(),
            zero_update=(),
        )
    )
    selection = select_track(geometry, georeliab)

    write_json_artifact(output_dir / 'run_manifest.json', manifest)
    write_json_artifact(output_dir / 'prediction_artifact.json', prediction)
    write_json_artifact(output_dir / 'audit_record.json', audit)
    _write_json(output_dir / 'readiness.json', readiness.to_dict())
    _write_json(output_dir / 'geometry_gate.json', geometry.to_dict())
    _write_json(output_dir / 'georeliab_gate.json', georeliab.to_dict())
    _write_json(output_dir / 'selection.json', selection.to_dict())
    _write_json(
        output_dir / 'statistics.json',
        {
            'bootstrap': bootstrap.to_dict(),
            'tost': tost.to_dict(),
            'holm': {name: result.to_dict() for name, result in holm.items()},
        },
    )
    summary = {
        'scientific_validity': validity.value,
        'notice': 'NON-SCIENTIFIC fixture output; never cite as experiment evidence',
        'selection': selection.selected_track.value,
        'protocol_version': protocol.protocol_version,
        'split_scene_count': split_report.total_scenes,
        'output_dir': str(output_dir.resolve()),
    }
    _write_json(output_dir / 'dry_run_summary.json', summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m georeliab_mve')
    subparsers = parser.add_subparsers(dest='command', required=True)

    dry_run = subparsers.add_parser('dry-run')
    dry_run.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    dry_run.add_argument('--output-dir', type=Path, required=True)

    readiness = subparsers.add_parser('readiness')
    readiness.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)

    evaluate = subparsers.add_parser('evaluate-gates')
    evaluate.add_argument('--geometry', type=Path, required=True)
    evaluate.add_argument('--georeliab', type=Path, required=True)
    evaluate.add_argument('--output', type=Path, required=True)

    validate = subparsers.add_parser('validate-artifact')
    validate.add_argument(
        '--type',
        choices=('manifest', 'prediction', 'audit'),
        required=True,
    )
    validate.add_argument('path', type=Path)
    prepare = subparsers.add_parser('prepare-georeliab')
    prepare.add_argument('--operation', choices=PREPARE_OPERATIONS, required=True)
    prepare.add_argument('--data-root', type=Path, required=True)
    prepare.add_argument('--state', type=Path, required=True)
    prepare.add_argument('--overlay', type=Path)
    prepare.add_argument('--stage', choices=('smoke', 'test'))
    prepare.add_argument('--dry-run', action='store_true')

    audit = subparsers.add_parser('audit-georeliab')
    audit.add_argument('--input', type=Path)
    audit.add_argument('--stage-evidence', type=Path)
    audit.add_argument('--output', type=Path)
    audit.add_argument('--manifest', type=Path)
    audit.add_argument('--prediction', type=Path)
    audit.add_argument('--gt-points', type=Path)
    audit.add_argument('--gt-cameras', type=Path)
    audit.add_argument('--obs-mask', type=Path)
    audit.add_argument('--obs-bb')
    audit.add_argument('--obs-res', type=float)
    audit.add_argument('--output-dir', type=Path)

    storage = subparsers.add_parser('storage-audit')
    storage.add_argument('--root', type=Path, default=Path('.'))
    storage.add_argument('--output-dir', type=Path)
    storage.add_argument('--apply-plan', type=Path)
    storage.add_argument('--expected-plan-sha256')

    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] in {'preflight-real', 'run-georeliab'}:
        from .runner import cli_main as runner_cli_main

        return runner_cli_main(raw_argv)
    args = build_parser().parse_args(argv)
    try:
        if args.command == 'dry-run':
            summary = run_dry_run(args.protocol, args.output_dir)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if args.command == 'readiness':
            protocol = ProtocolConfig.load(args.protocol)
            report = assess_readiness(
                protocol.resources,
                mode=RunMode.REAL,
                requirements=protocol.resource_groups,
            )
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
            return 0 if report.ready else 2
        if args.command == 'evaluate-gates':
            geometry = evaluate_geometry_gate(_geometry_input(_load_json(args.geometry)))
            georeliab_payload = _load_json(args.georeliab)
            stage_path_value = georeliab_payload.get('stage_evidence_path')
            stage_sha_value = georeliab_payload.get('stage_evidence_sha256')
            if not isinstance(stage_path_value, str) or not isinstance(stage_sha_value, str):
                raise ValueError('evaluate-gates requires GeoReliab stage audit output with stage_evidence_path/sha256')
            stage_path = Path(stage_path_value)
            if _sha256_file(stage_path) != stage_sha_value:
                raise ValueError('GeoReliab stage evidence digest mismatch')
            georeliab_evidence = load_stage_evidence_manifest(stage_path)
            georeliab = evaluate_georeliab_gate(georeliab_evidence.to_gate_input())
            selection = select_track(geometry, georeliab)
            payload = {
                'geometry': geometry.to_dict(),
                'georeliab': georeliab.to_dict(),
                'selection': selection.to_dict(),
            }
            _write_json(args.output, payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
            if selection.selected_track in (
                SelectedTrack.GEOMETRY,
                SelectedTrack.GEORELIAB,
            ):
                return 0
            if selection.selected_track is SelectedTrack.STOP:
                return 1
            return 2
        if args.command == 'validate-artifact':
            artifact_types = {
                'manifest': RunManifest,
                'prediction': PredictionArtifact,
                'audit': AuditRecord,
            }
            read_json_artifact(args.path, artifact_types[args.type])
            print(f'VALID {args.type}: {args.path}')
            return 0
        if args.command == 'prepare-georeliab':
            payload = run_prepare_operation(
                operation=args.operation,
                data_root=args.data_root,
                state_path=args.state,
                dry_run=args.dry_run,
                overlay_path=args.overlay,
                stage=args.stage,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == 'audit-georeliab':
            if args.stage_evidence is not None:
                if args.output is None:
                    raise ValueError('stage evidence audit requires --output')
                evidence = load_stage_evidence_manifest(args.stage_evidence)
                gate_input = evidence.to_gate_input()
                georeliab = evaluate_georeliab_gate(gate_input)
                payload = {
                    'stage_evidence_path': str(args.stage_evidence),
                    'stage_evidence_sha256': _sha256_file(args.stage_evidence),
                    'gate_input': evidence.to_dict(),
                    'georeliab_gate': georeliab.to_dict(),
                    'p5_skip_reason': evidence.p5_skip_reason,
                }
                _write_json(args.output, payload)
                print(json.dumps(payload, indent=2, sort_keys=True))
                return 0
            if args.manifest is not None:
                required = (
                    args.prediction,
                    args.gt_points,
                    args.gt_cameras,
                    args.obs_mask,
                    args.obs_bb,
                    args.obs_res,
                    args.output_dir,
                )
                if any(item is None for item in required):
                    raise ValueError('bundle audit requires prediction, GT, ObsMask, and output-dir arguments')
                obs_bb = [float(item) for item in args.obs_bb.split(',')]
                payload = write_dense_audit_bundle(
                    manifest_path=args.manifest,
                    prediction_path=args.prediction,
                    gt_points_path=args.gt_points,
                    gt_cameras_path=args.gt_cameras,
                    obs_mask_path=args.obs_mask,
                    obs_bb=obs_bb,
                    obs_res=args.obs_res,
                    output_dir=args.output_dir,
                )
                print(json.dumps(payload, indent=2, sort_keys=True))
                return 0
            if args.input is not None:
                raise ValueError('scientific GeoReliab audit requires --stage-evidence; aggregate --input is disabled')
            raise ValueError('GeoReliab audit requires --stage-evidence or bundle audit arguments')
        if args.command == 'storage-audit':
            source_root = Path(__file__).resolve().parents[1]
            if args.apply_plan is not None:
                if not args.expected_plan_sha256:
                    raise ValueError('--apply-plan requires --expected-plan-sha256')
                payload = apply_storage_plan(
                    args.root,
                    args.apply_plan,
                    expected_plan_sha256=args.expected_plan_sha256,
                )
            else:
                snapshot, plan = storage_audit(
                    args.root,
                    source_root=source_root,
                    output_dir=args.output_dir,
                )
                output_dir = args.output_dir or args.root / 'artifacts'
                payload = {
                    'status': snapshot['status'],
                    'storage_before': str(output_dir / 'storage_before.json'),
                    'storage_plan': str(output_dir / 'storage_plan.json'),
                    'logical_bytes': snapshot['logical_bytes'],
                    'allocated_bytes': snapshot['allocated_bytes'],
                    'projection': snapshot['projection'],
                    'plan_file_sha256': plan['plan_file_sha256'],
                }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload.get('status') == 'PASS' else 2
    except (KeyError, TypeError, ValueError, AuditError, StorageAuditError, OSError, json.JSONDecodeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    raise AssertionError('unreachable command')
