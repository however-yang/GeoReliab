'''Command-line entry points for protocol checks, dry-run, and gate evaluation.'''

from __future__ import annotations

import argparse
from contextlib import redirect_stderr
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
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
    load_stage_evidence_manifest,
    write_dense_audit_bundle,
)
from .storage_audit import (
    StorageAuditError,
    apply_storage_plan,
    storage_audit,
)
from .execution_governance import (
    ExecutionGovernanceError,
    archive_superseded_results,
    create_p2a_selection_manifest,
    evaluate_p2a_completion,
    record_gpu_selection,
    validate_gpu_selection,
    validate_storage_refactor_rollout,
)
from .science_lock import BASE_PROJECT_COMMIT
from .v4_execution import V4ExecutionError
from .v4_rectified_closure import (
    V4RectifiedClosureError,
    create_rectified_member_closure,
    materialize_missing_rectified_members,
    prepare_rectified_resource_schedule,
    validate_rectified_member_closure,
)
from .v4_authorization import (
    create_attempt_execution_authorization,
    create_attempt_hardware_preflight,
    create_execution_authorization,
    create_hardware_preflight,
    materialize_attempt_resources,
    validate_attempt_execution_authorization,
    validate_attempt_resources,
    validate_execution_authorization,
)
from .v4_attempt03_authorization import (
    create_attempt03_execution_authorization,
    create_attempt03_gpu_preflight,
    revalidate_attempt03_resources,
    validate_attempt03_execution_authorization,
)
from .v4_attempt04_authorization import (
    create_attempt04_execution_authorization,
    create_attempt04_gpu_preflight,
    nvidia_smi_attempt04_inventory,
    revalidate_attempt04_resources,
    validate_attempt04_execution_authorization,
)
from .v4_overlay_resource_resolution import (
    OverlayResolutionError,
    parse_resource_overrides,
    resolve_overlay_resources,
    validate_overlay_resource_resolution,
)
from .v4_attempt05_execution import (
    ATTEMPT05_RUN_NAME,
    build_attempt05_scientific_schedule,
    create_attempt05_execution_preflight,
    create_attempt05_start_receipt,
    authorize_attempt05_next_dispatch,
    evaluate_attempt05_resource_gate,
    finalize_attempt05_scientific_bundle,
    load_attempt05_authorized_context,
    rehydrate_attempt05_ledger_totals,
    validate_attempt05_execution_preflight,
)
from .v4_attempt05_inputs import (
    prepare_attempt05_inputs,
    validate_attempt05_input_closure,
)
from .v4_attempt05_orchestrator import (
    construct_attempt05_runtime_bindings,
    make_attempt05_runtime_executors,
    run_attempt05_pipeline,
)
from .v4_counterfactuals import (
    ModelIndependentState,
    V4SplitAssignment,
    parse_scientific_schedule,
    validate_scientific_schedule,
)
from .v4_metrics import NativeWarningCalibration
from .v4_qualification import build_hourly_monitor_status


DEFAULT_PROTOCOL = Path('configs/dual_mve_protocol.toml')
ATTEMPT02_COMMANDS = frozenset(
    {
        'v4-attempt02-prepare-resources',
        'v4-attempt02-gpu-preflight',
        'v4-attempt02-create-execution-authorization',
        'v4-attempt02-validate-execution-authorization',
    }
)
ATTEMPT03_COMMANDS = frozenset(
    {
        'v4-attempt03-revalidate-resources',
        'v4-attempt03-gpu-preflight',
        'v4-attempt03-create-execution-authorization',
        'v4-attempt03-validate-execution-authorization',
    }
)
ATTEMPT04_COMMANDS = frozenset(
    {
        'v4-attempt04-revalidate-resources',
        'v4-attempt04-gpu-preflight',
        'v4-attempt04-create-execution-authorization',
        'v4-attempt04-validate-execution-authorization',
    }
)
OVERLAY_RESOLUTION_COMMANDS = frozenset(
    {
        'v4-resolve-overlay-resources',
        'v4-validate-overlay-resource-resolution',
    }
)
ATTEMPT05_COMMANDS = frozenset(
    {
        'v4-attempt05-prepare-inputs',
        'v4-attempt05-preflight',
        'v4-attempt05-run',
        'v4-attempt05-status',
        'v4-attempt05-finalize',
    }
)
ATTEMPT02_REQUIRED_HISTORY_PATHS = (
    Path(
        'docs/evidence/v4_gpu_authorization/'
        'fa9a784c449303de0bb4ba67db92d0fbd418e10b-attempt-01-corrected/'
        'v4-preflight-decision.corrected.json'
    ),
    Path(
        'docs/evidence/v4_gpu_authorization/'
        'fa9a784c449303de0bb4ba67db92d0fbd418e10b-attempt-01-corrected/'
        'erratum.json'
    ),
    Path(
        'docs/evidence/v4_gpu_authorization/'
        'fa9a784c449303de0bb4ba67db92d0fbd418e10b/'
        'v4-hardware-preflight.json'
    ),
    Path(
        'docs/evidence/v4_gpu_authorization/'
        'fa9a784c449303de0bb4ba67db92d0fbd418e10b/'
        'v4-preflight-decision.json'
    ),
)


def _attempt02_required_history(
    paths: list[Path], *, source_root: Path
) -> tuple[Path, ...]:
    provided = {path.resolve() for path in paths}
    required = {
        (source_root / relative).resolve()
        for relative in ATTEMPT02_REQUIRED_HISTORY_PATHS
    }
    missing = sorted(str(path) for path in required - provided)
    if missing:
        raise V4ExecutionError(
            'V4_ATTEMPT01_CANONICAL_HISTORY_REQUIRED:' + ','.join(missing)
        )
    for path in required:
        if not path.is_file():
            raise V4ExecutionError('V4_ATTEMPT01_HISTORY_MISSING:' + str(path))
    return tuple(path.resolve() for path in paths)


def _attempt02_failure_payload(exc: BaseException) -> dict[str, str]:
    return {
        'status': 'FAIL',
        'reason_code': str(exc) or type(exc).__name__,
        'error_type': type(exc).__name__,
    }


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


def _json_payload_or_list(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _attempt05_json_list(path: Path, *keys: str) -> list[Any]:
    payload = _json_payload_or_list(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f'{path} must contain a JSON list or one of {keys}')


def _attempt05_states(path: Path) -> tuple[ModelIndependentState, ...]:
    return tuple(
        ModelIndependentState.from_dict(row)
        for row in _attempt05_json_list(path, 'states', 'model_independent_states')
    )


def _attempt05_schedule(path: Path):
    return validate_scientific_schedule(
        parse_scientific_schedule(path.read_text(encoding='utf-8'))
    )


def _attempt05_split(path: Path) -> V4SplitAssignment:
    payload = _load_json(path)
    splits = payload.get('splits')
    if not isinstance(splits, Mapping):
        raise ValueError('v4 split assignment requires splits object')
    provenance = payload.get('protocol_provenance')
    if not isinstance(provenance, Mapping):
        raise ValueError('v4 split assignment requires protocol provenance')
    return V4SplitAssignment(
        protocol_provenance_items=tuple(sorted((str(k), str(v)) for k, v in provenance.items())),
        calibration=tuple(int(v) for v in splits.get('calibration', ())),
        dev=tuple(int(v) for v in splits.get('dev', ())),
        reference=tuple(int(v) for v in splits.get('reference', ())),
        test=tuple(int(v) for v in splits.get('test', ())),
        excluded_incomplete=tuple(int(v) for v in payload.get('excluded_incomplete', ())),
        unassigned_verified=tuple(int(v) for v in payload.get('unassigned_verified', ())),
        inventory_sha256=str(payload.get('inventory_sha256')),
        fingerprint_sha256=str(payload.get('fingerprint_sha256')),
        schema_version=str(payload.get('schema_version')),
    )


def _attempt05_calibration_schedule(path: Path) -> tuple[Mapping[str, object], ...]:
    return tuple(_attempt05_json_list(path, 'calibration_schedule', 'units'))


def _attempt05_calibrations(path: Path) -> tuple[NativeWarningCalibration, ...]:
    rows = _attempt05_json_list(path, 'native_warning_calibrations', 'calibrations')
    return tuple(
        NativeWarningCalibration(
            model_id=str(row['model_id']),
            scene_ids=tuple(int(v) for v in row['scene_ids']),
            sorted_warning_scores=tuple(float(v) for v in row['sorted_warning_scores']),
            alarm_threshold=float(row['alarm_threshold']),
            split_fingerprint_sha256=str(row['split_fingerprint_sha256']),
            inventory_sha256=str(row['inventory_sha256']),
            calibration_identifier=str(row['calibration_identifier']),
            schema_version=str(row.get('schema_version', 'georeliab-v4-native-warning-calibration-1.0')),
        )
        for row in rows
    )

def _attempt05_overlay_config_path(context: Any) -> Path:
    raw_receipt = context.authorization.get("resource_receipt_path")
    if not isinstance(raw_receipt, str) or not raw_receipt:
        raise ValueError("Attempt-05 authorization requires resource receipt path")
    receipt_path = Path(raw_receipt)
    receipt = _load_json(receipt_path)
    bindings = receipt.get("resource_bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("Attempt-05 resource receipt requires resource bindings")
    overlay = bindings.get("overlay")
    if not isinstance(overlay, Mapping):
        raise ValueError("Attempt-05 resource receipt requires overlay binding")
    raw_overlay_path = overlay.get("path") or overlay.get("overlay_descriptor_path")
    if not isinstance(raw_overlay_path, str) or not raw_overlay_path:
        raise ValueError("Attempt-05 resource receipt requires overlay path")
    overlay_path = Path(raw_overlay_path)
    expected_sha = overlay.get("sha256") or overlay.get("overlay_descriptor_sha256")
    if (
        isinstance(expected_sha, str)
        and expected_sha
        and hashlib.sha256(overlay_path.read_bytes()).hexdigest() != expected_sha
    ):
        raise ValueError("Attempt-05 overlay digest mismatch")
    return overlay_path

def _attempt05_authorized_gpu_inventory(
    context: Any,
    *,
    sampler: Any = nvidia_smi_attempt04_inventory,
) -> Mapping[str, object]:
    """Project a host inventory onto the one GPU bound by Attempt-04."""

    raw = sampler()
    devices = raw.get("devices") if isinstance(raw, Mapping) else None
    if not isinstance(devices, list):
        raise V4ExecutionError("V4_ATTEMPT05_PREFLIGHT_SAMPLE_INVALID")
    selected = context.selected_gpu
    matches = [
        device
        for device in devices
        if isinstance(device, Mapping)
        and device.get("uuid") == selected.get("uuid")
        and device.get("pci_bus_id") == selected.get("pci_bus_id")
        and device.get("index") == selected.get("index")
    ]
    if len(matches) != 1:
        raise V4ExecutionError("V4_ATTEMPT05_PREFLIGHT_GPU_IDENTITY_MISMATCH")
    projected = dict(raw)
    projected["devices"] = [dict(matches[0])]
    return projected


def _attempt05_get_or_create_preflight(
    context: Any,
    *,
    authorization_path: Path,
    tooling_commit: str,
    tooling_tree: str,
) -> Mapping[str, object]:
    """Resume only from a validated immutable preflight boundary."""

    preflight_path = context.run_root / "v4-attempt05-execution-preflight.json"
    partial_path = preflight_path.with_name(preflight_path.name + ".partial")
    if partial_path.exists():
        raise V4ExecutionError("V4_ATTEMPT05_PREFLIGHT_PARTIAL_PRESENT")
    if preflight_path.exists():
        preflight = validate_attempt05_execution_preflight(
            preflight_path,
            authorization_path=authorization_path,
        )
        if (
            preflight.get("attempt05_tooling_commit") != tooling_commit
            or preflight.get("attempt05_tooling_tree") != tooling_tree
        ):
            raise V4ExecutionError("V4_ATTEMPT05_TOOLING_REVISION_MISMATCH")
        return preflight
    return create_attempt05_execution_preflight(
        authorization_path=authorization_path,
        attempt05_tooling_commit=tooling_commit,
        attempt05_tooling_tree=tooling_tree,
        nvidia_smi_sampler=lambda: _attempt05_authorized_gpu_inventory(context),
        sleeper=time.sleep,
        cuda_mapping_probe=lambda: _attempt05_cuda_mapping_probe(context),
    )


def _attempt05_cuda_mapping_probe(context: Any) -> Mapping[str, object]:
    selected_index = int(context.selected_gpu["index"])
    code = f"""
import json
import subprocess
import torch
visible = torch.cuda.device_count()
props = torch.cuda.get_device_properties(0) if visible == 1 else None
query = subprocess.check_output([
    'nvidia-smi',
    '--id={selected_index}',
    '--query-gpu=uuid,pci.bus_id,name,memory.total',
    '--format=csv,noheader,nounits',
], text=True).strip()
parts = [part.strip() for part in query.split(',')]
print(json.dumps({{
    'visible_device_count': visible,
    'logical_device': 'cuda:0' if visible == 1 else None,
    'mapped_uuid': parts[0] if len(parts) >= 1 else None,
    'mapped_pci_bus_id': parts[1] if len(parts) >= 2 else None,
    'mapped_name': props.name if props is not None else None,
    'total_memory_mib': int(props.total_memory // (1024 * 1024)) if props is not None else None,
    'model_loads': 0,
    'model_forwards': 0,
    'checkpoint_loads': 0,
}}))
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(selected_index)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, Mapping):
        raise V4ExecutionError("V4_ATTEMPT05_CUDA_MAPPING_PROBE_INVALID")
    return payload



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

    archive = subparsers.add_parser('archive-superseded')
    archive.add_argument('--root', type=Path, required=True)
    archive.add_argument('--expected-project-commit', default=BASE_PROJECT_COMMIT)
    archive.add_argument('--expected-p1-items', type=int, default=16)
    archive.add_argument('--expected-p2-items', type=int, default=75)

    p2a = subparsers.add_parser('prepare-p2a')
    p2a.add_argument('--root', type=Path, required=True)
    p2a.add_argument('--storage-before', type=Path)
    p2a.add_argument('--storage-plan', type=Path)
    p2a.add_argument('--output', type=Path)

    validate_p2a = subparsers.add_parser('validate-p2a')
    validate_p2a.add_argument('--root', type=Path, required=True)
    validate_p2a.add_argument('--selection', type=Path)
    validate_p2a.add_argument('--output', type=Path)

    record_gpu = subparsers.add_parser('record-gpu-selection')
    record_gpu.add_argument('--root', type=Path, required=True)
    record_gpu.add_argument('--project-commit', required=True)
    record_gpu.add_argument('--device', choices=('cuda:0', 'cuda:1'), required=True)
    record_gpu.add_argument('--explicit-user-selection', action='store_true')

    validate_gpu = subparsers.add_parser('validate-gpu-selection')
    validate_gpu.add_argument('--root', type=Path, required=True)
    validate_gpu.add_argument('--project-commit', required=True)
    validate_gpu.add_argument('--device', choices=('cuda:0', 'cuda:1'), required=True)

    rollout = subparsers.add_parser('validate-storage-refactor')
    rollout.add_argument('--root', type=Path, required=True)
    rollout.add_argument('--output', type=Path)

    v4_preflight = subparsers.add_parser('v4-gpu-preflight')
    v4_preflight.add_argument('--output', type=Path, default=Path('artifacts/v4-hardware-preflight.json'))
    v4_preflight.add_argument('--requested-index', type=int, required=True)
    v4_preflight.add_argument('--schedule', type=Path)
    v4_preflight.add_argument('--project-commit', default='7381e60050143a78fca6a3ebde5706ae27d2c145')
    v4_preflight.add_argument('--project-tree', default='f4e2b1104496c817693aaa5989d0276d2ebe03e9')

    v4_create_auth = subparsers.add_parser('v4-create-execution-authorization')
    v4_create_auth.add_argument('--root', type=Path, default=Path('.'))
    v4_create_auth.add_argument('--receipt', type=Path, required=True)
    v4_create_auth.add_argument('--resource-inventory', type=Path, required=True)
    v4_create_auth.add_argument('--run-root', type=Path, required=True)
    v4_create_auth.add_argument('--artifact-root', type=Path, required=True)
    v4_create_auth.add_argument('--final-evidence-path', type=Path, required=True)
    v4_create_auth.add_argument('--output', type=Path, default=Path('artifacts/v4-execution-authorization.json'))

    v4_validate_auth = subparsers.add_parser('v4-validate-execution-authorization')
    v4_validate_auth.add_argument('authorization', type=Path)

    attempt02_resources = subparsers.add_parser(
        'v4-attempt02-prepare-resources'
    )
    attempt02_resources.add_argument('--root', type=Path, required=True)
    attempt02_resources.add_argument(
        '--source-manifest', type=Path, required=True
    )
    attempt02_resources.add_argument('--output', type=Path, required=True)

    attempt02_preflight = subparsers.add_parser(
        'v4-attempt02-gpu-preflight'
    )
    attempt02_preflight.add_argument('--output', type=Path, required=True)
    attempt02_preflight.add_argument(
        '--resource-snapshot', type=Path, required=True
    )
    attempt02_preflight.add_argument(
        '--historical-evidence',
        type=Path,
        action='append',
        required=True,
    )

    attempt02_create_auth = subparsers.add_parser(
        'v4-attempt02-create-execution-authorization'
    )
    attempt02_create_auth.add_argument('--root', type=Path, required=True)
    attempt02_create_auth.add_argument('--receipt', type=Path, required=True)
    attempt02_create_auth.add_argument(
        '--resource-snapshot', type=Path, required=True
    )
    attempt02_create_auth.add_argument('--run-root', type=Path, required=True)
    attempt02_create_auth.add_argument(
        '--artifact-root', type=Path, required=True
    )
    attempt02_create_auth.add_argument(
        '--final-evidence-path', type=Path, required=True
    )
    attempt02_create_auth.add_argument('--output', type=Path, required=True)

    attempt02_validate_auth = subparsers.add_parser(
        'v4-attempt02-validate-execution-authorization'
    )
    attempt02_validate_auth.add_argument('authorization', type=Path)

    attempt03_resources = subparsers.add_parser(
        'v4-attempt03-revalidate-resources'
    )
    attempt03_resources.add_argument('--worktree', type=Path, required=True)
    attempt03_resources.add_argument(
        '--runtime-root', type=Path, required=True
    )
    attempt03_resources.add_argument(
        '--rectified-root', type=Path, required=True
    )
    attempt03_resources.add_argument(
        '--closure-root', type=Path, required=True
    )
    attempt03_resources.add_argument('--overlay', type=Path, required=True)
    attempt03_resources.add_argument('--output', type=Path, required=True)

    attempt03_preflight = subparsers.add_parser(
        'v4-attempt03-gpu-preflight'
    )
    attempt03_preflight.add_argument('--worktree', type=Path, required=True)
    attempt03_preflight.add_argument(
        '--resource-receipt', type=Path, required=True
    )
    attempt03_preflight.add_argument('--output', type=Path, required=True)

    attempt03_create = subparsers.add_parser(
        'v4-attempt03-create-execution-authorization'
    )
    attempt03_create.add_argument('--worktree', type=Path, required=True)
    attempt03_create.add_argument(
        '--runtime-root', type=Path, required=True
    )
    attempt03_create.add_argument(
        '--resource-receipt', type=Path, required=True
    )
    attempt03_create.add_argument(
        '--hardware-snapshot', type=Path, required=True
    )
    attempt03_create.add_argument('--receipt', type=Path, required=True)
    attempt03_create.add_argument(
        '--authorization', type=Path, required=True
    )
    attempt03_create.add_argument('--run-root', type=Path, required=True)
    attempt03_create.add_argument('--artifact-root', type=Path, required=True)
    attempt03_create.add_argument('--gpu-ledger', type=Path, required=True)
    attempt03_create.add_argument(
        '--final-evidence-path', type=Path, required=True
    )

    attempt03_validate = subparsers.add_parser(
        'v4-attempt03-validate-execution-authorization'
    )
    attempt03_validate.add_argument('authorization', type=Path)

    attempt04_resources = subparsers.add_parser(
        'v4-attempt04-revalidate-resources'
    )
    attempt04_resources.add_argument('--worktree', type=Path, required=True)
    attempt04_resources.add_argument(
        '--runtime-root', type=Path, required=True
    )
    attempt04_resources.add_argument(
        '--rectified-root', type=Path, required=True
    )
    attempt04_resources.add_argument(
        '--closure-root', type=Path, required=True
    )
    attempt04_resources.add_argument('--overlay', type=Path, required=True)
    attempt04_resources.add_argument('--output', type=Path, required=True)

    attempt04_preflight = subparsers.add_parser(
        'v4-attempt04-gpu-preflight'
    )
    attempt04_preflight.add_argument('--worktree', type=Path, required=True)
    attempt04_preflight.add_argument(
        '--resource-receipt', type=Path, required=True
    )
    attempt04_preflight.add_argument('--output', type=Path, required=True)

    attempt04_create = subparsers.add_parser(
        'v4-attempt04-create-execution-authorization'
    )
    attempt04_create.add_argument('--worktree', type=Path, required=True)
    attempt04_create.add_argument(
        '--runtime-root', type=Path, required=True
    )
    attempt04_create.add_argument(
        '--resource-receipt', type=Path, required=True
    )
    attempt04_create.add_argument(
        '--hardware-snapshot', type=Path, required=True
    )
    attempt04_create.add_argument('--receipt', type=Path, required=True)
    attempt04_create.add_argument(
        '--authorization', type=Path, required=True
    )
    attempt04_create.add_argument('--run-root', type=Path, required=True)
    attempt04_create.add_argument('--artifact-root', type=Path, required=True)
    attempt04_create.add_argument('--gpu-ledger', type=Path, required=True)
    attempt04_create.add_argument(
        '--final-evidence-path', type=Path, required=True
    )

    attempt04_validate = subparsers.add_parser(
        'v4-attempt04-validate-execution-authorization'
    )
    attempt04_validate.add_argument('authorization', type=Path)
    overlay_resolve = subparsers.add_parser(
        'v4-resolve-overlay-resources'
    )
    overlay_resolve.add_argument('--worktree', type=Path, required=True)
    overlay_resolve.add_argument(
        '--runtime-root', type=Path, required=True
    )
    overlay_resolve.add_argument('--overlay', type=Path, required=True)
    overlay_resolve.add_argument(
        '--frozen-materialization', type=Path, required=True
    )
    overlay_resolve.add_argument('--output-dir', type=Path, required=True)
    overlay_resolve.add_argument(
        '--resource-override', action='append', default=[]
    )
    overlay_validate = subparsers.add_parser(
        'v4-validate-overlay-resource-resolution'
    )
    overlay_validate.add_argument('receipt', type=Path)

    bootstrap_rectified = subparsers.add_parser('v4-prepare-rectified-resource-schedule')
    bootstrap_rectified.add_argument('--protocol', type=Path, required=True)
    bootstrap_rectified.add_argument('--split-view-manifest', type=Path, required=True)
    bootstrap_rectified.add_argument('--output-dir', type=Path, required=True)
    rectified = subparsers.add_parser('v4-rectified-closure')
    rectified.add_argument('--root', type=Path, required=True)
    rectified.add_argument('--expected-set', type=Path, required=True)
    rectified.add_argument('--output-dir', type=Path, required=True)

    validate_rectified = subparsers.add_parser('v4-validate-rectified-closure')
    validate_rectified.add_argument('--root', type=Path, required=True)
    validate_rectified.add_argument('--expected-set', type=Path, required=True)
    validate_rectified.add_argument('--manifest', type=Path, required=True)
    validate_rectified.add_argument('--output-dir', type=Path, required=True)

    materialize_rectified = subparsers.add_parser(
        'v4-materialize-missing-rectified-members'
    )
    materialize_rectified.add_argument('--root', type=Path, required=True)
    materialize_rectified.add_argument('--expected-set', type=Path, required=True)
    materialize_rectified.add_argument('--official-rectified-archive', required=True)
    attempt05_prepare = subparsers.add_parser('v4-attempt05-prepare-inputs')
    attempt05_prepare.add_argument('--authorization', type=Path, required=True)
    attempt05_prepare.add_argument('--dtu-root', type=Path, required=True)
    attempt05_prepare.add_argument('--rectified-closure-manifest', type=Path, required=True)
    attempt05_prepare.add_argument('--fog-root', type=Path, required=True)
    attempt05_prepare.add_argument('--output-dir', type=Path, required=True)

    attempt05_preflight = subparsers.add_parser('v4-attempt05-preflight')
    attempt05_preflight.add_argument('--authorization', type=Path, required=True)
    attempt05_preflight.add_argument('--states', type=Path, required=True)
    attempt05_preflight.add_argument('--schedule', type=Path, required=True)
    attempt05_preflight.add_argument('--split-assignment', type=Path, required=True)
    attempt05_preflight.add_argument('--calibration-schedule', type=Path, required=True)
    attempt05_preflight.add_argument('--input-closure', type=Path, required=True)
    attempt05_preflight.add_argument('--tooling-commit', required=True)
    attempt05_preflight.add_argument('--tooling-tree', required=True)
    attempt05_preflight.add_argument('--resume', action='store_true')

    attempt05_run = subparsers.add_parser('v4-attempt05-run')
    attempt05_run.add_argument('--authorization', type=Path, required=True)
    attempt05_run.add_argument('--input-closure-dir', type=Path, required=True)
    attempt05_run.add_argument('--tooling-commit', required=True)
    attempt05_run.add_argument('--tooling-tree', required=True)
    attempt05_run.add_argument('--resume', action='store_true')

    attempt05_status = subparsers.add_parser('v4-attempt05-status')
    attempt05_status.add_argument('--authorization', type=Path, required=True)
    attempt05_status.add_argument('--schedule', type=Path, required=True)

    attempt05_finalize = subparsers.add_parser('v4-attempt05-finalize')
    attempt05_finalize.add_argument('--authorization', type=Path, required=True)
    attempt05_finalize.add_argument('--states', type=Path, required=True)
    attempt05_finalize.add_argument('--schedule', type=Path, required=True)
    attempt05_finalize.add_argument('--split-assignment', type=Path, required=True)
    attempt05_finalize.add_argument('--native-warning-calibrations', type=Path, required=True)
    attempt05_finalize.add_argument('--record-path', type=Path, action='append', required=True)

    materialize_rectified.add_argument('--output-dir', type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] in {'preflight-real', 'run-georeliab'}:
        from .runner import cli_main as runner_cli_main

        return runner_cli_main(raw_argv)
    command_hint = raw_argv[0] if raw_argv else None
    rectified_commands = {
        'v4-rectified-closure',
        'v4-validate-rectified-closure',
        'v4-materialize-missing-rectified-members',
    }
    try:
        if (
            command_hint in ATTEMPT02_COMMANDS
            or command_hint in ATTEMPT03_COMMANDS
            or command_hint in ATTEMPT04_COMMANDS
            or command_hint in OVERLAY_RESOLUTION_COMMANDS
            or command_hint in ATTEMPT05_COMMANDS
            or command_hint in rectified_commands
        ):
            with redirect_stderr(StringIO()):
                args = build_parser().parse_args(raw_argv)
        else:
            args = build_parser().parse_args(raw_argv)
    except SystemExit as exc:
        if command_hint in ATTEMPT02_COMMANDS and exc.code != 0:
            print(
                json.dumps(
                    {
                        'status': 'FAIL',
                        'reason_code': 'V4_ATTEMPT02_CLI_ARGUMENT_ERROR',
                        'error_type': 'ArgumentParserError',
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        if command_hint in ATTEMPT03_COMMANDS and exc.code != 0:
            print(
                json.dumps(
                    {
                        'status': 'FAIL',
                        'reason_code': 'V4_ATTEMPT03_CLI_ARGUMENT_ERROR',
                        'error_type': 'ArgumentParserError',
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        if command_hint in ATTEMPT04_COMMANDS and exc.code != 0:
            print(
                json.dumps(
                    {
                        'status': 'FAIL',
                        'reason_code': 'V4_ATTEMPT04_CLI_ARGUMENT_ERROR',
                        'attempt_id': 'attempt-04',
                        'scientific_result': 'NO_SCIENTIFIC_RESULT',
                        'error_type': 'ArgumentParserError',
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        if command_hint in OVERLAY_RESOLUTION_COMMANDS and exc.code != 0:
            print(
                json.dumps(
                    {
                        'status': 'FAIL',
                        'reason_code': 'V4_OVERLAY_CLI_ARGUMENT_ERROR',
                        'scientific_result': 'NO_SCIENTIFIC_RESULT',
                        'error_type': 'ArgumentParserError',
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        if command_hint in ATTEMPT05_COMMANDS and exc.code != 0:
            print(
                json.dumps(
                    {
                        'status': 'FAIL',
                        'reason_code': 'V4_ATTEMPT05_CLI_ARGUMENT_ERROR',
                        'attempt_id': 'attempt-05',
                        'scientific_result': 'NO_SCIENTIFIC_RESULT',
                        'error_type': 'ArgumentParserError',
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        if command_hint in rectified_commands and exc.code != 0:
            reason_code = 'V4_RECTIFIED_CLI_ARGUMENT_ERROR'
            if '--expected-set' not in raw_argv:
                reason_code = 'V4_RECTIFIED_EXPECTED_SET_REQUIRED'
            print(
                json.dumps(
                    {
                        'status': 'FAIL',
                        'reason_code': reason_code,
                        'error_type': 'ArgumentParserError',
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        raise
    try:
        if args.command == 'v4-attempt02-prepare-resources':
            payload = materialize_attempt_resources(
                root=args.root,
                source_manifest_path=args.source_manifest,
                output_path=args.output,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload.get('status') == 'PASS' else 2
        if args.command == 'v4-attempt02-gpu-preflight':
            source_root = Path(__file__).resolve().parents[1]
            history = _attempt02_required_history(
                args.historical_evidence, source_root=source_root
            )
            resources = validate_attempt_resources(args.resource_snapshot)
            payload = create_attempt_hardware_preflight(
                output_path=args.output,
                schedule_sha256=str(resources.get('schedule_sha256')),
                historical_evidence_paths=history,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload.get('status') == 'PASS' else 2
        if args.command == 'v4-attempt02-create-execution-authorization':
            payload = create_attempt_execution_authorization(
                root=args.root,
                receipt_path=args.receipt,
                resource_snapshot_path=args.resource_snapshot,
                run_root=args.run_root,
                artifact_root=args.artifact_root,
                final_evidence_path=args.final_evidence_path,
                output_path=args.output,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload.get('status') == 'PASS' else 2
        if args.command == 'v4-attempt02-validate-execution-authorization':
            payload = validate_attempt_execution_authorization(
                args.authorization
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == 'v4-attempt03-revalidate-resources':
            payload = revalidate_attempt03_resources(
                worktree=args.worktree,
                runtime_root=args.runtime_root,
                rectified_root=args.rectified_root,
                closure_root=args.closure_root,
                overlay_path=args.overlay,
                output_path=args.output,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload.get('status') == 'PASS' else 2
        if args.command == 'v4-attempt03-gpu-preflight':
            payload = create_attempt03_gpu_preflight(
                worktree=args.worktree,
                resource_receipt_path=args.resource_receipt,
                output_path=args.output,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload.get('status') == 'PASS' else 2
        if args.command == (
            'v4-attempt03-create-execution-authorization'
        ):
            payload = create_attempt03_execution_authorization(
                worktree=args.worktree,
                runtime_root=args.runtime_root,
                resource_receipt_path=args.resource_receipt,
                hardware_snapshot_path=args.hardware_snapshot,
                receipt_path=args.receipt,
                authorization_path=args.authorization,
                run_root=args.run_root,
                artifact_root=args.artifact_root,
                gpu_ledger_path=args.gpu_ledger,
                final_evidence_path=args.final_evidence_path,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == (
            'v4-attempt03-validate-execution-authorization'
        ):
            payload = validate_attempt03_execution_authorization(
                args.authorization
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == 'v4-attempt04-revalidate-resources':
            payload = revalidate_attempt04_resources(
                worktree=args.worktree,
                runtime_root=args.runtime_root,
                rectified_root=args.rectified_root,
                closure_root=args.closure_root,
                overlay_path=args.overlay,
                output_path=args.output,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload.get('status') == 'PASS' else 2
        if args.command == 'v4-attempt04-gpu-preflight':
            payload = create_attempt04_gpu_preflight(
                worktree=args.worktree,
                resource_receipt_path=args.resource_receipt,
                output_path=args.output,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload.get('status') == 'PASS' else 2
        if args.command == (
            'v4-attempt04-create-execution-authorization'
        ):
            payload = create_attempt04_execution_authorization(
                worktree=args.worktree,
                runtime_root=args.runtime_root,
                resource_receipt_path=args.resource_receipt,
                hardware_snapshot_path=args.hardware_snapshot,
                receipt_path=args.receipt,
                authorization_path=args.authorization,
                run_root=args.run_root,
                artifact_root=args.artifact_root,
                gpu_ledger_path=args.gpu_ledger,
                final_evidence_path=args.final_evidence_path,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == (
            'v4-attempt04-validate-execution-authorization'
        ):
            payload = validate_attempt04_execution_authorization(
                args.authorization
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == 'v4-resolve-overlay-resources':
            payload = resolve_overlay_resources(
                worktree=args.worktree,
                runtime_root=args.runtime_root,
                overlay_path=args.overlay,
                frozen_materialization_path=args.frozen_materialization,
                output_dir=args.output_dir,
                resource_overrides=parse_resource_overrides(
                    args.resource_override
                ),
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == 'v4-validate-overlay-resource-resolution':
            payload = validate_overlay_resource_resolution(args.receipt)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == 'v4-prepare-rectified-resource-schedule':
            payload = prepare_rectified_resource_schedule(
                protocol_path=args.protocol,
                split_view_manifest_path=args.split_view_manifest,
                output_dir=args.output_dir,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == 'v4-rectified-closure':
            payload = create_rectified_member_closure(
                root=args.root,
                output_dir=args.output_dir,
                expected_set_path=args.expected_set,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == 'v4-validate-rectified-closure':
            payload = validate_rectified_member_closure(
                root=args.root,
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                expected_set_path=args.expected_set,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == 'v4-materialize-missing-rectified-members':
            payload = materialize_missing_rectified_members(
                root=args.root,
                official_rectified_archive=args.official_rectified_archive,
                output_dir=args.output_dir,
                expected_set_path=args.expected_set,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
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
        if args.command == 'v4-gpu-preflight':
            schedule_sha = _sha256_file(args.schedule) if args.schedule is not None else None
            payload = create_hardware_preflight(
                output_path=args.output,
                requested_physical_index=args.requested_index,
                project_commit=args.project_commit,
                project_tree=args.project_tree,
                schedule_sha256=schedule_sha,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload.get('status') == 'PASS' else 2
        if args.command == 'v4-create-execution-authorization':
            payload = create_execution_authorization(
                root=args.root,
                receipt_path=args.receipt,
                resource_inventory_path=args.resource_inventory,
                run_root=args.run_root,
                artifact_root=args.artifact_root,
                final_evidence_path=args.final_evidence_path,
                output_path=args.output,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == 'v4-validate-execution-authorization':
            payload = validate_execution_authorization(args.authorization)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == 'v4-attempt05-prepare-inputs':
            context = load_attempt05_authorized_context(
                authorization_path=args.authorization
            )
            closure = prepare_attempt05_inputs(
                context=context,
                dtu_root=args.dtu_root,
                rectified_closure_manifest=args.rectified_closure_manifest,
                fog_root=args.fog_root,
                output_dir=args.output_dir,
                source_root=Path(__file__).resolve().parents[1],
            )
            payload = {
                'status': closure.status,
                'attempt_id': 'attempt-05',
                'run_name': ATTEMPT05_RUN_NAME,
                'scientific_result': 'NO_SCIENTIFIC_RESULT',
                'attempt04_authorization_sha256': context.authorization_sha256,
                'input_closure_path': str(closure.manifest_path),
                'input_closure_sha256': closure.manifest_sha256,
                'split_assignment_path': str(closure.split_assignment_path),
                'split_assignment_sha256': closure.split_assignment_sha256,
                'state_inventory_path': str(closure.state_inventory_path),
                'state_inventory_sha256': closure.state_inventory_sha256,
                'scientific_schedule_path': str(closure.scientific_schedule_path),
                'scientific_schedule_sha256': closure.scientific_schedule_sha256,
                'calibration_schedule_path': str(closure.calibration_schedule_path),
                'calibration_schedule_sha256': closure.calibration_schedule_sha256,
                'runtime_binding_path': str(closure.runtime_binding_path),
                'runtime_binding_sha256': closure.runtime_binding_sha256,
                'model_independent_states': len(closure.model_independent_states),
                'scientific_units': len(closure.scientific_schedule.units),
                'calibration_l3_units': len(closure.calibration_schedule),
                'max_model_execution_units': 440,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == 'v4-attempt05-preflight':
            context = load_attempt05_authorized_context(
                authorization_path=args.authorization
            )
            states = _attempt05_states(args.states)
            schedule = _attempt05_schedule(args.schedule)
            rebuilt = build_attempt05_scientific_schedule(states)
            if rebuilt.schedule_sha256 != schedule.schedule_sha256:
                raise V4ExecutionError('V4_ATTEMPT05_SCHEDULE_STATE_MISMATCH')
            split_assignment = _attempt05_split(args.split_assignment)
            calibration_schedule = _attempt05_calibration_schedule(
                args.calibration_schedule
            )
            input_closure = validate_attempt05_input_closure(args.input_closure)
            input_storage = dict(input_closure["budgeted_input_storage"])
            input_storage["input_closure_sha256"] = _sha256_file(
                args.input_closure
            )
            preflight = _attempt05_get_or_create_preflight(
                context,
                authorization_path=args.authorization,
                tooling_commit=args.tooling_commit,
                tooling_tree=args.tooling_tree,
            )
            receipt = create_attempt05_start_receipt(
                authorization_path=args.authorization,
                schedule=schedule,
                model_independent_states=states,
                split_assignment=split_assignment,
                calibration_schedule=calibration_schedule,
                resume=args.resume,
                attempt05_tooling_commit=args.tooling_commit,
                attempt05_tooling_tree=args.tooling_tree,
                input_storage=input_storage,
            )
            payload = {
                'status': 'MVE_RUN_STARTED',
                'attempt_id': 'attempt-05',
                'scientific_result': 'NO_SCIENTIFIC_RESULT',
                'execution_preflight_path': str(preflight['preflight_path']),
                'execution_preflight_sha256': preflight['preflight_file_sha256'],
                'start_receipt_sha256': receipt['start_receipt_sha256'],
                'runtime_paths': receipt['runtime_paths'],
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == 'v4-attempt05-run':
            input_dir = args.input_closure_dir
            states = _attempt05_states(input_dir / 'v4-model-independent-states.json')
            schedule = _attempt05_schedule(input_dir / 'v4-scientific-schedule-400.json')
            split_assignment = _attempt05_split(input_dir / 'v4-split-assignment.json')
            calibration_schedule = _attempt05_calibration_schedule(
                input_dir / 'v4-calibration-l3-schedule-40.json'
            )
            input_closure_path = input_dir / 'v4-attempt05-input-closure.json'
            input_closure = validate_attempt05_input_closure(input_closure_path)
            input_storage = dict(input_closure["budgeted_input_storage"])
            input_storage["input_closure_sha256"] = _sha256_file(
                input_closure_path
            )
            decision = authorize_attempt05_next_dispatch(
                authorization_path=args.authorization,
                schedule=schedule,
                resume=args.resume,
            )
            if decision.status != 'PASS':
                payload = {
                    'status': f'V4_MVE_FAILED_WITH_REASON={decision.reason_code}',
                    'attempt_id': 'attempt-05',
                    'scientific_result': 'NO_SCIENTIFIC_RESULT',
                    'dispatch_gate_status': decision.status,
                    'dispatch_gate_reason_code': decision.reason_code,
                    'retry_count': 0,
                }
                print(json.dumps(payload, indent=2, sort_keys=True))
                return 2
            context = load_attempt05_authorized_context(
                authorization_path=args.authorization
            )
            binding_set = construct_attempt05_runtime_bindings(
                runtime_binding_path=input_dir / 'v4-runtime-state-bindings.json',
                model_independent_states=states,
                scientific_schedule=schedule,
                calibration_schedule=calibration_schedule,
                split_assignment=split_assignment,
                context=context,
                overlay_config_path=_attempt05_overlay_config_path(context),
                attempt05_tooling_commit=args.tooling_commit,
                attempt05_tooling_tree=args.tooling_tree,
                resume=args.resume,
            )
            calibration_executor, scientific_executor = make_attempt05_runtime_executors(
                calibration_bindings=binding_set.calibration_bindings,
                scientific_bindings=binding_set.scientific_bindings,
            )
            try:
                result = run_attempt05_pipeline(
                    authorization_path=args.authorization,
                    scientific_schedule=schedule,
                    model_independent_states=states,
                    split_assignment=split_assignment,
                    calibration_schedule=calibration_schedule,
                    attempt05_tooling_commit=args.tooling_commit,
                    attempt05_tooling_tree=args.tooling_tree,
                    ledger_path=context.gpu_ledger_path,
                    calibration_executor=calibration_executor,
                    scientific_executor=scientific_executor,
                    resume=True,
                    input_storage=input_storage,
                )
            finally:
                provider = binding_set.adapter_provider
                if provider is not None:
                    provider.close()
            payload = {
                'status': result.status,
                'attempt_id': 'attempt-05',
                'scientific_result': 'SCIENTIFIC_RESULT_AVAILABLE'
                if result.status == 'V4_MVE_COMPLETED'
                else 'NO_SCIENTIFIC_RESULT',
                'calibration_units_completed': result.calibration_units_completed,
                'scientific_units_completed': result.scientific_units_completed,
                'invalid_scientific_units': result.invalid_scientific_units,
                'gpu_inference_seconds': result.gpu_inference_seconds,
                'wall_runtime_seconds': result.wall_runtime_seconds,
                'logical_bytes': result.logical_bytes,
                'allocated_bytes': result.allocated_bytes,
                'peak_memory_mb': result.peak_memory_mb,
                'retry_count': 0,
                'finalizer_result': result.finalizer_result,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if result.status == 'V4_MVE_COMPLETED' else 2
        if args.command == 'v4-attempt05-status':
            context = load_attempt05_authorized_context(
                authorization_path=args.authorization
            )
            schedule = _attempt05_schedule(args.schedule)
            decision = authorize_attempt05_next_dispatch(
                authorization_path=args.authorization,
                schedule=schedule,
                resume=True,
            )
            totals = rehydrate_attempt05_ledger_totals(context.gpu_ledger_path)
            resource = evaluate_attempt05_resource_gate(
                gpu_inference_seconds=totals.gpu_inference_seconds,
                wall_runtime_seconds=totals.wall_runtime_seconds,
                new_logical_bytes=totals.logical_bytes,
                new_allocated_bytes=totals.allocated_bytes,
            )
            gpu_active_hours = totals.gpu_inference_seconds / 3600.0
            qualification_monitor = build_hourly_monitor_status(
                stage="ATTEMPT05_FROZEN",
                stage_completed=0,
                stage_total=1,
                attempt06_valid_completed=0,
                attempt06_elapsed_seconds=0.0,
                cumulative_materialization_elapsed_seconds=totals.wall_runtime_seconds,
                cumulative_gpu_active_hours=gpu_active_hours,
                cumulative_card_hours=0.0,
                cumulative_storage_bytes=totals.logical_bytes,
                invalid_count=totals.invalid_units,
                duplicate_count=0,
                identity_mismatch_count=0,
            )
            payload = {
                'status': decision.status,
                'reason_code': decision.reason_code,
                'attempt_id': 'attempt-05',
                'scientific_result': 'NO_SCIENTIFIC_RESULT',
                'run_root': str(context.run_root),
                'artifact_root': str(context.artifact_root),
                'resource_gate': {'status': resource.status, 'reason_code': resource.reason_code},
                'qualification_monitor': qualification_monitor,
                'ledger': {
                    'gpu_inference_seconds': totals.gpu_inference_seconds,
                    'wall_runtime_seconds': totals.wall_runtime_seconds,
                    'logical_bytes': totals.logical_bytes,
                    'allocated_bytes': totals.allocated_bytes,
                    'completed_units': totals.completed_units,
                    'calibration_units_completed': totals.calibration_units_completed,
                    'scientific_units_completed': totals.scientific_units_completed,
                    'invalid_units': totals.invalid_units,
                    'failed_units': totals.failed_units,
                    'retry_count': 0,
                    'peak_memory_mb': totals.peak_memory_mb,
                    'run_started': totals.run_started,
                    'finalized': totals.finalized,
                },
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if decision.status != 'FAIL' else 2
        if args.command == 'v4-attempt05-finalize':
            result = finalize_attempt05_scientific_bundle(
                authorization_path=args.authorization,
                record_paths=tuple(args.record_path),
                scientific_schedule=_attempt05_schedule(args.schedule),
                model_independent_states=_attempt05_states(args.states),
                split_assignment=_attempt05_split(args.split_assignment),
                native_warning_calibrations=_attempt05_calibrations(
                    args.native_warning_calibrations
                ),
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        source_root = Path(__file__).resolve().parents[1]
        if args.command == 'archive-superseded':
            payload = archive_superseded_results(
                args.root,
                expected_commit=args.expected_project_commit,
                expected_p1_items=args.expected_p1_items,
                expected_p2_items=args.expected_p2_items,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload.get('status') == 'PASS' else 2
        if args.command == 'prepare-p2a':
            payload = create_p2a_selection_manifest(
                args.root,
                source_root=source_root,
                storage_before_path=args.storage_before,
                storage_plan_path=args.storage_plan,
                output_path=args.output,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == 'validate-p2a':
            payload = evaluate_p2a_completion(
                args.root,
                selection_path=args.selection,
                source_root=source_root,
                output_path=args.output,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload.get('status') == 'PASS' else 2
        if args.command == 'record-gpu-selection':
            payload = record_gpu_selection(
                args.root,
                source_root=source_root,
                project_commit=args.project_commit,
                device=args.device,
                explicit_user_selection=args.explicit_user_selection,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == 'validate-gpu-selection':
            payload = validate_gpu_selection(
                args.root,
                source_root=source_root,
                project_commit=args.project_commit,
                device=args.device,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == 'validate-storage-refactor':
            payload = validate_storage_refactor_rollout(
                args.root,
                source_root=source_root,
                output_path=args.output,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload.get('status') == 'PASS' else 2
        if args.command == 'storage-audit':
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
    except (KeyError, TypeError, ValueError, AuditError, StorageAuditError, ExecutionGovernanceError, V4ExecutionError, V4RectifiedClosureError, OverlayResolutionError, OSError, json.JSONDecodeError) as exc:
        if args.command in OVERLAY_RESOLUTION_COMMANDS:
            print(
                json.dumps(
                    {
                        'status': 'FAIL',
                        'reason_code': str(exc) or type(exc).__name__,
                        'scientific_result': 'NO_SCIENTIFIC_RESULT',
                        'error_type': type(exc).__name__,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        if args.command in ATTEMPT05_COMMANDS:
            print(
                json.dumps(
                    {
                        'status': 'FAIL',
                        'reason_code': str(exc) or type(exc).__name__,
                        'attempt_id': 'attempt-05',
                        'scientific_result': 'NO_SCIENTIFIC_RESULT',
                        'error_type': type(exc).__name__,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        if args.command in ATTEMPT04_COMMANDS:
            print(
                json.dumps(
                    {
                        'status': 'FAIL',
                        'reason_code': str(exc) or type(exc).__name__,
                        'attempt_id': 'attempt-04',
                        'scientific_result': 'NO_SCIENTIFIC_RESULT',
                        'error_type': type(exc).__name__,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        if args.command in ATTEMPT03_COMMANDS:
            print(
                json.dumps(
                    {
                        'status': 'FAIL',
                        'reason_code': str(exc) or type(exc).__name__,
                        'attempt_id': 'attempt-03',
                        'scientific_result': 'NO_SCIENTIFIC_RESULT',
                        'error_type': type(exc).__name__,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        if args.command in ATTEMPT02_COMMANDS:
            print(
                json.dumps(
                    _attempt02_failure_payload(exc),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        if str(args.command).startswith('v4-rectified') or args.command in {
            'v4-prepare-rectified-resource-schedule',
            'v4-materialize-missing-rectified-members',
            'v4-validate-rectified-closure',
        }:
            print(
                json.dumps(
                    {
                        'status': 'FAIL',
                        'reason_code': str(exc) or type(exc).__name__,
                        'error_type': type(exc).__name__,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    raise AssertionError('unreachable command')



