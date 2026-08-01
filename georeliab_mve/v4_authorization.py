"""CPU-only GeoReliab v4 GPU preflight and execution authorization.

This module only inspects already-prepared inputs and hardware metadata. It does
not load model checkpoints, instantiate adapters, execute forwards, compute
scientific metrics, acquire the scientific execution lock, or touch the GPU
inference ledger.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import platform
import subprocess
import time
from typing import Any, Callable
from uuid import uuid4

from .v4_counterfactuals import (
    FOG_STATES,
    LIGHTING_STATES,
    ModelIndependentState,
    SCIENTIFIC_MODELS,
    SCIENTIFIC_STATES,
    TEST_SCENE_IDS,
    build_scientific_schedule,
    parse_scientific_schedule,
    validate_scientific_schedule,
)
from .v4_execution import (
    BYTE_CATASTROPHE,
    BYTE_TARGET,
    GPU_CATASTROPHE_SECONDS,
    GPU_TARGET_SECONDS,
    SCIENTIFIC_MVE,
    V4ExecutionError,
    V4ExecutionReceipt,
)
from .v4_science_lock import V4_PROTOCOL_ID, V4_PROTOCOL_SHA256


ATTEMPT_ID = 'attempt-02'
EXCLUDED_GPU_UUID = 'GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa'
ATTEMPT_SCHEMA_PREFIX = 'georeliab-v4-attempt-02'
ATTEMPT_INVENTORY_SCHEMA_VERSION = f'{ATTEMPT_SCHEMA_PREFIX}-gpu-inventory-1.0'
ATTEMPT_PREFLIGHT_SCHEMA_VERSION = f'{ATTEMPT_SCHEMA_PREFIX}-hardware-preflight-1.0'
ATTEMPT_RECEIPT_SCHEMA_VERSION = f'{ATTEMPT_SCHEMA_PREFIX}-gpu-receipt-1.0'
ATTEMPT_RESOURCE_SCHEMA_VERSION = f'{ATTEMPT_SCHEMA_PREFIX}-resources-1.0'
ATTEMPT_AUTHORIZATION_SCHEMA_VERSION = (
    f'{ATTEMPT_SCHEMA_PREFIX}-execution-authorization-1.0'
)
ATTEMPT_RESOURCE_KEYS = (
    'model_bindings',
    'dtu_archives',
    'v4_split',
    'state_inventory_200',
    'fog_manifest',
    'scientific_schedule_400',
    'environment_locks',
    'science_lock',
)


IMPLEMENTATION_ANCHOR_COMMIT = "7381e60050143a78fca6a3ebde5706ae27d2c145"
IMPLEMENTATION_ANCHOR_TREE = "f4e2b1104496c817693aaa5989d0276d2ebe03e9"
V4_HARDWARE_PREFLIGHT_SCHEMA_VERSION = "georeliab-v4-hardware-preflight-1.0"
V4_EXECUTION_AUTHORIZATION_SCHEMA_VERSION = "georeliab-v4-execution-authorization-1.0"
V4_PREFLIGHT_DECISION_SCHEMA_VERSION = "georeliab-v4-preflight-decision-1.0"
AUTHORIZED_GPU_MODEL = "NVIDIA A100 80GB PCIe"
MIN_FREE_MEMORY_BYTES = 16 * 1024 * 1024 * 1024
FORMAL_SAMPLE_INTERVAL_SECONDS = 5.0
AUTHORIZED_FINALIZER = "georeliab_mve.v4_execution:finalize_v4_scientific_bundle"
AUTHORIZED_RESOURCE_KEYS = (
    "raw_data",
    "model_weights_config",
    "upstream_commits_manifests",
    "split",
    "state_inventory_200",
    "fog_manifest",
    "scientific_schedule_400",
    "environment_locks",
)
PROCESS_PRESENT_REASON_CODES = frozenset(
    {
        "V4_GPU_NON_GEORELIAB_COMPUTE_PROCESS_PRESENT",
        "V4_GPU_GEORELIAB_RESIDUAL_COMPUTE_PROCESS_PRESENT",
        "V4_GPU_ACTIVE_COMPUTE_PROCESS_IDENTITY_INCOMPLETE",
        "V4_GPU_ACTIVE_COMPUTE_PROCESS_STATE_UNSTABLE",
    }
)
NO_ACTIVE_PROCESS_REASON_CODES = frozenset(
    {"V4_GPU_RECEIPT_NO_ACTIVE_COMPUTE_PROCESS"}
)
def _require_attempt_id(value: object) -> str:
    if value != ATTEMPT_ID:
        raise V4ExecutionError('V4_ATTEMPT_ID_MISMATCH')
    return ATTEMPT_ID


def _require_attempt_path(path: Path) -> Path:
    lowered = tuple(part.lower() for part in path.resolve().parts)
    if ATTEMPT_ID not in lowered or 'attempt-01' in lowered:
        raise V4ExecutionError('V4_ATTEMPT_PATH_MISMATCH')
    return path.resolve()


def _attempt_root_from_path(path: Path) -> Path:
    resolved = _require_attempt_path(path)
    lowered = tuple(part.lower() for part in resolved.parts)
    positions = [
        index for index, part in enumerate(lowered) if part == ATTEMPT_ID
    ]
    if len(positions) != 1:
        raise V4ExecutionError('V4_ATTEMPT_PATH_MISMATCH')
    return Path(*resolved.parts[: positions[0] + 1])


def _reject_attempt_collision(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.exists() or path.with_name(path.name + '.partial').exists():
            raise V4ExecutionError('V4_ATTEMPT_ARTIFACT_COLLISION')


def _publish_attempt_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    validator: Callable[[Mapping[str, Any]], None],
) -> None:
    _require_attempt_path(path)
    _reject_attempt_collision((path,))
    _atomic_json(path, payload, validator=validator)


def _recursive_strings(value: object) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, Mapping):
        result: set[str] = set()
        for key, item in value.items():
            result.update(_recursive_strings(key))
            result.update(_recursive_strings(item))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = set()
        for item in value:
            result.update(_recursive_strings(item))
        return result
    return set()


def _historical_values(paths: Sequence[Path]) -> set[str]:
    def evidence_identities(value: object) -> set[str]:
        identities: set[str] = set()
        if isinstance(value, Mapping):
            for key, item in value.items():
                label = str(key).lower()
                is_identity = (
                    label == 'run_id'
                    or 'timestamp' in label
                    or label in {
                        'hardware_preflight_sha256',
                        'preflight_decision_sha256',
                        'decision_sha256',
                        'receipt_sha256',
                        'receipt_payload_sha256',
                        'authorization_sha256',
                    }
                )
                if is_identity and isinstance(item, str):
                    identities.add(item)
                identities.update(evidence_identities(item))
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes)
        ):
            for item in value:
                identities.update(evidence_identities(item))
        return identities

    values: set[str] = set()
    for path in paths:
        payload = load_json(path)
        if payload.get('attempt_id') == ATTEMPT_ID:
            raise V4ExecutionError('V4_ATTEMPT_REUSE_FORBIDDEN')
        values.update(evidence_identities(payload))
    return values


def _reject_historical_reuse(
    payload: Mapping[str, Any], historical_values: set[str]
) -> None:
    current = _recursive_strings(payload)
    reused = current & historical_values
    if reused:
        raise V4ExecutionError('V4_ATTEMPT_HISTORY_REUSE_FORBIDDEN')
    if any('attempt-01' in value.lower() for value in current):
        raise V4ExecutionError('V4_ATTEMPT_CROSS_LINK_FORBIDDEN')


def _split_csv_rows(
    text: str, expected_columns: int, *, reason_code: str
) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        row = [item.strip() for item in raw.split(',', expected_columns - 1)]
        if len(row) != expected_columns:
            raise V4ExecutionError(reason_code)
        rows.append(row)
    if not rows:
        raise V4ExecutionError(reason_code)
    return rows


def _required_smi_int(value: str, *, reason_code: str) -> int:
    if value.strip().lower() in {
        '',
        'n/a',
        '[not supported]',
        'not supported',
        'unknown',
    }:
        raise V4ExecutionError(reason_code)
    try:
        return int(float(value.strip()))
    except ValueError as exc:
        raise V4ExecutionError(reason_code) from exc


def nvidia_smi_all_gpu_inventory_sample(
    *, command_runner: Callable[..., str] | None = None
) -> dict[str, object]:
    '''Collect one CPU/read-only inventory of every visible physical GPU.'''

    runner = _run_text_command if command_runner is None else command_runner
    try:
        rows = _split_csv_rows(
            runner(
                (
                    'nvidia-smi',
                    '--query-gpu=index,uuid,name,memory.total,memory.free,'
                    'memory.used,utilization.gpu,temperature.gpu,'
                    'mig.mode.current,ecc.errors.uncorrected.volatile.total,'
                    'driver_version',
                    '--format=csv,noheader,nounits',
                )
            ),
            11,
            reason_code='V4_GPU_INVENTORY_UNAVAILABLE',
        )
        cuda_runtime = _nvidia_smi_cuda_runtime(runner)
        process_text = runner(
            (
                'nvidia-smi',
                '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                '--format=csv,noheader,nounits',
            )
        )
    except V4ExecutionError:
        raise
    except Exception as exc:
        raise V4ExecutionError('V4_GPU_INVENTORY_UNAVAILABLE') from exc

    processes_by_uuid: dict[str, list[dict[str, object]]] = {}
    for raw in process_text.splitlines():
        if not raw.strip():
            continue
        values = [item.strip() for item in raw.split(',', 3)]
        if len(values) != 4:
            raise V4ExecutionError('V4_GPU_PROCESS_ENUMERATION_UNPROVEN')
        gpu_uuid, pid_raw, process_name, used_raw = values
        pid = _required_smi_int(
            pid_raw, reason_code='V4_GPU_PROCESS_IDENTITY_UNPROVEN'
        )
        processes_by_uuid.setdefault(gpu_uuid, []).append(
            {
                'pid': pid,
                'owner': _process_owner(pid),
                'cwd': _process_cwd(pid),
                'cmdline': _process_cmdline(pid),
                'process_name': process_name,
                'used_memory_bytes': _required_smi_int(
                    used_raw,
                    reason_code='V4_GPU_PROCESS_MEMORY_UNPROVEN',
                )
                * 1024
                * 1024,
            }
        )

    devices: list[dict[str, object]] = []
    for row in rows:
        ecc_count = _required_smi_int(
            row[9], reason_code='V4_GPU_ECC_HEALTH_UNPROVEN'
        )
        device_uuid = row[1]
        devices.append(
            {
                'index': _required_smi_int(
                    row[0], reason_code='V4_GPU_INDEX_UNPROVEN'
                ),
                'uuid': device_uuid,
                'model': row[2],
                'total_memory_bytes': _required_smi_int(
                    row[3], reason_code='V4_GPU_MEMORY_UNPROVEN'
                )
                * 1024
                * 1024,
                'free_memory_bytes': _required_smi_int(
                    row[4], reason_code='V4_GPU_MEMORY_UNPROVEN'
                )
                * 1024
                * 1024,
                'used_memory_bytes': _required_smi_int(
                    row[5], reason_code='V4_GPU_MEMORY_UNPROVEN'
                )
                * 1024
                * 1024,
                'utilization_gpu_percent': _required_smi_int(
                    row[6], reason_code='V4_GPU_UTILIZATION_UNPROVEN'
                ),
                'temperature_c': _required_smi_int(
                    row[7], reason_code='V4_GPU_TEMPERATURE_UNPROVEN'
                ),
                'driver_version': row[10],
                'cuda_runtime': cuda_runtime,
                'mig_mode': row[8],
                'ecc_health': 'OK' if ecc_count == 0 else 'ERROR',
                'ecc_uncorrected_volatile_total': ecc_count,
                'compute_processes': processes_by_uuid.get(device_uuid, []),
            }
        )
    unknown_process_gpu = set(processes_by_uuid) - {
        str(device['uuid']) for device in devices
    }
    if unknown_process_gpu:
        raise V4ExecutionError('V4_GPU_PROCESS_DEVICE_IDENTITY_UNPROVEN')
    return {
        'schema_version': ATTEMPT_INVENTORY_SCHEMA_VERSION,
        'attempt_id': ATTEMPT_ID,
        'hostname': platform.node(),
        'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'driver_version': rows[0][10],
        'cuda_runtime': cuda_runtime,
        'devices': devices,
    }


def _validate_attempt_inventory(sample: Mapping[str, object]) -> None:
    _require_attempt_id(sample.get('attempt_id'))
    if sample.get('schema_version') != ATTEMPT_INVENTORY_SCHEMA_VERSION:
        raise V4ExecutionError('V4_GPU_INVENTORY_SCHEMA_REQUIRED')
    if not _runtime_proven(sample.get('hostname')):
        raise V4ExecutionError('V4_GPU_HOST_IDENTITY_UNPROVEN')
    if not _runtime_proven(sample.get('timestamp_utc')):
        raise V4ExecutionError('V4_GPU_TIMESTAMP_UNPROVEN')
    if not _driver_proven(sample.get('driver_version')):
        raise V4ExecutionError('V4_GPU_DRIVER_VERSION_UNPROVEN')
    if not _runtime_proven(sample.get('cuda_runtime')):
        raise V4ExecutionError('V4_GPU_CUDA_RUNTIME_UNPROVEN')
    devices = sample.get('devices')
    if not isinstance(devices, list):
        raise V4ExecutionError('V4_GPU_INVENTORY_SCHEMA_REQUIRED')
    seen_uuid: set[str] = set()
    seen_index: set[int] = set()
    for device in devices:
        if not isinstance(device, Mapping):
            raise V4ExecutionError('V4_GPU_INVENTORY_SCHEMA_REQUIRED')
        uuid = device.get('uuid')
        index = device.get('index')
        if (
            not isinstance(uuid, str)
            or not uuid.startswith('GPU-')
            or type(index) is not int
            or index < 0
            or uuid in seen_uuid
            or index in seen_index
        ):
            raise V4ExecutionError('V4_GPU_IDENTITY_UNPROVEN')
        seen_uuid.add(uuid)
        seen_index.add(index)
        for key in (
            'total_memory_bytes',
            'free_memory_bytes',
            'used_memory_bytes',
            'utilization_gpu_percent',
            'temperature_c',
            'ecc_uncorrected_volatile_total',
        ):
            if type(device.get(key)) is not int or int(device[key]) < 0:
                raise V4ExecutionError('V4_GPU_INVENTORY_SCHEMA_REQUIRED')
        if not _runtime_proven(device.get('model')):
            raise V4ExecutionError('V4_GPU_MODEL_UNPROVEN')
        if not _driver_proven(device.get('driver_version')):
            raise V4ExecutionError('V4_GPU_DRIVER_VERSION_UNPROVEN')
        if not _runtime_proven(device.get('cuda_runtime')):
            raise V4ExecutionError('V4_GPU_CUDA_RUNTIME_UNPROVEN')
        if not _runtime_proven(device.get('mig_mode')):
            raise V4ExecutionError('V4_GPU_MIG_STATE_UNPROVEN')
        if not _runtime_proven(device.get('ecc_health')):
            raise V4ExecutionError('V4_GPU_ECC_HEALTH_UNPROVEN')
        processes = device.get('compute_processes')
        if not isinstance(processes, list):
            raise V4ExecutionError('V4_GPU_PROCESS_ENUMERATION_UNPROVEN')


def _device_map(sample: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    _validate_attempt_inventory(sample)
    devices = sample['devices']
    assert isinstance(devices, list)
    return {str(device['uuid']): device for device in devices}


def _process_signature(device: Mapping[str, object]) -> tuple[tuple[object, ...], ...]:
    processes = device.get('compute_processes')
    if not isinstance(processes, list):
        raise V4ExecutionError('V4_GPU_PROCESS_ENUMERATION_UNPROVEN')
    rows: list[tuple[object, ...]] = []
    for process in processes:
        if not isinstance(process, Mapping):
            raise V4ExecutionError('V4_GPU_PROCESS_IDENTITY_UNPROVEN')
        pid = process.get('pid')
        owner = process.get('owner')
        cwd = process.get('cwd')
        cmdline = process.get('cmdline')
        memory = process.get('used_memory_bytes')
        if (
            type(pid) is not int
            or pid <= 0
            or not _runtime_proven(owner)
            or not _runtime_proven(cwd)
            or not _runtime_proven(cmdline)
            or type(memory) is not int
            or memory < 0
        ):
            raise V4ExecutionError('V4_GPU_PROCESS_IDENTITY_UNPROVEN')
        rows.append((pid, owner, cwd, cmdline, memory))
    return tuple(sorted(rows))


def _candidate_reason(
    first: Mapping[str, object] | None,
    second: Mapping[str, object] | None,
) -> str | None:
    if first is None or second is None:
        return 'V4_GPU_MAPPING_DRIFT'
    uuid = first.get('uuid')
    if uuid == EXCLUDED_GPU_UUID:
        return 'V4_GPU_UUID_EXCLUDED'
    if (
        second.get('uuid') != uuid
        or second.get('index') != first.get('index')
        or first.get('total_memory_bytes') != second.get('total_memory_bytes')
    ):
        return 'V4_GPU_MAPPING_DRIFT'
    try:
        first_processes = _process_signature(first)
        second_processes = _process_signature(second)
    except V4ExecutionError as exc:
        return str(exc)
    if first_processes != second_processes:
        return 'V4_GPU_PROCESS_STATE_UNSTABLE'
    if first_processes:
        return 'V4_GPU_COMPUTE_PROCESS_PRESENT'
    for device in (first, second):
        if device.get('model') != AUTHORIZED_GPU_MODEL:
            return 'V4_GPU_MODEL_NOT_AUTHORIZED'
        if str(device.get('mig_mode')).strip().lower() not in {
            'disabled',
            'off',
            '0',
        }:
            return 'V4_GPU_MIG_ENABLED'
        if device.get('utilization_gpu_percent') != 0:
            return 'V4_GPU_NOT_IDLE'
        if int(device.get('free_memory_bytes', -1)) < MIN_FREE_MEMORY_BYTES:
            return 'V4_GPU_FREE_MEMORY_INSUFFICIENT'
        if (
            str(device.get('ecc_health')).upper() not in {'OK', 'PASS', '0'}
            or device.get('ecc_uncorrected_volatile_total') != 0
        ):
            return 'V4_GPU_HEALTH_ERROR'
        if not _driver_proven(device.get('driver_version')):
            return 'V4_GPU_DRIVER_VERSION_UNPROVEN'
        if not _runtime_proven(device.get('cuda_runtime')):
            return 'V4_GPU_CUDA_RUNTIME_UNPROVEN'
    if first.get('driver_version') != second.get('driver_version'):
        return 'V4_GPU_DRIVER_VERSION_DRIFT'
    if first.get('cuda_runtime') != second.get('cuda_runtime'):
        return 'V4_GPU_CUDA_RUNTIME_DRIFT'
    return None


def select_attempt_gpu(
    samples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    '''Select once from two complete inventories; UUID remains authoritative.'''

    if len(samples) != 2:
        raise V4ExecutionError('V4_GPU_PREFLIGHT_TWO_SAMPLES_REQUIRED')
    first_map = _device_map(samples[0])
    second_map = _device_map(samples[1])
    if samples[0].get('hostname') != samples[1].get('hostname'):
        raise V4ExecutionError('V4_GPU_HOST_IDENTITY_DRIFT')
    if samples[0].get('timestamp_utc') == samples[1].get('timestamp_utc'):
        raise V4ExecutionError('V4_GPU_SAMPLE_INDEPENDENCE_UNPROVEN')

    evaluations: list[dict[str, object]] = []
    eligible: list[Mapping[str, object]] = []
    for uuid in sorted(set(first_map) | set(second_map)):
        first = first_map.get(uuid)
        second = second_map.get(uuid)
        reason = _candidate_reason(first, second)
        evaluation = {
            'uuid': uuid,
            'index_sample_1': None if first is None else first.get('index'),
            'index_sample_2': None if second is None else second.get('index'),
            'eligible': reason is None,
            'reason_code': reason or 'V4_GPU_CANDIDATE_ELIGIBLE',
        }
        evaluations.append(evaluation)
        if reason is None and second is not None:
            eligible.append(second)
    eligible.sort(
        key=lambda device: (
            -int(device['total_memory_bytes']),
            -int(device['free_memory_bytes']),
            str(device['uuid']),
        )
    )
    if not eligible:
        return {
            'status': 'FAIL',
            'reason_code': 'V4_NO_ELIGIBLE_IDLE_GPU',
            'candidate_evaluations': evaluations,
            'selected_gpu': None,
        }
    selected = dict(eligible[0])
    return {
        'status': 'PASS',
        'reason_code': 'V4_GPU_SELECTED_BY_STABLE_IDENTITY',
        'candidate_evaluations': evaluations,
        'selected_gpu': selected,
    }


def _default_attempt_torch_probe(
    model_id: str,
    selected_uuid: str,
    selected_index: int,
    expected: Mapping[str, object],
) -> dict[str, object]:
    code = (
        'import json,torch;'
        'ok=torch.cuda.is_available() and torch.cuda.device_count()==1;'
        'p=torch.cuda.get_device_properties(0) if ok else None;'
        'u=getattr(p,\'uuid\',None) if p else None;'
        'u=u.decode(\'utf-8\') if isinstance(u,bytes) else u;'
        'print(json.dumps(dict('
        'model_instantiated=False,checkpoint_loaded=False,'
        'forward_executed=False,torch_cuda_available=torch.cuda.is_available(),'
        'torch_device_count=torch.cuda.device_count(),'
        'torch_current_device=torch.cuda.current_device() if ok else None,'
        'mapped_device_uuid=str(u) if u else None,'
        'mapped_device_model=p.name if p else None,'
        'mapped_total_memory_bytes=p.total_memory if p else None)))'
    )
    python = _frozen_python_for_model(model_id)
    payload = json.loads(
        _run_text_command(
            (python, '-c', code), env={'CUDA_VISIBLE_DEVICES': selected_uuid}
        )
    )
    post = nvidia_smi_hardware_sample(selected_index)
    return {
        'model_id': model_id,
        **payload,
        'post_probe_mig_mode': post.get('mig_mode'),
        'post_probe_ecc_health': post.get('ecc_health'),
        'residual_compute_process_count': len(post.get('compute_processes', ())),
        'expected_uuid': expected.get('uuid'),
    }


def _evaluate_attempt_probes(
    probes: Sequence[Mapping[str, object]], selected: Mapping[str, object]
) -> dict[str, object]:
    if len(probes) != len(SCIENTIFIC_MODELS):
        return _fail('V4_GPU_TORCH_PROBE_MODEL_SET_MISMATCH')
    seen: set[str] = set()
    for probe in probes:
        model = probe.get('model_id')
        if model not in SCIENTIFIC_MODELS or str(model) in seen:
            return _fail('V4_GPU_TORCH_PROBE_MODEL_SET_MISMATCH')
        seen.add(str(model))
        if (
            probe.get('model_instantiated') is not False
            or probe.get('checkpoint_loaded') is not False
            or probe.get('forward_executed') is not False
        ):
            return _fail('V4_GPU_TORCH_PROBE_SCOPE_VIOLATION')
        if (
            probe.get('torch_cuda_available') is not True
            or probe.get('torch_device_count') != 1
            or probe.get('torch_current_device') != 0
        ):
            return _fail('V4_GPU_TORCH_PROBE_VISIBLE_DEVICE_MISMATCH')
        if (
            probe.get('mapped_device_uuid') != selected.get('uuid')
            or probe.get('mapped_device_model') != selected.get('model')
            or probe.get('mapped_total_memory_bytes')
            != selected.get('total_memory_bytes')
        ):
            return _fail('V4_GPU_TORCH_PROBE_DEVICE_MISMATCH')
        if (
            str(probe.get('post_probe_mig_mode')).lower() != 'disabled'
            or probe.get('post_probe_ecc_health') != 'OK'
        ):
            return _fail('V4_GPU_TORCH_PROBE_POST_HEALTH_FAILED')
        if probe.get('residual_compute_process_count') != 0:
            return _fail('V4_GPU_TORCH_PROBE_RESIDUAL_PROCESS')
    return {'status': 'PASS', 'reason_code': 'V4_GPU_MAPPING_PROBES_PASS'}


def _sign_attempt_payload(
    payload: Mapping[str, Any], field: str
) -> dict[str, object]:
    signed = dict(payload)
    signed[field] = _sha_json(payload)
    return signed


def _validate_attempt_signature(
    payload: Mapping[str, Any], field: str, reason_code: str
) -> None:
    expected = payload.get(field)
    unsigned = {key: value for key, value in payload.items() if key != field}
    if not _is_sha(expected) or _sha_json(unsigned) != expected:
        raise V4ExecutionError(reason_code)


def _validate_attempt_preflight_payload(payload: Mapping[str, Any]) -> None:
    _require_attempt_id(payload.get('attempt_id'))
    if payload.get('schema_version') != ATTEMPT_PREFLIGHT_SCHEMA_VERSION:
        raise V4ExecutionError('V4_ATTEMPT_PREFLIGHT_SCHEMA_REQUIRED')
    if (
        payload.get('implementation_commit') != IMPLEMENTATION_ANCHOR_COMMIT
        or payload.get('implementation_tree') != IMPLEMENTATION_ANCHOR_TREE
    ):
        raise V4ExecutionError('V4_ATTEMPT_PREFLIGHT_ANCHOR_MISMATCH')
    if payload.get('sample_interval_seconds') != FORMAL_SAMPLE_INTERVAL_SECONDS:
        raise V4ExecutionError('V4_GPU_SAMPLE_INTERVAL_MUST_BE_5_SECONDS')
    if payload.get('status') not in {'PASS', 'FAIL'}:
        raise V4ExecutionError('V4_ATTEMPT_PREFLIGHT_SCHEMA_REQUIRED')
    if not _is_sha(payload.get('run_id'), length=32):
        raise V4ExecutionError('V4_ATTEMPT_RUN_ID_INVALID')
    samples = payload.get('inventory_samples')
    if not isinstance(samples, list):
        raise V4ExecutionError('V4_ATTEMPT_PREFLIGHT_SCHEMA_REQUIRED')
    if len(samples) == 2:
        selection = select_attempt_gpu(samples)
        if selection.get('selected_gpu') != payload.get('selected_gpu'):
            raise V4ExecutionError('V4_GPU_SELECTION_TAMPER')
        if selection.get('candidate_evaluations') != payload.get(
            'candidate_evaluations'
        ):
            raise V4ExecutionError('V4_GPU_SELECTION_TAMPER')
        if selection['status'] == 'FAIL':
            if (
                payload.get('status') != 'FAIL'
                or payload.get('reason_code') != 'V4_NO_ELIGIBLE_IDLE_GPU'
            ):
                raise V4ExecutionError('V4_GPU_SELECTION_TAMPER')
        elif payload.get('status') == 'PASS':
            selected = selection.get('selected_gpu')
            assert isinstance(selected, Mapping)
            probe_decision = _evaluate_attempt_probes(
                payload.get('model_environment_probes', ()), selected
            )
            if probe_decision['status'] != 'PASS':
                raise V4ExecutionError(str(probe_decision['reason_code']))
            if payload.get('reason_code') != 'V4_ATTEMPT_PREFLIGHT_PASS':
                raise V4ExecutionError('V4_GPU_SELECTION_TAMPER')
        elif payload.get('selected_gpu') is not None:
            probes = payload.get('model_environment_probes', ())
            if not isinstance(probes, list):
                raise V4ExecutionError('V4_ATTEMPT_PREFLIGHT_SCHEMA_REQUIRED')
    elif payload.get('status') != 'FAIL' or payload.get('selected_gpu') is not None:
        raise V4ExecutionError('V4_ATTEMPT_PREFLIGHT_SCHEMA_REQUIRED')


def _validate_attempt_decision_payload(payload: Mapping[str, Any]) -> None:
    _require_attempt_id(payload.get('attempt_id'))
    if payload.get('schema_version') != (
        f'{ATTEMPT_SCHEMA_PREFIX}-preflight-decision-1.0'
    ):
        raise V4ExecutionError('V4_ATTEMPT_DECISION_SCHEMA_REQUIRED')
    _validate_attempt_signature(
        payload, 'decision_sha256', 'V4_ATTEMPT_DECISION_TAMPER'
    )
    snapshot_path = Path(str(payload.get('hardware_preflight_path')))
    _require_attempt_path(snapshot_path)
    if (
        not snapshot_path.is_file()
        or sha256_file(snapshot_path) != payload.get('hardware_preflight_sha256')
    ):
        raise V4ExecutionError('V4_ATTEMPT_PREFLIGHT_TAMPER')
    snapshot = load_json(snapshot_path)
    _validate_attempt_preflight_payload(snapshot)
    for key in ('run_id', 'status', 'reason_code', 'selected_gpu'):
        if payload.get(key) != snapshot.get(key):
            raise V4ExecutionError('V4_ATTEMPT_DECISION_LINKAGE_MISMATCH')
    if payload.get('terminal_status') != snapshot.get('status'):
        raise V4ExecutionError('V4_ATTEMPT_DECISION_LINKAGE_MISMATCH')
    if payload.get('scientific_result') != 'NO_SCIENTIFIC_RESULT':
        raise V4ExecutionError('V4_ATTEMPT_DECISION_SCIENCE_FORBIDDEN')


def _validate_attempt_receipt_payload(payload: Mapping[str, Any]) -> None:
    _require_attempt_id(payload.get('attempt_id'))
    if payload.get('schema_version') != ATTEMPT_RECEIPT_SCHEMA_VERSION:
        raise V4ExecutionError('V4_ATTEMPT_RECEIPT_SCHEMA_REQUIRED')
    _validate_attempt_signature(
        payload, 'receipt_payload_sha256', 'V4_ATTEMPT_RECEIPT_TAMPER'
    )
    if (
        payload.get('implementation_commit') != IMPLEMENTATION_ANCHOR_COMMIT
        or payload.get('implementation_tree') != IMPLEMENTATION_ANCHOR_TREE
    ):
        raise V4ExecutionError('V4_AUTHORIZATION_STALE_ANCHOR')
    if (
        payload.get('protocol_id') != V4_PROTOCOL_ID
        or payload.get('protocol_sha256') != V4_PROTOCOL_SHA256
    ):
        raise V4ExecutionError('V4_GPU_RECEIPT_PROTOCOL_MISMATCH')
    if not _is_sha(payload.get('schedule_sha256')):
        raise V4ExecutionError('V4_AUTHORIZATION_SCHEDULE_REQUIRED')
    if (
        payload.get('max_concurrent_gpus') != 1
        or payload.get('sequential_model_execution') is not True
        or payload.get('sequential_unit_execution') is not True
        or payload.get('fallback_allowed') is not False
        or payload.get('device_switch_allowed') is not False
        or payload.get('retry_allowed') is not False
    ):
        raise V4ExecutionError('V4_GPU_RECEIPT_NO_FALLBACK_REQUIRED')
    snapshot_path = Path(str(payload.get('hardware_preflight_path')))
    decision_path = Path(str(payload.get('preflight_decision_path')))
    for path, digest, reason in (
        (
            snapshot_path,
            payload.get('hardware_preflight_sha256'),
            'V4_ATTEMPT_PREFLIGHT_TAMPER',
        ),
        (
            decision_path,
            payload.get('preflight_decision_sha256'),
            'V4_ATTEMPT_DECISION_TAMPER',
        ),
    ):
        _require_attempt_path(path)
        if not path.is_file() or sha256_file(path) != digest:
            raise V4ExecutionError(reason)
    snapshot = load_json(snapshot_path)
    decision = load_json(decision_path)
    _validate_attempt_preflight_payload(snapshot)
    _validate_attempt_decision_payload(decision)
    if snapshot.get('status') != 'PASS' or decision.get('status') != 'PASS':
        raise V4ExecutionError('V4_ATTEMPT_PASS_EVIDENCE_REQUIRED')
    selected = snapshot.get('selected_gpu')
    if not isinstance(selected, Mapping):
        raise V4ExecutionError('V4_ATTEMPT_PASS_EVIDENCE_REQUIRED')
    if (
        payload.get('run_id') != snapshot.get('run_id')
        or payload.get('selected_gpu_uuid') != selected.get('uuid')
        or payload.get('selected_physical_index') != selected.get('index')
        or payload.get('selected_gpu_model') != selected.get('model')
        or payload.get('selected_total_memory_bytes')
        != selected.get('total_memory_bytes')
    ):
        raise V4ExecutionError('V4_GPU_RECEIPT_DEVICE_MISMATCH')


def create_attempt_hardware_preflight(
    *,
    output_path: Path,
    schedule_sha256: str,
    inventory_sampler: Callable[[], Mapping[str, object]] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    sample_interval_seconds: float = FORMAL_SAMPLE_INTERVAL_SECONDS,
    probe_runner: Callable[
        [str, str, int, Mapping[str, object]], Mapping[str, object]
    ] = _default_attempt_torch_probe,
    historical_evidence_paths: Sequence[Path] = (),
) -> dict[str, object]:
    '''Materialize attempt-02 selection evidence without model/science work.'''

    _require_attempt_path(output_path)
    if sample_interval_seconds != FORMAL_SAMPLE_INTERVAL_SECONDS:
        raise V4ExecutionError('V4_GPU_SAMPLE_INTERVAL_MUST_BE_5_SECONDS')
    if not _is_sha(schedule_sha256):
        raise V4ExecutionError('V4_AUTHORIZATION_SCHEDULE_REQUIRED')
    decision_path = output_path.with_name('v4-preflight-decision.json')
    receipt_path = output_path.with_name('v4-execution-receipt.json')
    authorization_path = output_path.with_name('v4-execution-authorization.json')
    _reject_attempt_collision(
        (output_path, decision_path, receipt_path, authorization_path)
    )
    history = _historical_values(historical_evidence_paths)
    sampler = (
        nvidia_smi_all_gpu_inventory_sample
        if inventory_sampler is None
        else inventory_sampler
    )
    run_id = uuid4().hex
    samples: list[dict[str, object]] = []
    selection: dict[str, object] = {
        'status': 'FAIL',
        'reason_code': 'V4_GPU_INVENTORY_UNAVAILABLE',
        'candidate_evaluations': [],
        'selected_gpu': None,
    }
    error: str | None = None
    try:
        samples.append(dict(sampler()))
        sleeper(FORMAL_SAMPLE_INTERVAL_SECONDS)
        samples.append(dict(sampler()))
        selection = select_attempt_gpu(samples)
    except V4ExecutionError as exc:
        error = str(exc)
        selection['reason_code'] = str(exc)
    except Exception as exc:
        error = str(exc)

    probes: list[dict[str, object]] = []
    status = str(selection['status'])
    reason = str(selection['reason_code'])
    selected = selection.get('selected_gpu')
    if status == 'PASS' and isinstance(selected, Mapping):
        try:
            selected_uuid = str(selected['uuid'])
            selected_index = int(selected['index'])
            for model in SCIENTIFIC_MODELS:
                probes.append(
                    dict(
                        probe_runner(
                            model,
                            selected_uuid,
                            selected_index,
                            selected,
                        )
                    )
                )
            probe_decision = _evaluate_attempt_probes(probes, selected)
            status = str(probe_decision['status'])
            reason = (
                'V4_ATTEMPT_PREFLIGHT_PASS'
                if status == 'PASS'
                else str(probe_decision['reason_code'])
            )
        except V4ExecutionError as exc:
            status = 'FAIL'
            reason = str(exc)
            error = str(exc)
        except Exception as exc:
            status = 'FAIL'
            reason = 'V4_GPU_TORCH_PROBE_FAILED'
            error = str(exc)

    snapshot: dict[str, object] = {
        'schema_version': ATTEMPT_PREFLIGHT_SCHEMA_VERSION,
        'attempt_id': ATTEMPT_ID,
        'run_id': run_id,
        'implementation_commit': IMPLEMENTATION_ANCHOR_COMMIT,
        'implementation_tree': IMPLEMENTATION_ANCHOR_TREE,
        'sample_interval_seconds': FORMAL_SAMPLE_INTERVAL_SECONDS,
        'status': status,
        'reason_code': reason,
        'inventory_samples': samples,
        'candidate_evaluations': selection['candidate_evaluations'],
        'selected_gpu': selected,
        'model_environment_probes': probes,
        'no_fallback_or_switch': True,
    }
    if error is not None:
        snapshot['error'] = error
    _reject_historical_reuse(snapshot, history)
    _publish_attempt_json(
        output_path, snapshot, validator=_validate_attempt_preflight_payload
    )
    snapshot_sha = sha256_file(output_path)
    decision = _sign_attempt_payload(
        {
            'schema_version': f'{ATTEMPT_SCHEMA_PREFIX}-preflight-decision-1.0',
            'attempt_id': ATTEMPT_ID,
            'run_id': run_id,
            'status': status,
            'reason_code': reason,
            'terminal_status': status,
            'selected_gpu': selected,
            'hardware_preflight_path': str(output_path.resolve()),
            'hardware_preflight_sha256': snapshot_sha,
            'scientific_result': 'NO_SCIENTIFIC_RESULT',
        },
        'decision_sha256',
    )
    _reject_historical_reuse(decision, history)
    _publish_attempt_json(
        decision_path, decision, validator=_validate_attempt_decision_payload
    )
    result: dict[str, object] = {
        'status': status,
        'reason_code': reason,
        'hardware_preflight_path': str(output_path),
        'hardware_preflight_sha256': snapshot_sha,
        'preflight_decision_path': str(decision_path),
        'preflight_decision_sha256': sha256_file(decision_path),
    }
    if status != 'PASS' or not isinstance(selected, Mapping):
        return result
    receipt = _sign_attempt_payload(
        {
            'schema_version': ATTEMPT_RECEIPT_SCHEMA_VERSION,
            'attempt_id': ATTEMPT_ID,
            'run_id': run_id,
            'implementation_commit': IMPLEMENTATION_ANCHOR_COMMIT,
            'implementation_tree': IMPLEMENTATION_ANCHOR_TREE,
            'protocol_id': V4_PROTOCOL_ID,
            'protocol_sha256': V4_PROTOCOL_SHA256,
            'schedule_sha256': schedule_sha256,
            'hardware_preflight_path': str(output_path.resolve()),
            'hardware_preflight_sha256': snapshot_sha,
            'preflight_decision_path': str(decision_path.resolve()),
            'preflight_decision_sha256': sha256_file(decision_path),
            'selected_gpu_uuid': selected['uuid'],
            'selected_physical_index': selected['index'],
            'selected_gpu_model': selected['model'],
            'selected_total_memory_bytes': selected['total_memory_bytes'],
            'max_concurrent_gpus': 1,
            'sequential_model_execution': True,
            'sequential_unit_execution': True,
            'fallback_allowed': False,
            'device_switch_allowed': False,
            'retry_allowed': False,
            'nonce': uuid4().hex,
        },
        'receipt_payload_sha256',
    )
    _reject_historical_reuse(receipt, history)
    _publish_attempt_json(
        receipt_path, receipt, validator=_validate_attempt_receipt_payload
    )
    result['receipt_path'] = str(receipt_path)
    result['receipt_sha256'] = sha256_file(receipt_path)
    return result


def validate_attempt_receipt(path: Path) -> dict[str, object]:
    _require_attempt_path(path)
    payload = load_json(path)
    _validate_attempt_receipt_payload(payload)
    return payload


def _resolve_production_input(root: Path, path: Path) -> Path:
    resolved = _resolve_under_root(root, path, must_exist=True)
    lowered = tuple(part.lower() for part in resolved.parts)
    forbidden_names = {
        'v4-hardware-preflight.json',
        'v4-preflight-decision.json',
        'v4-execution-receipt.json',
        'v4-execution-authorization.json',
    }
    if (
        '.pytest_cache' in lowered
        or 'attempt-01' in lowered
        or resolved.name.lower() in forbidden_names
    ):
        raise V4ExecutionError('V4_PRODUCTION_INPUT_SOURCE_FORBIDDEN')
    return resolved


def _validated_production_file(
    root: Path,
    *,
    path_value: object,
    digest_value: object,
    label: str,
    tamper_reason: str | None = None,
) -> dict[str, str]:
    path = _resolve_production_input(root, Path(str(path_value)))
    reason = tamper_reason or f'V4_RESOURCE_TAMPER:{label}'
    if not _is_sha(digest_value) or sha256_file(path) != digest_value:
        raise V4ExecutionError(reason)
    return {'path': str(path), 'sha256': str(digest_value)}


def _validated_file_binding(
    root: Path, value: object, *, label: str
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {'path', 'sha256'}:
        raise V4ExecutionError(f'V4_RESOURCE_BINDING_SCHEMA_REQUIRED:{label}')
    return _validated_production_file(
        root,
        path_value=value.get('path'),
        digest_value=value.get('sha256'),
        label=label,
    )


def _validated_model_bindings(root: Path, value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(SCIENTIFIC_MODELS):
        raise V4ExecutionError('V4_MODEL_BINDINGS_SCHEMA_REQUIRED')
    normalized: dict[str, object] = {}
    for model in SCIENTIFIC_MODELS:
        row = value[model]
        if not isinstance(row, Mapping) or set(row) != {
            'weights',
            'config',
            'upstream_commit',
        }:
            raise V4ExecutionError('V4_MODEL_BINDINGS_SCHEMA_REQUIRED')
        upstream = row.get('upstream_commit')
        if not _is_sha(upstream, length=40):
            raise V4ExecutionError('V4_MODEL_UPSTREAM_COMMIT_REQUIRED')
        normalized[model] = {
            'weights': _validated_file_binding(
                root, row['weights'], label=f'{model}:weights'
            ),
            'config': _validated_file_binding(
                root, row['config'], label=f'{model}:config'
            ),
            'upstream_commit': upstream,
        }
    return normalized


def _validated_archive_bindings(root: Path, value: object) -> dict[str, object]:
    expected = {'SampleSet', 'Points', 'Rectified'}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise V4ExecutionError('V4_DTU_ARCHIVE_BINDINGS_SCHEMA_REQUIRED')
    normalized: dict[str, object] = {}
    for name in ('SampleSet', 'Points', 'Rectified'):
        row = value[name]
        if not isinstance(row, Mapping) or set(row) != {
            'path',
            'sha256',
            'etag',
            'central_directory_sha256',
            'referenced_members',
        }:
            raise V4ExecutionError('V4_DTU_ARCHIVE_BINDINGS_SCHEMA_REQUIRED')
        archive_file = _validated_production_file(
            root,
            path_value=row.get('path'),
            digest_value=row.get('sha256'),
            label=f'dtu_archive:{name}',
            tamper_reason='V4_DTU_ARCHIVE_TAMPER',
        )
        if not _runtime_proven(row.get('etag')) or not _is_sha(
            row.get('central_directory_sha256')
        ):
            raise V4ExecutionError('V4_DTU_ARCHIVE_IDENTITY_UNPROVEN')
        members = row.get('referenced_members')
        if not isinstance(members, list) or not members:
            raise V4ExecutionError('V4_DTU_MEMBER_DIGESTS_REQUIRED')
        normalized_members: list[dict[str, str]] = []
        seen: set[str] = set()
        for member in members:
            if not isinstance(member, Mapping) or set(member) != {
                'member',
                'sha256',
            }:
                raise V4ExecutionError('V4_DTU_MEMBER_DIGESTS_REQUIRED')
            member_name = member.get('member')
            member_sha = member.get('sha256')
            if (
                not _runtime_proven(member_name)
                or str(member_name) in seen
                or not _is_sha(member_sha)
            ):
                raise V4ExecutionError('V4_DTU_MEMBER_DIGESTS_REQUIRED')
            seen.add(str(member_name))
            normalized_members.append(
                {'member': str(member_name), 'sha256': str(member_sha)}
            )
        normalized[name] = {
            **archive_file,
            'etag': row['etag'],
            'central_directory_sha256': row['central_directory_sha256'],
            'referenced_members': normalized_members,
        }
    return normalized


def _read_state_inventory(path: Path) -> tuple[ModelIndependentState, ...]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(payload, Mapping):
        rows = payload.get('states')
    else:
        rows = payload
    if not isinstance(rows, list) or len(rows) != 200:
        raise V4ExecutionError('V4_STATE_INVENTORY_NOT_EXACT_200')
    try:
        states = tuple(ModelIndependentState.from_dict(row) for row in rows)
    except (TypeError, ValueError) as exc:
        raise V4ExecutionError('V4_STATE_INVENTORY_INVALID') from exc
    keys = [(state.scene_id, state.state_id) for state in states]
    expected = [
        (scene_id, state_id)
        for scene_id in TEST_SCENE_IDS
        for state_id in SCIENTIFIC_STATES
    ]
    if keys != expected or len(set(keys)) != 200:
        raise V4ExecutionError('V4_STATE_INVENTORY_NOT_EXACT_200')
    return states


def _validate_split_binding(path: Path) -> None:
    payload = load_json(path)
    scenes = payload.get('test_scene_ids', payload.get('test'))
    if scenes != list(TEST_SCENE_IDS):
        raise V4ExecutionError('V4_SPLIT_TEST_SCENES_MISMATCH')


def _validate_fog_binding(path: Path) -> None:
    payload = load_json(path)
    if (
        payload.get('model') != 'Koschmieder'
        or payload.get('severity_family') != 'beta-only'
        or payload.get('state_ids') != list(FOG_STATES)
    ):
        raise V4ExecutionError('V4_FOG_MANIFEST_SCOPE_MISMATCH')


def _validate_attempt_resource_payload(payload: Mapping[str, Any]) -> None:
    _require_attempt_id(payload.get('attempt_id'))
    if payload.get('schema_version') != ATTEMPT_RESOURCE_SCHEMA_VERSION:
        raise V4ExecutionError('V4_RESOURCE_INVENTORY_SCHEMA_REQUIRED')
    _validate_attempt_signature(
        payload, 'resource_snapshot_sha256', 'V4_RESOURCE_SNAPSHOT_TAMPER'
    )
    if (
        payload.get('implementation_commit') != IMPLEMENTATION_ANCHOR_COMMIT
        or payload.get('implementation_tree') != IMPLEMENTATION_ANCHOR_TREE
    ):
        raise V4ExecutionError('V4_AUTHORIZATION_STALE_ANCHOR')
    if (
        payload.get('protocol_id') != V4_PROTOCOL_ID
        or payload.get('protocol_sha256') != V4_PROTOCOL_SHA256
    ):
        raise V4ExecutionError('V4_AUTHORIZATION_PROTOCOL_MISMATCH')
    resources = payload.get('resources')
    if not isinstance(resources, Mapping) or set(resources) != set(
        ATTEMPT_RESOURCE_KEYS
    ):
        raise V4ExecutionError('V4_RESOURCE_INVENTORY_SCHEMA_REQUIRED')
    state_binding = resources['state_inventory_200']
    schedule_binding = resources['scientific_schedule_400']
    if not isinstance(state_binding, Mapping) or not isinstance(
        schedule_binding, Mapping
    ):
        raise V4ExecutionError('V4_RESOURCE_INVENTORY_SCHEMA_REQUIRED')
    state_path = Path(str(state_binding.get('path')))
    schedule_path = Path(str(schedule_binding.get('path')))
    for path, binding in (
        (state_path, state_binding),
        (schedule_path, schedule_binding),
    ):
        _require_attempt_path(path)
        if (
            not path.is_file()
            or not _is_sha(binding.get('sha256'))
            or sha256_file(path) != binding.get('sha256')
        ):
            raise V4ExecutionError('V4_RESOURCE_TAMPER')
    state_payload = load_json(state_path)
    if (
        state_payload.get('attempt_id') != ATTEMPT_ID
        or state_payload.get('state_count') != 200
        or not isinstance(state_payload.get('states'), list)
        or len(state_payload['states']) != 200
    ):
        raise V4ExecutionError('V4_STATE_INVENTORY_NOT_EXACT_200')
    states = tuple(
        ModelIndependentState.from_dict(row) for row in state_payload['states']
    )
    schedule_payload = load_json(schedule_path)
    if schedule_payload.get('attempt_id') != ATTEMPT_ID:
        raise V4ExecutionError('V4_ATTEMPT_ID_MISMATCH')
    schedule = validate_scientific_schedule(schedule_payload.get('schedule', {}))
    rebuilt = build_scientific_schedule(states)
    if schedule.schedule_sha256 != rebuilt.schedule_sha256:
        raise V4ExecutionError('V4_SCHEDULE_TAMPER')
    l3_keys = [
        (unit.model_id, unit.scene_id)
        for unit in schedule.units
        if unit.state_id == 'L3'
    ]
    if len(l3_keys) != 40 or len(set(l3_keys)) != 40:
        raise V4ExecutionError('V4_SCHEDULE_L3_DEDUP_REQUIRED')
    if payload.get('schedule_sha256') != schedule.schedule_sha256:
        raise V4ExecutionError('V4_SCHEDULE_TAMPER')
    if payload.get('state_count') != 200 or payload.get('unit_count') != 400:
        raise V4ExecutionError('V4_AUTHORIZATION_SCOPE_EXPANDED')


def materialize_attempt_resources(
    *, root: Path, source_manifest_path: Path, output_path: Path
) -> dict[str, object]:
    '''Materialize the production 200-state/400-unit attempt-02 resources.'''

    resolved_root = root.resolve()
    if resolved_root != _attempt_root_from_path(output_path):
        raise V4ExecutionError('V4_ATTEMPT_PATH_MISMATCH')
    state_path = output_path.with_name('v4-state-inventory-200.json')
    schedule_path = output_path.with_name('v4-scientific-schedule-400.json')
    _reject_attempt_collision((state_path, schedule_path, output_path))
    source_path = _resolve_production_input(resolved_root, source_manifest_path)
    source = load_json(source_path)
    expected_keys = {
        'attempt_id',
        'model_bindings',
        'dtu_archives',
        'v4_split',
        'state_inventory_200',
        'fog_manifest',
        'environment_locks',
        'science_lock',
    }
    if set(source) != expected_keys:
        raise V4ExecutionError('V4_RESOURCE_SOURCE_SCHEMA_REQUIRED')
    _require_attempt_id(source.get('attempt_id'))
    models = _validated_model_bindings(
        resolved_root, source.get('model_bindings')
    )
    archives = _validated_archive_bindings(
        resolved_root, source.get('dtu_archives')
    )
    split = _validated_file_binding(
        resolved_root, source.get('v4_split'), label='v4_split'
    )
    _validate_split_binding(Path(split['path']))
    source_states = _validated_file_binding(
        resolved_root,
        source.get('state_inventory_200'),
        label='state_inventory_200',
    )
    states = _read_state_inventory(Path(source_states['path']))
    fog = _validated_file_binding(
        resolved_root, source.get('fog_manifest'), label='fog_manifest'
    )
    _validate_fog_binding(Path(fog['path']))
    environments_raw = source.get('environment_locks')
    if not isinstance(environments_raw, Mapping) or set(
        environments_raw
    ) != set(SCIENTIFIC_MODELS):
        raise V4ExecutionError('V4_ENVIRONMENT_LOCKS_SCHEMA_REQUIRED')
    environments = {
        model: _validated_file_binding(
            resolved_root,
            environments_raw[model],
            label=f'{model}:environment_lock',
        )
        for model in SCIENTIFIC_MODELS
    }
    science_lock = _validated_file_binding(
        resolved_root, source.get('science_lock'), label='science_lock'
    )
    schedule = build_scientific_schedule(states)
    state_payload = {
        'schema_version': f'{ATTEMPT_SCHEMA_PREFIX}-state-inventory-1.0',
        'attempt_id': ATTEMPT_ID,
        'state_count': 200,
        'source_path': source_states['path'],
        'source_sha256': source_states['sha256'],
        'states': [state.to_dict() for state in states],
    }
    schedule_payload = {
        'schema_version': f'{ATTEMPT_SCHEMA_PREFIX}-schedule-wrapper-1.0',
        'attempt_id': ATTEMPT_ID,
        'state_count': 200,
        'unit_count': 400,
        'l3_inference_count': 40,
        'schedule': schedule.to_dict(),
    }
    _publish_attempt_json(
        state_path,
        state_payload,
        validator=lambda staged: (
            None
            if staged.get('attempt_id') == ATTEMPT_ID
            and staged.get('state_count') == 200
            and isinstance(staged.get('states'), list)
            and len(staged['states']) == 200
            else (_ for _ in ()).throw(
                V4ExecutionError('V4_STATE_INVENTORY_NOT_EXACT_200')
            )
        ),
    )
    _publish_attempt_json(
        schedule_path,
        schedule_payload,
        validator=lambda staged: validate_scientific_schedule(
            staged.get('schedule', {})
        ),
    )
    resources = {
        'model_bindings': models,
        'dtu_archives': archives,
        'v4_split': split,
        'state_inventory_200': {
            'path': str(state_path.resolve()),
            'sha256': sha256_file(state_path),
        },
        'fog_manifest': fog,
        'scientific_schedule_400': {
            'path': str(schedule_path.resolve()),
            'sha256': sha256_file(schedule_path),
        },
        'environment_locks': environments,
        'science_lock': science_lock,
    }
    payload = _sign_attempt_payload(
        {
            'schema_version': ATTEMPT_RESOURCE_SCHEMA_VERSION,
            'attempt_id': ATTEMPT_ID,
            'implementation_commit': IMPLEMENTATION_ANCHOR_COMMIT,
            'implementation_tree': IMPLEMENTATION_ANCHOR_TREE,
            'protocol_id': V4_PROTOCOL_ID,
            'protocol_sha256': V4_PROTOCOL_SHA256,
            'source_manifest_path': str(source_path),
            'source_manifest_sha256': sha256_file(source_path),
            'state_count': 200,
            'unit_count': 400,
            'schedule_sha256': schedule.schedule_sha256,
            'resources': resources,
        },
        'resource_snapshot_sha256',
    )
    _publish_attempt_json(
        output_path, payload, validator=_validate_attempt_resource_payload
    )
    validate_attempt_resources(output_path)
    return {
        'status': 'PASS',
        'resource_snapshot_path': str(output_path),
        'resource_snapshot_sha256': sha256_file(output_path),
        'schedule_sha256': schedule.schedule_sha256,
        'state_count': 200,
        'unit_count': 400,
    }


def validate_attempt_resources(path: Path) -> dict[str, object]:
    root = _attempt_root_from_path(path)
    payload = load_json(path)
    _validate_attempt_resource_payload(payload)
    _validate_all_attempt_resource_files(payload, root=root)
    return payload


def _validate_all_attempt_resource_files(
    payload: Mapping[str, Any], *, root: Path
) -> None:
    resources = payload.get('resources')
    if not isinstance(resources, Mapping):
        raise V4ExecutionError('V4_RESOURCE_INVENTORY_SCHEMA_REQUIRED')
    _validated_model_bindings(root, resources.get('model_bindings'))
    _validated_archive_bindings(root, resources.get('dtu_archives'))
    split = _validated_file_binding(
        root, resources.get('v4_split'), label='v4_split'
    )
    fog = _validated_file_binding(
        root, resources.get('fog_manifest'), label='fog_manifest'
    )
    _validated_file_binding(
        root, resources.get('science_lock'), label='science_lock'
    )
    environments = resources.get('environment_locks')
    if not isinstance(environments, Mapping) or set(environments) != set(
        SCIENTIFIC_MODELS
    ):
        raise V4ExecutionError('V4_ENVIRONMENT_LOCKS_SCHEMA_REQUIRED')
    for model in SCIENTIFIC_MODELS:
        _validated_file_binding(
            root,
            environments[model],
            label=f'{model}:environment_lock',
        )
    state = _validated_file_binding(
        root,
        resources.get('state_inventory_200'),
        label='state_inventory_200',
    )
    schedule = _validated_file_binding(
        root,
        resources.get('scientific_schedule_400'),
        label='scientific_schedule_400',
    )
    _require_attempt_path(Path(state['path']))
    _require_attempt_path(Path(schedule['path']))
    state_payload = load_json(Path(state['path']))
    _validated_production_file(
        root,
        path_value=state_payload.get('source_path'),
        digest_value=state_payload.get('source_sha256'),
        label='state_inventory_source',
    )
    _validated_production_file(
        root,
        path_value=payload.get('source_manifest_path'),
        digest_value=payload.get('source_manifest_sha256'),
        label='source_manifest',
        tamper_reason='V4_RESOURCE_SOURCE_TAMPER',
    )
    _validate_split_binding(Path(split['path']))
    _validate_fog_binding(Path(fog['path']))


def _validate_attempt_authorization_payload(payload: Mapping[str, Any]) -> None:
    _require_attempt_id(payload.get('attempt_id'))
    if payload.get('schema_version') != ATTEMPT_AUTHORIZATION_SCHEMA_VERSION:
        raise V4ExecutionError('V4_AUTHORIZATION_SCHEMA_REQUIRED')
    _validate_attempt_signature(
        payload, 'authorization_sha256', 'V4_AUTHORIZATION_TAMPER'
    )
    if (
        payload.get('implementation_commit') != IMPLEMENTATION_ANCHOR_COMMIT
        or payload.get('implementation_tree') != IMPLEMENTATION_ANCHOR_TREE
    ):
        raise V4ExecutionError('V4_AUTHORIZATION_STALE_ANCHOR')
    if (
        payload.get('protocol_id') != V4_PROTOCOL_ID
        or payload.get('protocol_sha256') != V4_PROTOCOL_SHA256
    ):
        raise V4ExecutionError('V4_AUTHORIZATION_PROTOCOL_MISMATCH')
    if payload.get('finalizer') != AUTHORIZED_FINALIZER:
        raise V4ExecutionError('V4_AUTHORIZATION_FINALIZER_MISMATCH')
    if payload.get('authorized_scope') != _authorized_scope():
        raise V4ExecutionError('V4_AUTHORIZATION_SCOPE_EXPANDED')
    root = Path(str(payload.get('root'))).resolve()
    paths = (
        Path(str(payload.get('run_root'))),
        Path(str(payload.get('artifact_root'))),
        Path(str(payload.get('final_evidence_path'))),
    )
    _resolve_under_root(root, paths[0], must_exist=True)
    _resolve_under_root(root, paths[1], must_exist=True)
    _resolve_under_root(root, paths[2])
    for path in paths:
        _require_attempt_path(path)
    receipt_path = Path(str(payload.get('receipt_path')))
    resources_path = Path(str(payload.get('resource_snapshot_path')))
    _require_attempt_path(receipt_path)
    _require_attempt_path(resources_path)
    _resolve_under_root(root, receipt_path, must_exist=True)
    _resolve_under_root(root, resources_path, must_exist=True)
    if (
        not receipt_path.is_file()
        or sha256_file(receipt_path) != payload.get('receipt_sha256')
    ):
        raise V4ExecutionError('V4_RECEIPT_TAMPER')
    if (
        not resources_path.is_file()
        or sha256_file(resources_path) != payload.get('resource_snapshot_file_sha256')
    ):
        raise V4ExecutionError('V4_RESOURCE_SNAPSHOT_TAMPER')
    receipt = validate_attempt_receipt(receipt_path)
    resources = validate_attempt_resources(resources_path)
    if (
        receipt.get('schedule_sha256') != resources.get('schedule_sha256')
        or receipt.get('schedule_sha256') != payload.get('schedule_sha256')
        or receipt.get('selected_gpu_uuid') != payload.get('selected_gpu_uuid')
        or receipt.get('selected_physical_index')
        != payload.get('selected_physical_index')
        or receipt.get('run_id') != payload.get('run_id')
    ):
        raise V4ExecutionError('V4_AUTHORIZATION_RECEIPT_MISMATCH')
    if (
        payload.get('max_concurrent_gpus') != 1
        or payload.get('sequential_execution') is not True
        or payload.get('fallback_allowed') is not False
        or payload.get('device_switch_allowed') is not False
        or payload.get('retry_allowed') is not False
    ):
        raise V4ExecutionError('V4_AUTHORIZATION_SCOPE_EXPANDED')


def create_attempt_execution_authorization(
    *,
    root: Path,
    receipt_path: Path,
    resource_snapshot_path: Path,
    run_root: Path,
    artifact_root: Path,
    final_evidence_path: Path,
    output_path: Path,
) -> dict[str, object]:
    '''Authorize only the exact attempt-02 receipt and production resources.'''

    _require_attempt_path(output_path)
    _reject_attempt_collision((output_path,))
    resolved_root = root.resolve()
    receipt = validate_attempt_receipt(receipt_path)
    resources = validate_attempt_resources(resource_snapshot_path)
    if receipt.get('schedule_sha256') != resources.get('schedule_sha256'):
        raise V4ExecutionError('V4_AUTHORIZATION_SCHEDULE_MISMATCH')
    resolved_run = _resolve_under_root(resolved_root, run_root, must_exist=True)
    resolved_artifact = _resolve_under_root(
        resolved_root, artifact_root, must_exist=True
    )
    resolved_evidence = _resolve_under_root(
        resolved_root, final_evidence_path
    )
    for path in (resolved_run, resolved_artifact, resolved_evidence):
        _require_attempt_path(path)
    payload = _sign_attempt_payload(
        {
            'schema_version': ATTEMPT_AUTHORIZATION_SCHEMA_VERSION,
            'attempt_id': ATTEMPT_ID,
            'run_id': receipt['run_id'],
            'implementation_commit': IMPLEMENTATION_ANCHOR_COMMIT,
            'implementation_tree': IMPLEMENTATION_ANCHOR_TREE,
            'protocol_id': V4_PROTOCOL_ID,
            'protocol_sha256': V4_PROTOCOL_SHA256,
            'receipt_path': str(receipt_path.resolve()),
            'receipt_sha256': sha256_file(receipt_path),
            'resource_snapshot_path': str(resource_snapshot_path.resolve()),
            'resource_snapshot_file_sha256': sha256_file(
                resource_snapshot_path
            ),
            'resource_snapshot_sha256': resources[
                'resource_snapshot_sha256'
            ],
            'schedule_sha256': receipt['schedule_sha256'],
            'selected_gpu_uuid': receipt['selected_gpu_uuid'],
            'selected_physical_index': receipt['selected_physical_index'],
            'root': str(resolved_root),
            'run_root': str(resolved_run),
            'artifact_root': str(resolved_artifact),
            'final_evidence_path': str(resolved_evidence),
            'finalizer': AUTHORIZED_FINALIZER,
            'authorized_scope': _authorized_scope(),
            'max_concurrent_gpus': 1,
            'sequential_execution': True,
            'fallback_allowed': False,
            'device_switch_allowed': False,
            'retry_allowed': False,
        },
        'authorization_sha256',
    )
    _publish_attempt_json(
        output_path,
        payload,
        validator=_validate_attempt_authorization_payload,
    )
    return {
        'status': 'PASS',
        'authorization_path': str(output_path),
        'authorization_sha256': sha256_file(output_path),
    }


def validate_attempt_execution_authorization(
    path: Path,
) -> dict[str, object]:
    _require_attempt_path(path)
    payload = load_json(path)
    _validate_attempt_authorization_payload(payload)
    return payload


PROCESS_EVIDENCE_REASON_CODES = PROCESS_PRESENT_REASON_CODES | frozenset(
    {"V4_GPU_UNEXPLAINED_ACTIVITY"}
)


@dataclass(frozen=True, slots=True)
class V4ExecutionAuthorization:
    receipt_path: str
    receipt_sha256: str
    hardware_preflight_path: str
    hardware_preflight_sha256: str
    implementation_commit: str
    implementation_tree: str
    protocol_id: str
    protocol_sha256: str
    schedule_sha256: str
    resource_inventory: tuple[tuple[str, str, str], ...]
    root: str
    run_root: str
    artifact_root: str
    final_evidence_path: str
    finalizer: str
    schema_version: str = V4_EXECUTION_AUTHORIZATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["resource_inventory"] = [
            {"key": key, "path": path, "sha256": digest}
            for key, path, digest in self.resource_inventory
        ]
        payload["authorized_scope"] = _authorized_scope()
        payload["authorization_sha256"] = _sha_json(
            {key: value for key, value in payload.items() if key != "authorization_sha256"}
        )
        return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), default=str)
        + "\n"
    ).encode("utf-8")


def _atomic_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    validator: Callable[[Mapping[str, Any]], None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        with partial.open("wb") as handle:
            handle.write(_json_bytes(payload))
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        reloaded = json.loads(partial.read_text(encoding="utf-8"))
        if not isinstance(reloaded, dict):
            raise V4ExecutionError("V4_ATOMIC_JSON_STAGING_VALIDATION_FAILED")
        validator(reloaded)
        partial.replace(path)
    finally:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass


def _runtime_proven(value: object) -> bool:
    return isinstance(value, str) and value.strip() != "" and value.strip().lower() != "unknown"


def _driver_proven(value: object) -> bool:
    return isinstance(value, str) and value.strip() != "" and value.strip().lower() != "unknown"


def _validate_preflight_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != V4_HARDWARE_PREFLIGHT_SCHEMA_VERSION:
        raise V4ExecutionError("V4_GPU_PREFLIGHT_SCHEMA_REQUIRED")
    if payload.get("status") not in {"PASS", "FAIL"} or not isinstance(payload.get("reason_code"), str):
        raise V4ExecutionError("V4_GPU_PREFLIGHT_SCHEMA_REQUIRED")
    for sample in payload.get("samples", ()):
        if not isinstance(sample, Mapping) or not _runtime_proven(sample.get("cuda_runtime")):
            raise V4ExecutionError("V4_GPU_CUDA_RUNTIME_UNPROVEN")
        if not _driver_proven(sample.get("driver_version")):
            raise V4ExecutionError("V4_GPU_DRIVER_VERSION_UNPROVEN")
    for device in payload.get("devices", ()):
        if not isinstance(device, Mapping) or not _runtime_proven(device.get("cuda_runtime")):
            raise V4ExecutionError("V4_GPU_CUDA_RUNTIME_UNPROVEN")
        if not _driver_proven(device.get("driver_version")):
            raise V4ExecutionError("V4_GPU_DRIVER_VERSION_UNPROVEN")
    _validate_evidence_reason_consistency(payload)
    if payload.get("status") == "PASS":
        decision = _evaluate_basic(payload.get("samples", ()))
        if decision["status"] != "PASS":
            raise V4ExecutionError(str(decision["reason_code"]))
        probe_decision = _evaluate_probes(payload.get("model_environment_probes", ()), payload["samples"][-1])
        if probe_decision["status"] != "PASS":
            raise V4ExecutionError(str(probe_decision["reason_code"]))


def _validate_receipt_payload(payload: Mapping[str, Any]) -> None:
    receipt = V4ExecutionReceipt.from_mapping(payload)
    _validate_receipt_contract(receipt)


def _validate_authorization_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != V4_EXECUTION_AUTHORIZATION_SCHEMA_VERSION:
        raise V4ExecutionError("V4_AUTHORIZATION_SCHEMA_REQUIRED")
    expected_sha = payload.get("authorization_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "authorization_sha256"}
    if not _is_sha(expected_sha) or _sha_json(unsigned) != expected_sha:
        raise V4ExecutionError("V4_AUTHORIZATION_TAMPER")
    if payload.get("finalizer") != AUTHORIZED_FINALIZER:
        raise V4ExecutionError("V4_AUTHORIZATION_FINALIZER_MISMATCH")


def _validate_preflight_decision_payload(
    payload: Mapping[str, Any],
    *,
    expected_snapshot_path: Path | None = None,
    expected_authorization_commit: str | None = None,
    expected_authorization_tree: str | None = None,
    expected_run_id: str | None = None,
) -> None:
    if payload.get("schema_version") != V4_PREFLIGHT_DECISION_SCHEMA_VERSION:
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_SCHEMA_REQUIRED")
    if payload.get("implementation_commit") != IMPLEMENTATION_ANCHOR_COMMIT or payload.get("implementation_tree") != IMPLEMENTATION_ANCHOR_TREE:
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_ANCHOR_MISMATCH")
    authorization_commit = payload.get("authorization_commit")
    authorization_tree = payload.get("authorization_tree")
    if not _is_sha(authorization_commit, length=40) or not _is_sha(authorization_tree, length=40):
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_REVISION_MISMATCH")
    if expected_authorization_commit is not None and authorization_commit != expected_authorization_commit:
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_REVISION_MISMATCH")
    if expected_authorization_tree is not None and authorization_tree != expected_authorization_tree:
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_REVISION_MISMATCH")
    run_id = payload.get("run_id")
    if not _is_sha(run_id, length=32):
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_SCHEMA_REQUIRED")
    if expected_run_id is not None and run_id != expected_run_id:
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_RUN_MISMATCH")
    if not isinstance(payload.get("requested_physical_index"), int) or isinstance(payload.get("requested_physical_index"), bool):
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_SCHEMA_REQUIRED")
    if payload.get("status") != "BLOCKED" or not isinstance(payload.get("reason_code"), str):
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_BLOCKED_REQUIRED")
    if payload.get("terminal_status") != "BLOCKED" or payload.get("terminal_reason_code") != payload.get("reason_code"):
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_TERMINAL_REQUIRED")
    if payload.get("scientific_result") != "NO_SCIENTIFIC_RESULT":
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_NO_SCIENTIFIC_RESULT_REQUIRED")
    snapshot_path = Path(str(payload.get("hardware_preflight_path")))
    if expected_snapshot_path is not None and snapshot_path.resolve() != expected_snapshot_path.resolve():
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_SNAPSHOT_PATH_MISMATCH")
    if not snapshot_path.is_file():
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_SNAPSHOT_REQUIRED")
    snapshot_sha = payload.get("hardware_preflight_sha256")
    if not _is_sha(snapshot_sha) or sha256_file(snapshot_path) != snapshot_sha:
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_SNAPSHOT_TAMPER")
    snapshot = load_json(snapshot_path)
    if snapshot.get("schema_version") != V4_HARDWARE_PREFLIGHT_SCHEMA_VERSION or snapshot.get("status") != "FAIL":
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_SNAPSHOT_MISMATCH")
    if snapshot.get("reason_code") != payload.get("reason_code"):
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_SNAPSHOT_MISMATCH")
    if snapshot.get("project_commit") != IMPLEMENTATION_ANCHOR_COMMIT or snapshot.get("project_tree") != IMPLEMENTATION_ANCHOR_TREE:
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_SNAPSHOT_MISMATCH")
    if snapshot.get("authorization_commit") != authorization_commit or snapshot.get("authorization_tree") != authorization_tree:
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_SNAPSHOT_MISMATCH")
    if snapshot.get("requested_physical_index") != payload.get("requested_physical_index"):
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_SNAPSHOT_MISMATCH")
    if snapshot.get("run_id") != run_id:
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_RUN_MISMATCH")
    _validate_evidence_reason_consistency(snapshot)


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V4ExecutionError("V4_AUTHORIZATION_JSON_UNREADABLE") from exc
    if not isinstance(payload, dict):
        raise V4ExecutionError("V4_AUTHORIZATION_JSON_NOT_OBJECT")
    return payload


def _is_sha(value: object, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _current_authorization_revision() -> tuple[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        commit = _run_text_command(("git", "-C", str(repo_root), "rev-parse", "HEAD")).strip()
        tree = _run_text_command(("git", "-C", str(repo_root), "rev-parse", "HEAD^{tree}")).strip()
    except Exception as exc:
        raise V4ExecutionError("V4_AUTHORIZATION_REVISION_UNRESOLVED") from exc
    if not _is_sha(commit, length=40) or not _is_sha(tree, length=40):
        raise V4ExecutionError("V4_AUTHORIZATION_REVISION_UNRESOLVED")
    return commit, tree


def _resolve_authorization_revision(authorization_commit: str | None, authorization_tree: str | None) -> tuple[str, str]:
    if authorization_commit is None and authorization_tree is None:
        return _current_authorization_revision()
    if not _is_sha(authorization_commit, length=40) or not _is_sha(authorization_tree, length=40):
        raise V4ExecutionError("V4_AUTHORIZATION_REVISION_REQUIRED")
    return str(authorization_commit), str(authorization_tree)


def _resolve_under_root(root: Path, target: Path, *, must_exist: bool = False) -> Path:
    resolved_root = root.resolve()
    if resolved_root == Path(resolved_root.anchor):
        raise V4ExecutionError("V4_AUTHORIZATION_ROOT_INVALID")
    resolved = target.resolve(strict=must_exist)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise V4ExecutionError("V4_AUTHORIZATION_PATH_ESCAPE") from exc
    if not relative.parts:
        raise V4ExecutionError("V4_AUTHORIZATION_PATH_ESCAPE")
    return resolved


def _run_text_command(command: Sequence[str], *, env: Mapping[str, str] | None = None) -> str:
    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update({key: str(value) for key, value in env.items()})
    return subprocess.run(
        list(command),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=merged_env,
    ).stdout


def _parse_csv_row(text: str, expected_columns: int, *, reason_code: str) -> list[str]:
    rows = [row.strip() for row in text.splitlines() if row.strip()]
    if len(rows) != 1:
        raise V4ExecutionError(reason_code)
    values = [item.strip() for item in rows[0].split(",")]
    if len(values) != expected_columns:
        raise V4ExecutionError(reason_code)
    return values


def _int_from_smi(value: str) -> int:
    stripped = value.strip()
    if stripped in {"", "N/A", "[Not Supported]", "Not Supported"}:
        return 0
    return int(float(stripped))


def _bytes_from_mib(value: str) -> int:
    return _int_from_smi(value) * 1024 * 1024


def _cuda_runtime_from_nvidia_smi_banner(text: str) -> str:
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)*)", text)
    if match is None:
        raise V4ExecutionError("V4_GPU_CUDA_RUNTIME_UNPROVEN")
    return f"CUDA Version {match.group(1)}"


def _nvidia_smi_cuda_runtime(command_runner: Callable[..., str]) -> str:
    try:
        runtime = _cuda_runtime_from_nvidia_smi_banner(command_runner(("nvidia-smi",)))
    except V4ExecutionError:
        raise
    except Exception as exc:
        raise V4ExecutionError("V4_GPU_CUDA_RUNTIME_UNPROVEN") from exc
    if not _runtime_proven(runtime):
        raise V4ExecutionError("V4_GPU_CUDA_RUNTIME_UNPROVEN")
    return runtime


def _process_owner(pid: int) -> str | None:
    if os.name == "nt":
        return None
    try:
        import pwd

        uid_text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8").split("Uid:")[1].splitlines()[0].strip().split()[0]
        return pwd.getpwuid(int(uid_text)).pw_name
    except (OSError, IndexError, KeyError, ValueError):
        return None


def _process_cwd(pid: int) -> str | None:
    if os.name == "nt":
        return None
    try:
        return str(Path(f"/proc/{pid}/cwd").resolve())
    except OSError:
        return None


def _process_cmdline(pid: int) -> str | None:
    if os.name == "nt":
        return None
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except OSError:
        return None


def nvidia_smi_hardware_sample(
    requested_physical_index: int,
    *,
    command_runner: Callable[..., str] = _run_text_command,
) -> dict[str, object]:
    base = _parse_csv_row(
        command_runner((
            "nvidia-smi",
            "-i",
            str(requested_physical_index),
            "--query-gpu=index,uuid,name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu,driver_version",
            "--format=csv,noheader,nounits",
        )),
        9,
        reason_code="V4_GPU_BASIC_SAMPLE_UNAVAILABLE",
    )
    mig_ecc = _parse_csv_row(
        command_runner((
            "nvidia-smi",
            "-i",
            str(requested_physical_index),
            "--query-gpu=mig.mode.current,ecc.errors.uncorrected.volatile.total",
            "--format=csv,noheader,nounits",
        )),
        2,
        reason_code="V4_GPU_HEALTH_SAMPLE_UNAVAILABLE",
    )
    cuda_runtime = _nvidia_smi_cuda_runtime(command_runner)
    processes: list[dict[str, object]] = []
    try:
        proc_text = command_runner((
            "nvidia-smi",
            "-i",
            str(requested_physical_index),
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ))
    except Exception as exc:
        raise V4ExecutionError("V4_GPU_PROCESS_ENUMERATION_UNPROVEN") from exc
    for row in [line.strip() for line in proc_text.splitlines() if line.strip()]:
        gpu_uuid, pid_raw, process_name, used_raw = [item.strip() for item in row.split(",", 3)]
        pid = _int_from_smi(pid_raw)
        processes.append({
            "gpu_uuid": gpu_uuid,
            "pid": pid,
            "process_name": process_name,
            "owner": _process_owner(pid),
            "cwd": _process_cwd(pid),
            "cmdline": _process_cmdline(pid),
            "used_memory_bytes": _bytes_from_mib(used_raw),
        })
    return {
        "host": platform.node(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "requested_physical_index": requested_physical_index,
        "resolved_physical_index": _int_from_smi(base[0]),
        "device_uuid": base[1],
        "device_model": base[2],
        "total_memory_bytes": _bytes_from_mib(base[3]),
        "free_memory_bytes": _bytes_from_mib(base[4]),
        "used_memory_bytes": _bytes_from_mib(base[5]),
        "utilization_gpu_percent": _int_from_smi(base[6]),
        "temperature_c": _int_from_smi(base[7]),
        "driver_version": base[8],
        "cuda_runtime": cuda_runtime,
        "mig_mode": mig_ecc[0],
        "ecc_health": "OK" if _int_from_smi(mig_ecc[1]) == 0 else "ERROR",
        "compute_processes": processes,
    }


def _frozen_python_for_model(model_id: str) -> str:
    env_key = {"VGGT": "GEORELIAB_V4_VGGT_PYTHON", "MASt3R": "GEORELIAB_V4_MAST3R_PYTHON"}.get(model_id)
    if env_key is None:
        raise V4ExecutionError("V4_GPU_TORCH_PROBE_MODEL_SET_MISMATCH")
    value = os.environ.get(env_key)
    if not value:
        raise V4ExecutionError(f"V4_GPU_TORCH_PROBE_FROZEN_ENV_REQUIRED:{env_key}")
    return value


def _default_torch_probe(model_id: str, requested_physical_index: int, expected_sample: Mapping[str, object]) -> dict[str, object]:
    code = (
        "import json, torch;"
        "props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() and torch.cuda.device_count()==1 else None;"
        "payload={'torch_cuda_available': torch.cuda.is_available(),"
        "'torch_device_count': torch.cuda.device_count(),"
        "'torch_current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,"
        "'device_name': props.name if props else None,"
        "'total_memory_bytes': props.total_memory if props else None};"
        "print(json.dumps(payload, sort_keys=True))"
    )
    payload = json.loads(_run_text_command((_frozen_python_for_model(model_id), "-c", code), env={"CUDA_VISIBLE_DEVICES": str(requested_physical_index)}))
    post_probe_sample = nvidia_smi_hardware_sample(requested_physical_index)
    return {
        "model_id": model_id,
        "torch_device_count": payload.get("torch_device_count"),
        "torch_cuda_available": payload.get("torch_cuda_available"),
        "torch_current_device": payload.get("torch_current_device"),
        "mapped_device_uuid": post_probe_sample.get("device_uuid"),
        "mapped_device_model": payload.get("device_name"),
        "mapped_total_memory_bytes": payload.get("total_memory_bytes"),
        "post_probe_physical_model": post_probe_sample.get("device_model"),
        "post_probe_physical_total_memory_bytes": post_probe_sample.get("total_memory_bytes"),
        "compute_process_count": len(post_probe_sample.get("compute_processes", ())),
    }


def _fail(reason_code: str, **extra: object) -> dict[str, object]:
    return {"status": "FAIL", "reason_code": reason_code, **extra}


def _active_process_identity(
    process: Mapping[str, object],
) -> tuple[int, str, str] | None:
    pid = process.get("pid")
    owner = process.get("owner")
    cmdline = process.get("cmdline")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 0
        or not isinstance(owner, str)
        or not owner.strip()
        or not isinstance(cmdline, str)
        or not cmdline.strip()
    ):
        return None
    return pid, owner.strip(), cmdline.strip()


def _is_georeliab_process(process: Mapping[str, object]) -> bool:
    provenance = " ".join(
        str(process.get(key) or "")
        for key in ("process_name", "cwd", "cmdline")
    ).lower()
    return "georeliab" in provenance


def _normalize_active_compute_process_rows(
    samples: Sequence[Mapping[str, object]],
) -> list[list[Mapping[str, object]]] | None:
    process_rows: list[list[Mapping[str, object]]] = []
    for sample in samples:
        raw_processes = sample.get("compute_processes", ())
        if not isinstance(raw_processes, Sequence) or isinstance(
            raw_processes, (str, bytes)
        ):
            return None
        rows: list[Mapping[str, object]] = []
        for process in raw_processes:
            if not isinstance(process, Mapping):
                return None
            rows.append(process)
        process_rows.append(rows)
    return process_rows


def _classify_normalized_active_compute_process_rows(
    samples: Sequence[Mapping[str, object]],
    process_rows: Sequence[Sequence[Mapping[str, object]]],
) -> str | None:
    if any(
        _active_process_identity(process) is None
        for rows in process_rows
        for process in rows
    ):
        return "V4_GPU_ACTIVE_COMPUTE_PROCESS_IDENTITY_INCOMPLETE"

    if not any(process_rows):
        if any(sample.get("utilization_gpu_percent") != 0 for sample in samples):
            return "V4_GPU_UNEXPLAINED_ACTIVITY"
        return None

    identity_sets = [
        {_active_process_identity(process) for process in rows}
        for rows in process_rows
    ]
    if any(identities != identity_sets[0] for identities in identity_sets[1:]):
        return "V4_GPU_ACTIVE_COMPUTE_PROCESS_STATE_UNSTABLE"
    if any(
        not _is_georeliab_process(process)
        for rows in process_rows
        for process in rows
    ):
        return "V4_GPU_NON_GEORELIAB_COMPUTE_PROCESS_PRESENT"
    return "V4_GPU_GEORELIAB_RESIDUAL_COMPUTE_PROCESS_PRESENT"


def _classify_active_compute_processes(
    samples: Sequence[Mapping[str, object]],
) -> str | None:
    process_rows = _normalize_active_compute_process_rows(samples)
    if process_rows is None:
        return "V4_GPU_ACTIVE_COMPUTE_PROCESS_IDENTITY_INCOMPLETE"
    return _classify_normalized_active_compute_process_rows(samples, process_rows)


def _validate_evidence_reason_consistency(payload: Mapping[str, Any]) -> None:
    samples = payload.get("samples", ())
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise V4ExecutionError("V4_GPU_EVIDENCE_REASON_COUNT_CONTRADICTION")
    if not samples:
        if payload.get("reason_code") in (
            PROCESS_PRESENT_REASON_CODES | NO_ACTIVE_PROCESS_REASON_CODES
        ):
            raise V4ExecutionError("V4_GPU_EVIDENCE_REASON_COUNT_CONTRADICTION")
        return
    if any(not isinstance(sample, Mapping) for sample in samples):
        raise V4ExecutionError("V4_GPU_EVIDENCE_REASON_COUNT_CONTRADICTION")

    process_rows = _normalize_active_compute_process_rows(samples)
    if process_rows is None:
        raise V4ExecutionError("V4_GPU_EVIDENCE_REASON_COUNT_CONTRADICTION")
    observed_count = max(len(rows) for rows in process_rows)
    declared_count = payload.get("compute_process_count")
    if declared_count is not None and (
        not isinstance(declared_count, int)
        or isinstance(declared_count, bool)
        or declared_count != observed_count
    ):
        raise V4ExecutionError("V4_GPU_EVIDENCE_REASON_COUNT_CONTRADICTION")
    for device in payload.get("devices", ()):
        if (
            isinstance(device, Mapping)
            and "compute_process_count" in device
            and device.get("compute_process_count") != observed_count
        ):
            raise V4ExecutionError("V4_GPU_EVIDENCE_REASON_COUNT_CONTRADICTION")

    reason_code = payload.get("reason_code")
    if (
        reason_code in (PROCESS_PRESENT_REASON_CODES | NO_ACTIVE_PROCESS_REASON_CODES)
        and declared_count is None
    ):
        raise V4ExecutionError("V4_GPU_EVIDENCE_REASON_COUNT_CONTRADICTION")
    if reason_code in PROCESS_PRESENT_REASON_CODES and observed_count == 0:
        raise V4ExecutionError("V4_GPU_EVIDENCE_REASON_COUNT_CONTRADICTION")
    if reason_code in NO_ACTIVE_PROCESS_REASON_CODES and observed_count > 0:
        raise V4ExecutionError("V4_GPU_EVIDENCE_REASON_COUNT_CONTRADICTION")
    expected_reason = _classify_normalized_active_compute_process_rows(
        samples, process_rows
    )
    if expected_reason is not None and reason_code != expected_reason:
        raise V4ExecutionError("V4_GPU_EVIDENCE_REASON_COUNT_CONTRADICTION")
    if expected_reason is None and reason_code in PROCESS_EVIDENCE_REASON_CODES:
        raise V4ExecutionError("V4_GPU_EVIDENCE_REASON_COUNT_CONTRADICTION")
    if reason_code in NO_ACTIVE_PROCESS_REASON_CODES:
        raise V4ExecutionError("V4_GPU_LEGACY_INVERTED_REASON_FORBIDDEN")


def _evaluate_basic(samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(samples) != 2:
        return _fail("V4_GPU_PREFLIGHT_TWO_SAMPLES_REQUIRED")
    active_process_reason = _classify_active_compute_processes(samples)
    if active_process_reason is not None:
        return _fail(active_process_reason)
    first, second = samples
    for sample in samples:
        if sample.get("requested_physical_index") != sample.get("resolved_physical_index"):
            return _fail("V4_GPU_INDEX_RESOLUTION_MISMATCH")
        if not isinstance(sample.get("device_uuid"), str) or not sample.get("device_uuid"):
            return _fail("V4_GPU_UUID_MISSING")
        if sample.get("device_model") != AUTHORIZED_GPU_MODEL:
            return _fail("V4_GPU_MODEL_NOT_AUTHORIZED")
        if not _driver_proven(sample.get("driver_version")):
            return _fail("V4_GPU_DRIVER_VERSION_UNPROVEN")
        if str(sample.get("mig_mode")).lower() not in {"disabled", "off", "0"}:
            return _fail("V4_GPU_MIG_ENABLED")
        if str(sample.get("ecc_health")).upper() not in {"OK", "PASS", "NONE", "0"}:
            return _fail("V4_GPU_HEALTH_ERROR")
        if not isinstance(sample.get("free_memory_bytes"), int) or int(sample["free_memory_bytes"]) < MIN_FREE_MEMORY_BYTES:
            return _fail("V4_GPU_FREE_MEMORY_INSUFFICIENT")
    for key, reason in (("resolved_physical_index", "V4_GPU_INDEX_DRIFT"), ("device_uuid", "V4_GPU_UUID_DRIFT"), ("device_model", "V4_GPU_MODEL_DRIFT"), ("driver_version", "V4_GPU_DRIVER_VERSION_DRIFT")):
        if first.get(key) != second.get(key):
            return _fail(reason)
    return {"status": "PASS", "reason_code": "V4_GPU_BASIC_PREFLIGHT_PASS"}


def _evaluate_probes(probes: Sequence[Mapping[str, object]], sample: Mapping[str, object]) -> dict[str, object]:
    seen: set[str] = set()
    if len(probes) != len(SCIENTIFIC_MODELS):
        return _fail("V4_GPU_TORCH_PROBE_MODEL_SET_MISMATCH")
    for probe in probes:
        model_id = probe.get("model_id")
        if model_id not in SCIENTIFIC_MODELS or str(model_id) in seen:
            return _fail("V4_GPU_TORCH_PROBE_MODEL_SET_MISMATCH")
        seen.add(str(model_id))
        if probe.get("torch_cuda_available") is not True or probe.get("torch_device_count") != 1 or probe.get("torch_current_device") != 0:
            return _fail("V4_GPU_TORCH_PROBE_VISIBLE_DEVICE_MISMATCH")
        if (
            not isinstance(probe.get("mapped_device_uuid"), str)
            or not isinstance(probe.get("mapped_device_model"), str)
            or not isinstance(probe.get("mapped_total_memory_bytes"), int)
            or not isinstance(probe.get("post_probe_physical_model"), str)
            or not isinstance(probe.get("post_probe_physical_total_memory_bytes"), int)
            or not isinstance(probe.get("compute_process_count"), int)
        ):
            return _fail("V4_GPU_TORCH_PROBE_SCHEMA_REQUIRED")
        if (
            probe["mapped_device_uuid"] != sample.get("device_uuid")
            or probe["mapped_device_model"] != sample.get("device_model")
            or probe["mapped_total_memory_bytes"] != sample.get("total_memory_bytes")
            or probe["post_probe_physical_model"] != sample.get("device_model")
            or probe["post_probe_physical_total_memory_bytes"] != sample.get("total_memory_bytes")
        ):
            return _fail("V4_GPU_TORCH_PROBE_PHYSICAL_DEVICE_MISMATCH")
        if probe["compute_process_count"] != 0:
            return _fail("V4_GPU_TORCH_PROBE_LEFT_PROCESS")
    return {"status": "PASS", "reason_code": "V4_GPU_PREFLIGHT_PASS"}


def _remove_owned_preflight_siblings(output_path: Path) -> None:
    for name in (
        "v4-execution-receipt.json",
        "v4-execution-authorization.json",
        "authorization.json",
        "v4-execution-schedule.json",
        "v4-state-inventory.json",
    ):
        sibling = output_path.with_name(name)
        try:
            sibling.unlink()
        except FileNotFoundError:
            pass


def _preflight_decision_path(output_path: Path) -> Path:
    return output_path.with_name("v4-preflight-decision.json")


def _write_blocked_preflight_decision(
    *,
    output_path: Path,
    requested_physical_index: int,
    reason_code: str,
    authorization_commit: str,
    authorization_tree: str,
    run_id: str,
) -> dict[str, object]:
    snapshot_sha = sha256_file(output_path)
    decision_path = _preflight_decision_path(output_path)
    payload = {
        "schema_version": V4_PREFLIGHT_DECISION_SCHEMA_VERSION,
        "implementation_commit": IMPLEMENTATION_ANCHOR_COMMIT,
        "implementation_tree": IMPLEMENTATION_ANCHOR_TREE,
        "authorization_commit": authorization_commit,
        "authorization_tree": authorization_tree,
        "run_id": run_id,
        "requested_physical_index": requested_physical_index,
        "hardware_preflight_path": str(output_path),
        "hardware_preflight_sha256": snapshot_sha,
        "status": "BLOCKED",
        "reason_code": reason_code,
        "terminal_status": "BLOCKED",
        "terminal_reason_code": reason_code,
        "scientific_result": "NO_SCIENTIFIC_RESULT",
    }
    _atomic_json(
        decision_path,
        payload,
        validator=lambda staged: _validate_preflight_decision_payload(
            staged,
            expected_snapshot_path=output_path,
            expected_authorization_commit=authorization_commit,
            expected_authorization_tree=authorization_tree,
            expected_run_id=run_id,
        ),
    )
    return {
        "preflight_decision_path": str(decision_path),
        "preflight_decision_sha256": sha256_file(decision_path),
    }


def create_hardware_preflight(
    *,
    output_path: Path,
    requested_physical_index: int,
    project_commit: str = IMPLEMENTATION_ANCHOR_COMMIT,
    project_tree: str = IMPLEMENTATION_ANCHOR_TREE,
    scope: str = SCIENTIFIC_MVE,
    stage: str = SCIENTIFIC_MVE,
    schedule_sha256: str | None = None,
    authorization_commit: str | None = None,
    authorization_tree: str | None = None,
    sample_interval_seconds: float = 5.0,
    sampler: Callable[[int], Mapping[str, object]] = nvidia_smi_hardware_sample,
    sleeper: Callable[[float], None] = time.sleep,
    probe_runner: Callable[[str, int, Mapping[str, object]], Mapping[str, object]] = _default_torch_probe,
) -> dict[str, object]:
    if not isinstance(requested_physical_index, int) or isinstance(requested_physical_index, bool) or requested_physical_index < 0:
        raise V4ExecutionError("V4_GPU_INDEX_INVALID")
    if float(sample_interval_seconds) != FORMAL_SAMPLE_INTERVAL_SECONDS:
        raise V4ExecutionError("V4_GPU_SAMPLE_INTERVAL_MUST_BE_5_SECONDS")
    resolved_authorization_commit, resolved_authorization_tree = _resolve_authorization_revision(authorization_commit, authorization_tree)
    run_id = uuid4().hex
    samples: list[dict[str, object]] = []
    try:
        samples.append(dict(sampler(requested_physical_index)))
        sleeper(float(sample_interval_seconds))
        samples.append(dict(sampler(requested_physical_index)))
        decision = _evaluate_basic(samples)
    except V4ExecutionError as exc:
        decision = _fail(str(exc), error=str(exc))
    except Exception as exc:
        decision = _fail("V4_GPU_BASIC_SAMPLE_UNAVAILABLE", error=str(exc))
    probes: list[dict[str, object]] = []
    if decision["status"] == "PASS":
        try:
            probes = [dict(probe_runner(model, requested_physical_index, samples[-1])) for model in SCIENTIFIC_MODELS]
            decision = _evaluate_probes(probes, samples[-1])
        except V4ExecutionError as exc:
            decision = _fail(str(exc), error=str(exc))
        except Exception as exc:
            decision = _fail("V4_GPU_TORCH_PROBE_FAILED", error=str(exc))
    selected = samples[-1] if samples else {}
    process_rows = _normalize_active_compute_process_rows(samples)
    if process_rows is None:
        raise V4ExecutionError(
            "V4_GPU_ACTIVE_COMPUTE_PROCESS_IDENTITY_INCOMPLETE"
        )
    evidence_process_count = max((len(rows) for rows in process_rows), default=0)
    receipt_path = output_path.with_name("v4-execution-receipt.json")
    if decision["status"] != "PASS":
        _remove_owned_preflight_siblings(output_path)
    snapshot = {
        "schema_version": V4_HARDWARE_PREFLIGHT_SCHEMA_VERSION,
        "status": decision["status"],
        "reason_code": decision["reason_code"],
        "project_commit": project_commit,
        "project_tree": project_tree,
        "scope": scope,
        "stage": stage,
        "authorization_commit": resolved_authorization_commit,
        "authorization_tree": resolved_authorization_tree,
        "run_id": run_id,
        "requested_physical_index": requested_physical_index,
        "resolved_physical_index": selected.get("resolved_physical_index"),
        "sample_interval_seconds": float(sample_interval_seconds),
        "visible_gpu_count": 1 if decision["status"] == "PASS" else 0,
        "compute_process_count": evidence_process_count,
        "stable_sample_count": len(samples),
        "environment": {"CUDA_VISIBLE_DEVICES": str(requested_physical_index), "GEORELIAB_PHYSICAL_GPU_DEVICE": f"cuda:{requested_physical_index}"},
        "devices": [{
            "physical_index": selected.get("resolved_physical_index"),
            "uuid": selected.get("device_uuid"),
            "model": selected.get("device_model"),
            "driver_version": selected.get("driver_version"),
            "total_memory_bytes": selected.get("total_memory_bytes"),
            "free_memory_bytes": selected.get("free_memory_bytes"),
            "used_memory_bytes": selected.get("used_memory_bytes"),
            "utilization_gpu_percent": selected.get("utilization_gpu_percent"),
            "temperature_c": selected.get("temperature_c"),
            "cuda_runtime": selected.get("cuda_runtime"),
            "mig_mode": selected.get("mig_mode"),
            "ecc_health": selected.get("ecc_health"),
            "compute_process_count": evidence_process_count,
        }] if selected else [],
        "samples": [{"sample_index": index, **sample} for index, sample in enumerate(samples)],
        "model_environment_probes": probes,
    }
    if "error" in decision:
        snapshot["error"] = decision["error"]
    _atomic_json(output_path, snapshot, validator=_validate_preflight_payload)
    result: dict[str, object] = {
        "status": snapshot["status"],
        "reason_code": snapshot["reason_code"],
        "hardware_preflight_path": str(output_path),
        "hardware_preflight_sha256": sha256_file(output_path),
    }
    if snapshot["status"] != "PASS":
        result.update(_write_blocked_preflight_decision(
            output_path=output_path,
            requested_physical_index=requested_physical_index,
            reason_code=str(snapshot["reason_code"]),
            authorization_commit=resolved_authorization_commit,
            authorization_tree=resolved_authorization_tree,
            run_id=run_id,
        ))
        return result
    try:
        _preflight_decision_path(output_path).unlink()
    except FileNotFoundError:
        pass
    receipt = V4ExecutionReceipt(
        explicit_user_selection=True,
        project_commit=project_commit,
        project_tree=project_tree,
        protocol_id=V4_PROTOCOL_ID,
        protocol_sha256=V4_PROTOCOL_SHA256,
        scope=scope,
        stage=stage,
        schedule_sha256=schedule_sha256,
        hardware_preflight_path=str(output_path),
        hardware_preflight_sha256=str(result["hardware_preflight_sha256"]),
        requested_physical_index=requested_physical_index,
        resolved_physical_index=int(selected["resolved_physical_index"]),
        device_uuid=str(selected["device_uuid"]),
        device_model=str(selected["device_model"]),
        driver_version=str(selected["driver_version"]),
        total_memory_bytes=int(selected["total_memory_bytes"]),
        max_concurrent_gpus=1,
        sequential_model_execution=True,
        sequential_unit_execution=True,
        fallback_allowed=False,
        device_switch_allowed=False,
        retry_allowed=False,
        nonce=uuid4().hex,
    )
    _atomic_json(receipt_path, receipt.to_dict(), validator=_validate_receipt_payload)
    result["receipt_path"] = str(receipt_path)
    result["receipt_sha256"] = sha256_file(receipt_path)
    return result


def _rich_preflight_receipt_args(preflight_path: Path, receipt: V4ExecutionReceipt) -> dict[str, object]:
    payload = load_json(preflight_path)
    if payload.get("schema_version") != V4_HARDWARE_PREFLIGHT_SCHEMA_VERSION or payload.get("status") != "PASS":
        raise V4ExecutionError("V4_GPU_PREFLIGHT_NOT_PASS")
    return {
        "project_commit": receipt.project_commit,
        "project_tree": receipt.project_tree,
        "scope": receipt.scope,
        "stage": receipt.stage,
        "schedule_sha256": receipt.schedule_sha256,
        "hardware_preflight_path": preflight_path,
        "hardware_preflight_sha256": receipt.hardware_preflight_sha256,
        "requested_physical_index": receipt.requested_physical_index,
        "visible_gpu_count": payload.get("visible_gpu_count"),
        "active_gpu_count": payload.get("compute_process_count"),
        "lock_active": False,
    }


def validate_rich_preflight_for_receipt(preflight_path: Path, receipt: V4ExecutionReceipt) -> None:
    payload = load_json(preflight_path)
    if payload.get("schema_version") != V4_HARDWARE_PREFLIGHT_SCHEMA_VERSION or payload.get("status") != "PASS":
        raise V4ExecutionError("V4_GPU_PREFLIGHT_NOT_PASS")
    if payload.get("project_commit") != receipt.project_commit or payload.get("project_tree") != receipt.project_tree:
        raise V4ExecutionError("V4_GPU_RECEIPT_GIT_MISMATCH")
    if payload.get("requested_physical_index") != receipt.requested_physical_index or payload.get("resolved_physical_index") != receipt.resolved_physical_index:
        raise V4ExecutionError("V4_GPU_RECEIPT_DEVICE_MISMATCH")
    devices = payload.get("devices")
    if not isinstance(devices, list) or len(devices) != 1:
        raise V4ExecutionError("V4_GPU_RECEIPT_SINGLE_VISIBLE_GPU_REQUIRED")
    device = devices[0]
    if not isinstance(device, Mapping):
        raise V4ExecutionError("V4_GPU_RECEIPT_DEVICE_MISMATCH")
    if device.get("uuid") != receipt.device_uuid or device.get("model") != receipt.device_model or device.get("driver_version") != receipt.driver_version or device.get("total_memory_bytes") != receipt.total_memory_bytes:
        raise V4ExecutionError("V4_GPU_RECEIPT_DEVICE_MISMATCH")
    decision = _evaluate_basic(payload.get("samples", ()))
    if decision["status"] != "PASS":
        raise V4ExecutionError(str(decision["reason_code"]))
    probe_decision = _evaluate_probes(payload.get("model_environment_probes", ()), payload["samples"][-1])
    if probe_decision["status"] != "PASS":
        raise V4ExecutionError(str(probe_decision["reason_code"]))


def _load_receipt(receipt_path: Path, expected_sha256: str | None = None) -> V4ExecutionReceipt:
    if expected_sha256 is not None and sha256_file(receipt_path) != expected_sha256:
        raise V4ExecutionError("V4_RECEIPT_TAMPER")
    return V4ExecutionReceipt.from_mapping(load_json(receipt_path))


def _validate_receipt_contract(receipt: V4ExecutionReceipt) -> None:
    if receipt.explicit_user_selection is not True:
        raise V4ExecutionError("GPU_SELECTION_REQUIRED")
    if receipt.protocol_id != V4_PROTOCOL_ID or receipt.protocol_sha256 != V4_PROTOCOL_SHA256:
        raise V4ExecutionError("V4_GPU_RECEIPT_PROTOCOL_MISMATCH")
    if receipt.scope != SCIENTIFIC_MVE or receipt.stage != SCIENTIFIC_MVE:
        raise V4ExecutionError("V4_GPU_RECEIPT_SCOPE_MISMATCH")
    if receipt.max_concurrent_gpus != 1:
        raise V4ExecutionError("V4_GPU_RECEIPT_SINGLE_GPU_REQUIRED")
    if receipt.sequential_model_execution is not True or receipt.sequential_unit_execution is not True:
        raise V4ExecutionError("V4_GPU_RECEIPT_SEQUENTIAL_REQUIRED")
    if receipt.fallback_allowed is not False or receipt.device_switch_allowed is not False or receipt.retry_allowed is not False:
        raise V4ExecutionError("V4_GPU_RECEIPT_NO_FALLBACK_REQUIRED")
    if receipt.device_model != AUTHORIZED_GPU_MODEL:
        raise V4ExecutionError("V4_GPU_MODEL_NOT_AUTHORIZED")


def _validate_schedule_path(schedule_path: Path, expected_sha256: str) -> None:
    if sha256_file(schedule_path) != expected_sha256:
        raise V4ExecutionError("V4_SCHEDULE_TAMPER")
    schedule = validate_scientific_schedule(parse_scientific_schedule(schedule_path.read_text(encoding="utf-8")))
    if schedule.models != SCIENTIFIC_MODELS or len(schedule.units) != 400:
        raise V4ExecutionError("V4_AUTHORIZATION_SCOPE_EXPANDED")


def _inventory_from_manifest(root: Path, manifest_path: Path) -> tuple[tuple[str, str, str], ...]:
    manifest = load_json(manifest_path)
    if set(manifest) != set(AUTHORIZED_RESOURCE_KEYS):
        raise V4ExecutionError("V4_RESOURCE_INVENTORY_SCHEMA_REQUIRED")
    rows: list[tuple[str, str, str]] = []
    for key in AUTHORIZED_RESOURCE_KEYS:
        value = manifest[key]
        if isinstance(value, str):
            path = _resolve_under_root(root, Path(value), must_exist=True)
            expected_sha = sha256_file(path)
        elif isinstance(value, Mapping):
            path = _resolve_under_root(root, Path(str(value.get("path"))), must_exist=True)
            expected_sha = str(value.get("sha256"))
            if not _is_sha(expected_sha) or sha256_file(path) != expected_sha:
                raise V4ExecutionError("V4_RESOURCE_TAMPER")
        else:
            raise V4ExecutionError("V4_RESOURCE_INVENTORY_SCHEMA_REQUIRED")
        rows.append((key, str(path), expected_sha))
    return tuple(rows)


def _authorized_scope() -> dict[str, object]:
    return {
        "model_set": list(SCIENTIFIC_MODELS),
        "dataset": "DTU",
        "test_scene_ids": list(TEST_SCENE_IDS),
        "state_ids": list(SCIENTIFIC_STATES),
        "lighting_states": list(LIGHTING_STATES),
        "fog_states": list(FOG_STATES),
        "model_independent_states": 200,
        "execution_units": 400,
        "l3_reused_once": True,
        "fog_model": "Koschmieder",
        "severity_family": "beta-only",
        "primary_endpoint": "Pose",
        "supporting_endpoints": ["Fusion", "F-score"],
        "authorized_stop_gpu_seconds": GPU_TARGET_SECONDS,
        "authorized_stop_logical_bytes": BYTE_TARGET,
        "authorized_stop_allocated_bytes": BYTE_TARGET,
        "hard_ceiling_gpu_seconds": GPU_CATASTROPHE_SECONDS,
        "hard_ceiling_logical_bytes": BYTE_CATASTROPHE,
        "hard_ceiling_allocated_bytes": BYTE_CATASTROPHE,
        "forbidden": [
            "UAVLight",
            "v4.1",
            "third model",
            "second corruption family",
            "seed expansion",
            "severity expansion",
            "grid expansion",
            "automatic fallback",
            "parallel GPU execution",
        ],
        "max_concurrency": 1,
        "sequential_models": True,
        "fallback_allowed": False,
        "device_switch_allowed": False,
        "retry_allowed": False,
    }


def create_execution_authorization(
    *,
    root: Path,
    receipt_path: Path,
    resource_inventory_path: Path,
    run_root: Path,
    artifact_root: Path,
    final_evidence_path: Path,
    output_path: Path,
) -> dict[str, object]:
    try:
        output_path.unlink()
    except FileNotFoundError:
        pass
    try:
        output_path.with_name(output_path.name + ".partial").unlink()
    except FileNotFoundError:
        pass
    resolved_root = root.resolve()
    receipt = _load_receipt(receipt_path)
    if receipt.project_commit != IMPLEMENTATION_ANCHOR_COMMIT or receipt.project_tree != IMPLEMENTATION_ANCHOR_TREE:
        raise V4ExecutionError("V4_AUTHORIZATION_STALE_ANCHOR")
    _validate_receipt_contract(receipt)
    if receipt.schedule_sha256 is None:
        raise V4ExecutionError("V4_AUTHORIZATION_SCHEDULE_REQUIRED")
    preflight_path = Path(receipt.hardware_preflight_path)
    if not preflight_path.is_absolute():
        preflight_path = receipt_path.parent / preflight_path
    validate_rich_preflight_for_receipt(preflight_path, receipt)
    _validate_schedule_path(_resolve_under_root(resolved_root, Path(load_json(resource_inventory_path)["scientific_schedule_400"]["path"] if isinstance(load_json(resource_inventory_path).get("scientific_schedule_400"), Mapping) else load_json(resource_inventory_path)["scientific_schedule_400"]), must_exist=True), receipt.schedule_sha256)
    inventory = _inventory_from_manifest(resolved_root, resource_inventory_path)
    authorized = V4ExecutionAuthorization(
        receipt_path=str(receipt_path.resolve()),
        receipt_sha256=sha256_file(receipt_path),
        hardware_preflight_path=str(preflight_path.resolve()),
        hardware_preflight_sha256=sha256_file(preflight_path),
        implementation_commit=receipt.project_commit,
        implementation_tree=receipt.project_tree,
        protocol_id=V4_PROTOCOL_ID,
        protocol_sha256=V4_PROTOCOL_SHA256,
        schedule_sha256=receipt.schedule_sha256,
        resource_inventory=inventory,
        root=str(resolved_root),
        run_root=str(_resolve_under_root(resolved_root, run_root)),
        artifact_root=str(_resolve_under_root(resolved_root, artifact_root)),
        final_evidence_path=str(_resolve_under_root(resolved_root, final_evidence_path)),
        finalizer=AUTHORIZED_FINALIZER,
    )
    payload = authorized.to_dict()
    _atomic_json(output_path, payload, validator=_validate_authorization_payload)
    return {"status": "PASS", "authorization_path": str(output_path), "authorization_sha256": sha256_file(output_path)}


def validate_execution_authorization(path: Path) -> dict[str, object]:
    payload = load_json(path)
    if payload.get("schema_version") != V4_EXECUTION_AUTHORIZATION_SCHEMA_VERSION:
        raise V4ExecutionError("V4_AUTHORIZATION_SCHEMA_REQUIRED")
    expected_sha = payload.get("authorization_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "authorization_sha256"}
    if not _is_sha(expected_sha) or _sha_json(unsigned) != expected_sha:
        raise V4ExecutionError("V4_AUTHORIZATION_TAMPER")
    if payload.get("implementation_commit") != IMPLEMENTATION_ANCHOR_COMMIT or payload.get("implementation_tree") != IMPLEMENTATION_ANCHOR_TREE:
        raise V4ExecutionError("V4_AUTHORIZATION_STALE_ANCHOR")
    if payload.get("protocol_id") != V4_PROTOCOL_ID or payload.get("protocol_sha256") != V4_PROTOCOL_SHA256:
        raise V4ExecutionError("V4_AUTHORIZATION_PROTOCOL_MISMATCH")
    if payload.get("finalizer") != AUTHORIZED_FINALIZER:
        raise V4ExecutionError("V4_AUTHORIZATION_FINALIZER_MISMATCH")
    root = Path(str(payload.get("root"))).resolve()
    _resolve_under_root(root, Path(str(payload.get("run_root"))), must_exist=True)
    _resolve_under_root(root, Path(str(payload.get("artifact_root"))), must_exist=True)
    _resolve_under_root(root, Path(str(payload.get("final_evidence_path"))))
    scope = payload.get("authorized_scope")
    if scope != _authorized_scope():
        raise V4ExecutionError("V4_AUTHORIZATION_SCOPE_EXPANDED")
    receipt_path = Path(str(payload.get("receipt_path")))
    receipt = _load_receipt(receipt_path, str(payload.get("receipt_sha256")))
    if receipt.project_commit != payload.get("implementation_commit") or receipt.project_tree != payload.get("implementation_tree") or receipt.schedule_sha256 != payload.get("schedule_sha256"):
        raise V4ExecutionError("V4_AUTHORIZATION_RECEIPT_MISMATCH")
    _validate_receipt_contract(receipt)
    preflight_path = Path(str(payload.get("hardware_preflight_path")))
    if sha256_file(preflight_path) != payload.get("hardware_preflight_sha256"):
        raise V4ExecutionError("V4_GPU_PREFLIGHT_TAMPER")
    validate_rich_preflight_for_receipt(preflight_path, receipt)
    for row in payload.get("resource_inventory", ()):
        if not isinstance(row, Mapping) or row.get("key") not in AUTHORIZED_RESOURCE_KEYS:
            raise V4ExecutionError("V4_RESOURCE_INVENTORY_SCHEMA_REQUIRED")
        resource_path = _resolve_under_root(root, Path(str(row.get("path"))), must_exist=True)
        digest = row.get("sha256")
        if not _is_sha(digest) or sha256_file(resource_path) != digest:
            raise V4ExecutionError("V4_RESOURCE_TAMPER")
    if [row.get("key") for row in payload.get("resource_inventory", ())] != list(AUTHORIZED_RESOURCE_KEYS):
        raise V4ExecutionError("V4_RESOURCE_INVENTORY_SCHEMA_REQUIRED")
    return {"status": "PASS", "authorization_path": str(path), "authorization_sha256": sha256_file(path)}
