from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

import georeliab_mve.v4_attempt03_authorization as attempt03

from georeliab_mve.v4_attempt03_authorization import (
    ATTEMPT_ID,
    AUTHORIZED_GPU_MODEL,
    INVENTORY_SCHEMA,
    SAMPLE_INTERVAL_SECONDS,
    CLOSURE_RECEIPT_FILE_SHA256,
    EXPECTED_SET_FILE_SHA256,
    GROUP_INDEX_SHA256,
    MANIFEST_FILE_SHA256,
    MATERIALIZE_RECEIPT_FILE_SHA256,
    ORDERED_MEMBER_LIST_SHA256,
    SCHEDULE_BINDING_SHA256,
    SCHEDULE_FILE_SHA256,
    _attempt_path,
    _validate_authorization_payload,
    create_attempt03_execution_authorization,
    create_attempt03_gpu_preflight,
    nvidia_smi_attempt03_inventory,
    select_attempt03_gpu,
)
from georeliab_mve.v4_execution import V4ExecutionError


GIB = 1024 * 1024 * 1024


def test_attempt03_closure_digests_are_frozen() -> None:
    assert SCHEDULE_FILE_SHA256 == (
        '47ed0464409d0189cb301930ecaf8db5b40b540ef0c5459dfea01fd92444a6c3'
    )
    assert EXPECTED_SET_FILE_SHA256 == (
        'b64139b0c89b6a2dd5b94982d372daee3746c3852a8f5a4a597a8d4ff456d450'
    )
    assert MATERIALIZE_RECEIPT_FILE_SHA256 == (
        '9af0fc7e832466ffd2de5c8a4fecf2f67513804797342edca3cf01ad67e889a9'
    )
    assert MANIFEST_FILE_SHA256 == (
        '1634e75bd09ca2b446ef32d768eddcd7fc547473f19c304b53b5711d7ac53dbb'
    )
    assert CLOSURE_RECEIPT_FILE_SHA256 == (
        '7a4d77966102812313a89060af36decb0fe5add6291ca018c406e5abe3df30b5'
    )
    assert ORDERED_MEMBER_LIST_SHA256 == (
        '521ef283be964c195f77acc55a1e8458c16302a3d74e1bc2cf82c6582d7e0377'
    )
    assert GROUP_INDEX_SHA256 == (
        'bfaa2423b554518b6648e2077cbab77fe568025b1a05ad91c7674cd271145017'
    )
    assert SCHEDULE_BINDING_SHA256 == (
        '42785287dbc4be2854bb0bfb3df3881f944942a26fdc422e37905b243a09e930'
    )


def _device(
    *,
    index: int = 0,
    uuid: str = 'GPU-0000',
    pci: str = '00000000:17:00.0',
    total: int = 80 * GIB,
    free: int = 79 * GIB,
) -> dict[str, object]:
    return {
        'index': index,
        'uuid': uuid,
        'pci_bus_id': pci,
        'model': AUTHORIZED_GPU_MODEL,
        'total_memory_bytes': total,
        'free_memory_bytes': free,
        'used_memory_bytes': total - free,
        'utilization_gpu_percent': 0,
        'temperature_c': 31,
        'performance_state': 'P0',
        'compute_mode': 'Default',
        'mig_mode': 'Disabled',
        'ecc_uncorrected_volatile_total': 0,
        'ecc_health': 'OK',
        'driver_version': '580.95.05',
        'cuda_runtime': '13.0',
        'compute_processes': [],
    }


def _sample(
    number: int, devices: list[dict[str, object]]
) -> dict[str, object]:
    return {
        'schema_version': INVENTORY_SCHEMA,
        'attempt_id': ATTEMPT_ID,
        'hostname': 'a100-smli',
        'timestamp_utc': f'2026-08-02T00:00:{number * 5:02d}Z',
        'driver_version': '580.95.05',
        'cuda_runtime': '13.0',
        'devices': deepcopy(devices),
    }


def _samples(
    devices: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [_sample(index, devices) for index in range(3)]


def _resign(payload: dict[str, object], field: str) -> None:
    payload.pop(field, None)
    payload[field] = attempt03._sha_json(payload)


def _valid_resource_payload(tmp_path: Path) -> dict[str, object]:
    runtime_root = tmp_path.resolve()
    attempt_root = runtime_root / 'authorization-attempts' / ATTEMPT_ID
    payload: dict[str, object] = {
        'schema_version': attempt03.RESOURCE_SCHEMA,
        'attempt_id': ATTEMPT_ID,
        'status': 'PASS',
        'reason_code': 'V4_RESOURCE_CLOSURE_REVALIDATED',
        'tooling_commit': 'a' * 40,
        'tooling_tree': 'b' * 40,
        'scientific_anchor_commit': attempt03.SCIENTIFIC_ANCHOR_COMMIT,
        'scientific_anchor_tree': attempt03.SCIENTIFIC_ANCHOR_TREE,
        'protocol_id': attempt03.V4_PROTOCOL_ID,
        'protocol_sha256': attempt03.V4_PROTOCOL_SHA256,
        'runtime_root': str(runtime_root),
        'attempt_root': str(attempt_root),
        'closure_files': {
            label: {'path': f'/fixed/{label}', 'sha256': digest}
            for label, digest in attempt03._EXPECTED_FILE_DIGESTS.items()
        },
        'schedule_file_sha256': SCHEDULE_FILE_SHA256,
        'ordered_member_list_sha256': ORDERED_MEMBER_LIST_SHA256,
        'group_index_sha256': GROUP_INDEX_SHA256,
        'schedule_binding_sha256': SCHEDULE_BINDING_SHA256,
        'member_count': 960,
        'group_count': 160,
        'member_illuminations': ['L1', 'L2', 'L4', 'L5', 'L6', 'L7'],
        'l3_role': 'REFERENCE_EXCLUDED_FROM_RECTIFIED_MEMBER_CLOSURE',
        'missing_count': 0,
        'duplicate_count': 0,
        'orphan_count': 0,
        'symlink_count': 0,
        'partial_count': 0,
        'science_lock': {
            'status': attempt03.GEORELIAB_V4_PROTOCOL_READY,
            'execution_status': attempt03.GPU_SELECTION_REQUIRED,
            'scientific_result_status': attempt03.NO_SCIENTIFIC_RESULT,
            'protocol_id': attempt03.V4_PROTOCOL_ID,
            'protocol_sha256': attempt03.V4_PROTOCOL_SHA256,
        },
        'numerical_anchor_paths': [
            {'path': path, 'sha256': 'c' * 64}
            for path in attempt03._NUMERICAL_ANCHOR_PATHS
        ],
        'resource_bindings': {},
        'nvidia_smi_invocations': 0,
        'torch_probe_invocations': 0,
        'model_loads': 0,
        'model_forwards': 0,
        'scientific_result': 'NO_SCIENTIFIC_RESULT',
    }
    _resign(payload, 'resource_revalidation_sha256')
    return payload


def test_attempt03_selects_largest_idle_gpu_by_stable_identity() -> None:
    smaller = _device(
        index=0,
        uuid='GPU-bbbb',
        pci='00000000:17:00.0',
        total=40 * GIB,
        free=39 * GIB,
    )
    larger = _device(
        index=1,
        uuid='GPU-aaaa',
        pci='00000000:65:00.0',
    )

    result = select_attempt03_gpu(_samples([smaller, larger]))

    assert result['status'] == 'PASS'
    assert result['selected_gpu']['uuid'] == 'GPU-aaaa'
    assert result['selected_gpu']['index'] == 1
    assert result['selected_gpu']['pci_bus_id'] == '00000000:65:00.0'


@pytest.mark.parametrize(
    ('field', 'value', 'reason'),
    [
        ('utilization_gpu_percent', 1, 'V4_ATTEMPT03_GPU_UNEXPLAINED_ACTIVITY'),
        ('mig_mode', 'Enabled', 'V4_ATTEMPT03_GPU_MIG_ENABLED'),
        (
            'compute_mode',
            'Exclusive_Process',
            'V4_ATTEMPT03_GPU_COMPUTE_MODE_NOT_DEFAULT',
        ),
        ('ecc_health', 'ERROR', 'V4_ATTEMPT03_GPU_ECC_HEALTH_ERROR'),
        (
            'free_memory_bytes',
            8 * GIB,
            'V4_ATTEMPT03_GPU_FREE_MEMORY_INSUFFICIENT',
        ),
    ],
)
def test_attempt03_rejects_ineligible_hardware(
    field: str, value: object, reason: str
) -> None:
    device = _device()
    device[field] = value
    if field == 'free_memory_bytes':
        device['used_memory_bytes'] = device['total_memory_bytes'] - value
    if field == 'ecc_health':
        device['ecc_uncorrected_volatile_total'] = 1

    result = select_attempt03_gpu(_samples([device]))

    assert result['status'] == 'FAIL'
    assert result['candidate_evaluations'][0]['reason_code'] == reason


def test_attempt03_rejects_uuid_index_or_pci_drift() -> None:
    samples = _samples([_device()])
    samples[1]['devices'][0]['pci_bus_id'] = '00000000:18:00.0'

    result = select_attempt03_gpu(samples)

    assert result['status'] == 'FAIL'
    assert result['candidate_evaluations'][0]['reason_code'] == (
        'V4_ATTEMPT03_GPU_MAPPING_DRIFT'
    )


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('uuid', 'GPU-drifted'),
        ('index', 7),
        ('model', 'NVIDIA A100-SXM4-80GB'),
        ('free_memory_bytes', 78 * GIB),
        ('driver_version', '581.0'),
        ('cuda_runtime', '13.1'),
    ],
)
def test_attempt03_rejects_identity_or_memory_drift(
    field: str, value: object
) -> None:
    samples = _samples([_device()])
    samples[1]['devices'][0][field] = value
    if field == 'free_memory_bytes':
        samples[1]['devices'][0]['used_memory_bytes'] = 2 * GIB
    if field in {'driver_version', 'cuda_runtime'}:
        samples[1][field] = value

    result = select_attempt03_gpu(samples)

    assert result['status'] == 'FAIL'
    assert all(
        item['reason_code'] == 'V4_ATTEMPT03_GPU_MAPPING_DRIFT'
        for item in result['candidate_evaluations']
    )


def test_attempt03_rejects_process_state_change_between_samples() -> None:
    samples = _samples([_device()])
    samples[1]['devices'][0]['compute_processes'] = [
        {
            'pid': 19,
            'owner': 'smli',
            'cwd': '/srv/private/smli/other',
            'cmdline': 'python job.py',
            'process_name': 'python',
            'used_memory_bytes': GIB,
        }
    ]

    result = select_attempt03_gpu(samples)

    assert result['status'] == 'FAIL'
    assert result['candidate_evaluations'][0]['reason_code'] == (
        'V4_ATTEMPT03_GPU_COMPUTE_PROCESS_PRESENT'
    )


@pytest.mark.parametrize('missing', ['owner', 'cwd', 'cmdline', 'process_name'])
def test_attempt03_rejects_incomplete_process_identity(
    missing: str,
) -> None:
    process = {
        'pid': 23,
        'owner': 'smli',
        'cwd': '/srv/private/smli/other',
        'cmdline': 'python job.py',
        'process_name': 'python',
        'used_memory_bytes': GIB,
    }
    process[missing] = ''
    device = _device()
    device['compute_processes'] = [process]

    with pytest.raises(
        V4ExecutionError,
        match='V4_ATTEMPT03_GPU_PROCESS_IDENTITY_UNPROVEN',
    ):
        select_attempt03_gpu(_samples([device]))


def test_attempt03_rejects_samples_less_than_five_seconds_apart() -> None:
    samples = _samples([_device()])
    samples[1]['timestamp_utc'] = '2026-08-02T00:00:01Z'

    with pytest.raises(
        V4ExecutionError,
        match='V4_ATTEMPT03_GPU_SAMPLE_INDEPENDENCE_UNPROVEN',
    ):
        select_attempt03_gpu(samples)


def test_attempt03_rejects_any_compute_process() -> None:
    device = _device()
    device['compute_processes'] = [
        {
            'pid': 7,
            'owner': 'smli',
            'cwd': '/srv/private/smli/other',
            'cmdline': 'python train.py',
            'process_name': 'python',
            'used_memory_bytes': GIB,
        }
    ]

    result = select_attempt03_gpu(_samples([device]))

    assert result['status'] == 'FAIL'
    assert result['candidate_evaluations'][0]['reason_code'] == (
        'V4_ATTEMPT03_GPU_COMPUTE_PROCESS_PRESENT'
    )


def test_attempt03_rejects_unproven_process_owner() -> None:
    gpu_row = (
        '0, GPU-0000, 00000000:17:00.0, NVIDIA A100 80GB PCIe, '
        '81920, 80000, 1920, 0, 31, P0, Default, Disabled, 0, 580.95.05'
    )

    def runner(command) -> str:
        joined = ' '.join(command)
        if '--query-gpu=' in joined:
            return gpu_row
        if '--query-compute-apps=' in joined:
            return 'GPU-0000, 17, python, 1024'
        return '| CUDA Version: 13.0 |'

    with pytest.raises(
        V4ExecutionError,
        match='V4_ATTEMPT03_GPU_PROCESS_IDENTITY_UNPROVEN',
    ):
        nvidia_smi_attempt03_inventory(
            command_runner=runner,
            process_resolver=lambda _pid, _name: {
                'owner': '',
                'cwd': '/srv/private/smli/x',
                'cmdline': 'python x.py',
            },
        )


def test_resource_gate_fails_before_sampler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / 'authorization-attempts' / ATTEMPT_ID / 'resource.json'
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{}', encoding='utf-8')
    calls = 0

    def blocked(_path: Path) -> dict[str, object]:
        raise V4ExecutionError('V4_RESOURCE_CLOSURE_REVALIDATION_FAILED')

    def sampler() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _sample(calls, [_device()])

    monkeypatch.setattr(
        'georeliab_mve.v4_attempt03_authorization.'
        'validate_attempt03_resources',
        blocked,
    )
    with pytest.raises(
        V4ExecutionError,
        match='V4_RESOURCE_CLOSURE_REVALIDATION_FAILED',
    ):
        create_attempt03_gpu_preflight(
            worktree=tmp_path,
            resource_receipt_path=receipt,
            output_path=receipt.with_name('snapshot.json'),
            inventory_sampler=sampler,
            sleeper=lambda _: None,
            tooling_commit='a' * 40,
            tooling_tree='b' * 40,
        )
    assert calls == 0
    assert not receipt.with_name('snapshot.json').exists()


def test_resource_revalidation_failure_publishes_immutable_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = (
        tmp_path / 'authorization-attempts' / ATTEMPT_ID
        / 'v4-resource-revalidation.json'
    )
    monkeypatch.setattr(
        attempt03,
        '_tooling_revision',
        lambda _worktree: ('a' * 40, 'b' * 40),
    )
    monkeypatch.setattr(
        attempt03,
        '_revalidate_attempt03_resources_pass',
        lambda **_kwargs: (_ for _ in ()).throw(
            V4ExecutionError('V4_RECTIFIED_MEMBER_HASH_MISMATCH')
        ),
    )

    payload = attempt03.revalidate_attempt03_resources(
        worktree=tmp_path,
        runtime_root=tmp_path,
        rectified_root=tmp_path / 'materialized',
        closure_root=tmp_path / 'closure',
        overlay_path=tmp_path / 'overlay.toml',
        output_path=output,
    )

    assert output.is_file()
    assert payload['status'] == 'FAIL'
    assert payload['reason_code'] == (
        'V4_RESOURCE_CLOSURE_REVALIDATION_FAILED'
    )
    assert payload['underlying_reason_code'] == (
        'V4_RECTIFIED_MEMBER_HASH_MISMATCH'
    )
    assert payload['nvidia_smi_invocations'] == 0
    assert payload['torch_probe_invocations'] == 0
    assert payload['model_loads'] == 0
    assert payload['model_forwards'] == 0
    assert payload['gpu_inference_seconds'] == 0
    assert payload['pass_receipt_generated'] is False
    assert payload['execution_authorization_generated'] is False
    assert payload['scientific_result'] == 'NO_SCIENTIFIC_RESULT'
    attempt03._validate_resource_failure_payload(
        json.loads(output.read_text(encoding='utf-8'))
    )

    with pytest.raises(
        V4ExecutionError,
        match='V4_ATTEMPT03_ARTIFACT_COLLISION',
    ):
        attempt03.revalidate_attempt03_resources(
            worktree=tmp_path,
            runtime_root=tmp_path,
            rectified_root=tmp_path / 'materialized',
            closure_root=tmp_path / 'closure',
            overlay_path=tmp_path / 'overlay.toml',
            output_path=output,
        )


def test_attempt03_revalidation_rejects_alternate_output_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        V4ExecutionError,
        match='V4_ATTEMPT03_CANONICAL_OUTPUT_REQUIRED',
    ):
        attempt03.revalidate_attempt03_resources(
            worktree=tmp_path,
            runtime_root=tmp_path,
            rectified_root=tmp_path / 'materialized',
            closure_root=tmp_path / 'closure',
            overlay_path=tmp_path / 'overlay.toml',
            output_path=(
                tmp_path / 'authorization-attempts' / ATTEMPT_ID
                / 'alternate.json'
            ),
        )


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('member_count', 959),
        ('group_count', 159),
        ('member_illuminations', ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']),
        ('l3_role', 'IMPLICITLY_EXCLUDED'),
        ('missing_count', 1),
        ('nvidia_smi_invocations', 1),
        ('schedule_file_sha256', 'd' * 64),
    ],
)
def test_resource_receipt_rejects_resigned_semantic_tamper(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = _valid_resource_payload(tmp_path)
    payload[field] = value
    _resign(payload, 'resource_revalidation_sha256')

    with pytest.raises(V4ExecutionError):
        attempt03._validate_resource_payload(payload)


def test_resource_receipt_rejects_resigned_closure_digest_tamper(
    tmp_path: Path,
) -> None:
    payload = _valid_resource_payload(tmp_path)
    payload['closure_files']['manifest']['sha256'] = 'd' * 64
    _resign(payload, 'resource_revalidation_sha256')

    with pytest.raises(
        V4ExecutionError,
        match='V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER',
    ):
        attempt03._validate_resource_payload(payload)


def test_preflight_output_collision_stops_before_gpu_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / 'authorization-attempts' / ATTEMPT_ID
    root.mkdir(parents=True)
    resource = root / 'resource.json'
    resource.write_text('{}', encoding='utf-8')
    snapshot = root / 'snapshot.json'
    snapshot.write_text('{}', encoding='utf-8')
    calls = 0

    def sampler() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _sample(calls, [_device()])

    monkeypatch.setattr(
        attempt03,
        'validate_attempt03_resources',
        lambda _: pytest.fail('resource gate must not run after collision'),
    )

    with pytest.raises(
        V4ExecutionError,
        match='V4_ATTEMPT03_ARTIFACT_COLLISION',
    ):
        create_attempt03_gpu_preflight(
            worktree=tmp_path,
            resource_receipt_path=resource,
            output_path=snapshot,
            inventory_sampler=sampler,
            sleeper=lambda _: None,
            tooling_commit='a' * 40,
            tooling_tree='b' * 40,
        )
    assert calls == 0


def test_preflight_samples_exactly_three_without_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / 'authorization-attempts' / ATTEMPT_ID
    root.mkdir(parents=True)
    receipt = root / 'resource.json'
    receipt.write_text('{}', encoding='utf-8')
    commit = 'a' * 40
    tree = 'b' * 40
    monkeypatch.setattr(
        'georeliab_mve.v4_attempt03_authorization.'
        'validate_attempt03_resources',
        lambda _: {
            'tooling_commit': commit,
            'tooling_tree': tree,
            'runtime_root': str(tmp_path.resolve()),
            'attempt_root': str(root.resolve()),
        },
    )
    queued = iter(_samples([_device()]))
    sleeps: list[float] = []

    payload = create_attempt03_gpu_preflight(
        worktree=tmp_path,
        resource_receipt_path=receipt,
        output_path=root / 'snapshot.json',
        inventory_sampler=lambda: next(queued),
        sleeper=sleeps.append,
        tooling_commit=commit,
        tooling_tree=tree,
    )

    assert payload['status'] == 'PASS'
    assert payload['inventory_sample_count'] == 3
    assert payload['nvidia_smi_invocations_per_sample'] == 3
    assert payload['nvidia_smi_invocations'] == 9
    assert payload['torch_probe_invocations'] == 0
    assert payload['model_loads'] == 0
    assert payload['model_forwards'] == 0
    assert sleeps == [SAMPLE_INTERVAL_SECONDS, SAMPLE_INTERVAL_SECONDS]

    payload['nvidia_smi_invocations'] = 8
    _resign(payload, 'hardware_snapshot_sha256')
    with pytest.raises(
        V4ExecutionError,
        match='V4_ATTEMPT03_GPU_PROBE_SCOPE_VIOLATION',
    ):
        attempt03._validate_preflight_payload(payload)


def test_failed_preflight_cannot_create_receipt_or_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / 'authorization-attempts' / ATTEMPT_ID
    attempt.mkdir(parents=True)
    resource = attempt / 'resource.json'
    resource.write_text('{}', encoding='utf-8')
    commit = 'a' * 40
    tree = 'b' * 40
    resources = {
        'tooling_commit': commit,
        'tooling_tree': tree,
        'runtime_root': str(tmp_path.resolve()),
        'attempt_root': str(attempt.resolve()),
        'protocol_id': attempt03.V4_PROTOCOL_ID,
        'protocol_sha256': attempt03.V4_PROTOCOL_SHA256,
    }
    monkeypatch.setattr(
        attempt03, 'validate_attempt03_resources', lambda _: resources
    )
    busy = _device()
    busy['utilization_gpu_percent'] = 1
    queued = iter(_samples([busy]))
    snapshot = attempt / 'snapshot.json'

    preflight = create_attempt03_gpu_preflight(
        worktree=tmp_path,
        resource_receipt_path=resource,
        output_path=snapshot,
        inventory_sampler=lambda: next(queued),
        sleeper=lambda _: None,
        tooling_commit=commit,
        tooling_tree=tree,
    )
    assert preflight['status'] == 'FAIL'

    receipt = attempt / 'receipt.json'
    authorization = attempt / 'authorization.json'
    with pytest.raises(
        V4ExecutionError,
        match='V4_ATTEMPT03_PREFLIGHT_PASS_REQUIRED',
    ):
        create_attempt03_execution_authorization(
            worktree=tmp_path,
            runtime_root=tmp_path,
            resource_receipt_path=resource,
            hardware_snapshot_path=snapshot,
            receipt_path=receipt,
            authorization_path=authorization,
            run_root=tmp_path / 'runs' / ATTEMPT_ID,
            artifact_root=tmp_path / 'artifacts' / ATTEMPT_ID,
            gpu_ledger_path=tmp_path / 'logs' / 'attempt-03.jsonl',
            final_evidence_path=tmp_path / 'evidence' / 'attempt-03.json',
            tooling_commit=commit,
            tooling_tree=tree,
        )
    assert not receipt.exists()
    assert not authorization.exists()


def test_authorization_creation_does_not_dispatch_or_create_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / 'authorization-attempts' / ATTEMPT_ID
    attempt.mkdir(parents=True)
    resources_path = attempt / 'resources.json'
    snapshot_path = attempt / 'snapshot.json'
    resources_path.write_text('{}', encoding='utf-8')
    snapshot_path.write_text('{}', encoding='utf-8')
    commit = 'a' * 40
    tree = 'b' * 40
    selected = _device()
    real_validator = attempt03.validate_attempt03_execution_authorization
    monkeypatch.setattr(
        'georeliab_mve.v4_attempt03_authorization.'
        'validate_attempt03_resources',
        lambda _: {
            'tooling_commit': commit,
            'tooling_tree': tree,
            'runtime_root': str(tmp_path.resolve()),
            'attempt_root': str(attempt.resolve()),
        },
    )
    monkeypatch.setattr(
        'georeliab_mve.v4_attempt03_authorization.'
        'validate_attempt03_gpu_preflight',
        lambda _: {
            'tooling_commit': commit,
            'tooling_tree': tree,
            'status': 'PASS',
            'run_id': 'c' * 32,
            'selected_gpu': selected,
        },
    )
    monkeypatch.setattr(
        'georeliab_mve.v4_attempt03_authorization.'
        'validate_attempt03_execution_authorization',
        lambda path: json.loads(path.read_text(encoding='utf-8')),
    )
    ledger = tmp_path / 'ledgers' / 'gpu.jsonl'
    receipt_path = attempt / 'receipt.json'
    authorization_path = attempt / 'authorization.json'

    payload = create_attempt03_execution_authorization(
        worktree=tmp_path,
        runtime_root=tmp_path,
        resource_receipt_path=resources_path,
        hardware_snapshot_path=snapshot_path,
        receipt_path=receipt_path,
        authorization_path=authorization_path,
        run_root=tmp_path / 'runs',
        artifact_root=tmp_path / 'artifacts',
        gpu_ledger_path=ledger,
        final_evidence_path=tmp_path / 'evidence' / 'final.json',
        tooling_commit=commit,
        tooling_tree=tree,
    )

    assert payload['status'] == 'V4_MVE_EXECUTION_AUTHORIZED'
    assert payload['dispatcher_called'] is False
    assert payload['execution_lock_created'] is False
    assert payload['gpu_ledger_created'] is False
    assert payload['gpu_inference_seconds'] == 0
    assert not ledger.exists()

    assert real_validator(authorization_path)['status'] == (
        'V4_MVE_EXECUTION_AUTHORIZED'
    )

    receipt_payload = json.loads(receipt_path.read_text(encoding='utf-8'))
    receipt_payload['rectified_manifest_sha256'] = 'd' * 64
    _resign(receipt_payload, 'receipt_payload_sha256')
    with pytest.raises(
        V4ExecutionError,
        match='V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER',
    ):
        attempt03._validate_receipt_payload(receipt_payload)

    escaped = deepcopy(payload)
    escaped['runtime_paths']['artifact_root'] = str(
        (tmp_path.parent / 'escaped').resolve()
    )
    _resign(escaped, 'authorization_payload_sha256')
    with pytest.raises(
        V4ExecutionError,
        match='V4_ATTEMPT03_RUNTIME_PATH_ESCAPE',
    ):
        attempt03._validate_authorization_payload(escaped)

    closure_tamper = deepcopy(payload)
    closure_tamper['closure_digests']['manifest'] = 'd' * 64
    _resign(closure_tamper, 'authorization_payload_sha256')
    with pytest.raises(
        V4ExecutionError,
        match='V4_ATTEMPT03_AUTHORIZATION_SCOPE_MISMATCH',
    ):
        attempt03._validate_authorization_payload(closure_tamper)

    snapshot_path.write_text('{tampered:true}\n', encoding='utf-8')
    with pytest.raises(
        V4ExecutionError,
        match='V4_ATTEMPT03_PREFLIGHT_TAMPER',
    ):
        real_validator(authorization_path)


def test_authorization_scope_tamper_is_rejected() -> None:
    payload = {
        'schema_version': 'georeliab-v4-attempt-03-execution-authorization-1.0',
        'attempt_id': ATTEMPT_ID,
        'status': 'V4_MVE_EXECUTION_AUTHORIZED',
        'scientific_result': 'NO_SCIENTIFIC_RESULT',
        'authorized_scope': {},
        'budget': {},
        'finalizer': 'bad',
    }
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n'
    payload['authorization_payload_sha256'] = hashlib.sha256(
        raw.encode('ascii')
    ).hexdigest()

    with pytest.raises(V4ExecutionError):
        _validate_authorization_payload(payload)


@pytest.mark.parametrize('history', ['attempt-01', 'attempt-02'])
def test_attempt03_rejects_historical_namespace(
    tmp_path: Path, history: str
) -> None:
    with pytest.raises(
        V4ExecutionError,
        match='V4_ATTEMPT03_PATH_SCOPE_MISMATCH',
    ):
        _attempt_path(tmp_path / history / 'receipt.json')
