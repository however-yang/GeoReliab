from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path

import pytest

import georeliab_mve.v4_attempt04_authorization as attempt04
from georeliab_mve.v4_execution import V4ExecutionError


GIB = 1024 * 1024 * 1024


def _device(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        'index': 0,
        'uuid': 'GPU-attempt04',
        'pci_bus_id': '00000000:17:00.0',
        'model': 'NVIDIA A100 80GB PCIe',
        'total_memory_bytes': 80 * GIB,
        'free_memory_bytes': 79 * GIB,
        'used_memory_bytes': GIB,
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
    value.update(changes)
    return value


def _sample(number: int, devices: list[dict[str, object]]) -> dict[str, object]:
    return {
        'schema_version': attempt04.INVENTORY_SCHEMA,
        'attempt_id': attempt04.ATTEMPT_ID,
        'base_commit': attempt04.BASE_COMMIT,
        'base_tree': attempt04.BASE_TREE,
        'hostname': 'a100-smli',
        'timestamp_utc': f'2026-08-02T00:00:{number * 5:02d}Z',
        'driver_version': '580.95.05',
        'cuda_runtime': '13.0',
        'devices': deepcopy(devices),
    }


def _samples(devices: list[dict[str, object]]) -> list[dict[str, object]]:
    return [_sample(index, devices) for index in range(3)]


def test_attempt04_is_independent_and_bound_to_frozen_base() -> None:
    source = inspect.getsource(attempt04)
    assert 'v4_attempt03_authorization' not in source
    assert '._core' not in source
    assert attempt04.ATTEMPT_ID == 'attempt-04'
    assert attempt04.BASE_COMMIT == (
        '6e91a418983162d50a78a008fb41c540b8edf4c0'
    )
    assert attempt04.BASE_TREE == (
        '2f9b2641136ad864e3864eb92ad67ffc18e54056'
    )


def test_real_inventory_runner_preserves_base_identity() -> None:
    calls: list[tuple[str, ...]] = []

    def runner(command) -> str:
        command = tuple(command)
        calls.append(command)
        if any(value.startswith('--query-gpu=') for value in command):
            return (
                '0, GPU-real, 00000000:17:00.0, '
                'NVIDIA A100 80GB PCIe, 81920, 80896, 1024, '
                '0, 31, P0, Default, Disabled, 0, 580.95.05\n'
            )
        if '--query-compute-apps=gpu_uuid,pid,process_name,used_memory' in command:
            return ''
        return '| NVIDIA-SMI 580.95.05 Driver Version: 580.95.05 CUDA Version: 13.0 |\n'

    payload = attempt04.nvidia_smi_attempt04_inventory(
        command_runner=runner,
        process_resolver=lambda _pid, _name: pytest.fail(
            'no process identity should be requested'
        ),
    )
    assert len(calls) == 3
    assert payload['base_commit'] == attempt04.BASE_COMMIT
    assert payload['base_tree'] == attempt04.BASE_TREE
    samples = [deepcopy(payload) for _ in range(3)]
    for index, sample in enumerate(samples):
        sample['timestamp_utc'] = f'2026-08-02T00:00:{index * 5:02d}Z'
    result = attempt04.select_attempt04_gpu(samples)
    assert result['status'] == 'PASS'
    assert result['selected_gpu']['uuid'] == 'GPU-real'


def test_attempt04_selects_largest_idle_gpu_deterministically() -> None:
    small = _device(
        uuid='GPU-bbbb',
        total_memory_bytes=40 * GIB,
        free_memory_bytes=39 * GIB,
        used_memory_bytes=GIB,
    )
    large = _device(index=1, uuid='GPU-aaaa', pci_bus_id='00000000:65:00.0')
    result = attempt04.select_attempt04_gpu(_samples([small, large]))
    assert result['status'] == 'PASS'
    assert result['selected_gpu']['uuid'] == 'GPU-aaaa'


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('utilization_gpu_percent', 1),
        ('mig_mode', 'Enabled'),
        ('compute_mode', 'Exclusive_Process'),
        ('ecc_health', 'ERROR'),
        ('free_memory_bytes', 8 * GIB),
        ('model', 'NVIDIA H100 80GB HBM3'),
    ],
)
def test_attempt04_rejects_ineligible_gpu(field: str, value: object) -> None:
    device = _device(**{field: value})
    if field == 'free_memory_bytes':
        device['used_memory_bytes'] = 72 * GIB
    if field == 'ecc_health':
        device['ecc_uncorrected_volatile_total'] = 1
    result = attempt04.select_attempt04_gpu(_samples([device]))
    assert result['status'] == 'FAIL'
    assert result['selected_gpu'] is None


@pytest.mark.parametrize(
    'field', ['uuid', 'index', 'pci_bus_id', 'driver_version', 'cuda_runtime']
)
def test_attempt04_rejects_three_sample_identity_drift(field: str) -> None:
    samples = _samples([_device()])
    samples[1]['devices'][0][field] = 'drift' if field != 'index' else 8
    if field in {'driver_version', 'cuda_runtime'}:
        samples[1][field] = 'drift'
    if field == 'uuid':
        with pytest.raises(V4ExecutionError):
            attempt04.select_attempt04_gpu(samples)
    else:
        result = attempt04.select_attempt04_gpu(samples)
        assert result['status'] == 'FAIL'


def test_attempt04_rejects_process_state_drift() -> None:
    samples = _samples([_device()])
    samples[1]['devices'][0]['compute_processes'] = [
        {
            'pid': 17,
            'owner': 'smli',
            'cwd': '/srv/private/smli/other',
            'cmdline': 'python train.py',
            'process_name': 'python',
            'used_memory_bytes': GIB,
        }
    ]
    assert attempt04.select_attempt04_gpu(samples)['status'] == 'FAIL'


@pytest.mark.parametrize(
    ('phase', 'allowed'),
    [
        ('resource', set()),
        ('preflight', {'resource'}),
        ('authorization', {'resource', 'snapshot'}),
    ],
)
def test_attempt04_guard_accepts_only_exact_phase_predecessors(
    tmp_path: Path, phase: str, allowed: set[str]
) -> None:
    root = tmp_path / 'attempt-04'
    root.mkdir()
    paths = attempt04._artifact_paths(root)
    for label in allowed:
        paths[label].write_text('{}', encoding='utf-8')
    attempt04._guard_attempt_phase(root, phase)
    unexpected = next(label for label in paths if label not in allowed)
    paths[unexpected].write_text('{}', encoding='utf-8')
    with pytest.raises(V4ExecutionError, match='ARTIFACT_COLLISION'):
        attempt04._guard_attempt_phase(root, phase)


@pytest.mark.parametrize('label', list(attempt04._ATTEMPT_ARTIFACT_NAMES))
def test_attempt04_guard_rejects_any_partial(
    tmp_path: Path, label: str
) -> None:
    root = tmp_path / 'attempt-04'
    root.mkdir()
    path = attempt04._artifact_paths(root)[label]
    path.with_name(path.name + '.partial').write_text('partial')
    with pytest.raises(V4ExecutionError, match='ARTIFACT_COLLISION'):
        attempt04._guard_attempt_phase(root, 'resource')


def test_attempt04_rejects_prior_attempt_namespace(tmp_path: Path) -> None:
    for prior in ('attempt-01', 'attempt-02', 'attempt-03'):
        path = tmp_path / prior / 'v4-resource-revalidation.json'
        with pytest.raises(V4ExecutionError, match='HISTORY_REUSE'):
            attempt04._attempt_path(path)


def test_authorized_scope_is_frozen() -> None:
    scope = attempt04._authorized_scope()
    assert scope['models'] == ['VGGT', 'MASt3R']
    assert scope['rectified_member_count'] == 960
    assert scope['scientific_unit_count'] == 400
    assert scope['synthetic_severity_axis'] == 'beta-only'
    assert scope['primary_endpoint'] == 'Pose'
    assert scope['supporting_evidence'] == ['Fusion', 'F-score']
    assert scope['fallback_allowed'] is False
    assert scope['device_switch_allowed'] is False