from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import pytest

import georeliab_mve.v4_authorization as v4_authorization

from georeliab_mve.v4_authorization import (
    ATTEMPT_ID,
    ATTEMPT_INVENTORY_SCHEMA_VERSION,
    AUTHORIZED_GPU_MODEL,
    EXCLUDED_GPU_UUID,
    create_attempt_execution_authorization,
    create_attempt_hardware_preflight,
    materialize_attempt_resources,
    nvidia_smi_all_gpu_inventory_sample,
    select_attempt_gpu,
    validate_attempt_execution_authorization,
    validate_attempt_receipt,
    validate_attempt_resources,
)
from georeliab_mve.v4_counterfactuals import (
    AssetEvidence,
    DTU_LIGHTING_SEMANTIC_TO_PHYSICAL_TOKEN,
    LIGHTING_STATES,
    SCIENTIFIC_MODELS,
    SCIENTIFIC_STATES,
    TEST_SCENE_IDS,
    materialize_dtu_state_identity,
)
from georeliab_mve.v4_execution import V4ExecutionError


ORDERED_VIEWS = (1, 7, 13, 19, 25, 31, 37, 43)
SOURCE_ROOT = Path(__file__).resolve().parents[1]

def _sha(label: str) -> str:
    return hashlib.sha256(label.encode('utf-8')).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n', encoding='utf-8')


def _file(path: Path, text: str) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return {'path': str(path.resolve()), 'sha256': _sha_file(path)}


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _device(
    index: int,
    uuid: str,
    *,
    model: str = AUTHORIZED_GPU_MODEL,
    total_gib: int = 80,
    free_gib: int = 70,
    utilization: int = 0,
    mig: str = 'Disabled',
    ecc: str = 'OK',
    processes: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        'index': index,
        'uuid': uuid,
        'model': model,
        'total_memory_bytes': total_gib * 1024**3,
        'free_memory_bytes': free_gib * 1024**3,
        'used_memory_bytes': (total_gib - free_gib) * 1024**3,
        'utilization_gpu_percent': utilization,
        'temperature_c': 31,
        'driver_version': '570.1',
        'cuda_runtime': 'CUDA Version 12.8',
        'mig_mode': mig,
        'ecc_health': ecc,
        'ecc_uncorrected_volatile_total': 0 if ecc == 'OK' else 1,
        'compute_processes': [] if processes is None else processes,
    }


def _inventory(
    timestamp: str, devices: list[dict[str, object]]
) -> dict[str, object]:
    return {
        'schema_version': ATTEMPT_INVENTORY_SCHEMA_VERSION,
        'attempt_id': ATTEMPT_ID,
        'hostname': 'a100-host',
        'timestamp_utc': timestamp,
        'driver_version': '570.1',
        'cuda_runtime': 'CUDA Version 12.8',
        'devices': devices,
    }


def _samples(*devices: dict[str, object]) -> list[dict[str, object]]:
    return [
        _inventory('2026-08-01T00:00:00Z', deepcopy(list(devices))),
        _inventory('2026-08-01T00:00:05Z', deepcopy(list(devices))),
    ]


def _probe(
    model: str,
    uuid: str,
    index: int,
    selected: dict[str, object],
) -> dict[str, object]:
    del uuid, index
    return {
        'model_id': model,
        'model_instantiated': False,
        'checkpoint_loaded': False,
        'forward_executed': False,
        'torch_cuda_available': True,
        'torch_device_count': 1,
        'torch_current_device': 0,
        'mapped_device_uuid': selected['uuid'],
        'mapped_device_model': selected['model'],
        'mapped_total_memory_bytes': selected['total_memory_bytes'],
        'post_probe_mig_mode': 'Disabled',
        'post_probe_ecc_health': 'OK',
        'residual_compute_process_count': 0,
    }


def test_selection_excludes_uuid_and_uses_deterministic_total_free_uuid_order() -> None:
    excluded = _device(0, EXCLUDED_GPU_UUID, total_gib=96, free_gib=95)
    lexical_b = _device(1, 'GPU-b', total_gib=80, free_gib=70)
    lexical_a = _device(2, 'GPU-a', total_gib=80, free_gib=70)
    selected = select_attempt_gpu(_samples(excluded, lexical_b, lexical_a))
    assert selected['status'] == 'PASS'
    assert selected['selected_gpu']['uuid'] == 'GPU-a'
    assert selected['selected_gpu']['index'] == 2
    excluded_row = next(
        row
        for row in selected['candidate_evaluations']
        if row['uuid'] == EXCLUDED_GPU_UUID
    )
    assert excluded_row['reason_code'] == 'V4_GPU_UUID_EXCLUDED'


def test_all_gpu_inventory_records_full_identity_and_process_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        'georeliab_mve.v4_authorization._process_owner', lambda pid: 'owner'
    )
    monkeypatch.setattr(
        'georeliab_mve.v4_authorization._process_cwd', lambda pid: '/cwd'
    )
    monkeypatch.setattr(
        'georeliab_mve.v4_authorization._process_cmdline',
        lambda pid: 'python job.py',
    )

    def runner(command):
        joined = ' '.join(command)
        if '--query-gpu=' in joined:
            return (
                '0, GPU-a, NVIDIA A100 80GB PCIe, 81920, 70000, 11920, '
                '0, 31, Disabled, 0, 570.1\n'
                '1, GPU-b, NVIDIA A100 80GB PCIe, 81920, 60000, 21920, '
                '0, 33, Disabled, 0, 570.1\n'
            )
        if '--query-compute-apps=' in joined:
            return 'GPU-b, 42, python, 1024\n'
        return 'NVIDIA-SMI 570.1 CUDA Version: 12.8'

    sample = nvidia_smi_all_gpu_inventory_sample(command_runner=runner)
    assert sample['attempt_id'] == ATTEMPT_ID
    assert len(sample['devices']) == 2
    assert sample['devices'][0]['compute_processes'] == []
    process = sample['devices'][1]['compute_processes'][0]
    assert process == {
        'pid': 42,
        'owner': 'owner',
        'cwd': '/cwd',
        'cmdline': 'python job.py',
        'process_name': 'python',
        'used_memory_bytes': 1024 * 1024**2,
    }


@pytest.mark.parametrize(
    ('mutate', 'reason'),
    [
        (
            lambda rows: rows[1]['devices'][0].update(index=3),
            'V4_GPU_MAPPING_DRIFT',
        ),
        (
            lambda rows: rows[0]['devices'][0].update(mig_mode='Enabled'),
            'V4_GPU_MIG_ENABLED',
        ),
        (
            lambda rows: rows[0]['devices'][0].update(
                free_memory_bytes=15 * 1024**3
            ),
            'V4_GPU_FREE_MEMORY_INSUFFICIENT',
        ),
        (
            lambda rows: rows[0]['devices'][0].update(model='NVIDIA H100'),
            'V4_GPU_MODEL_NOT_AUTHORIZED',
        ),
    ],
)
def test_selection_fails_closed_for_mapping_mig_memory_and_model(
    mutate, reason: str
) -> None:
    rows = _samples(_device(0, 'GPU-good'))
    mutate(rows)
    result = select_attempt_gpu(rows)
    assert result['status'] == 'FAIL'
    assert result['reason_code'] == 'V4_NO_ELIGIBLE_IDLE_GPU'
    assert result['candidate_evaluations'][0]['reason_code'] == reason


def test_selection_rejects_unknown_process_owner_and_process_instability() -> None:
    process = {
        'pid': 12,
        'owner': None,
        'cwd': '/work',
        'cmdline': 'python worker.py',
        'used_memory_bytes': 1,
    }
    unknown = select_attempt_gpu(
        _samples(_device(0, 'GPU-good', processes=[process]))
    )
    assert unknown['candidate_evaluations'][0]['reason_code'] == (
        'V4_GPU_PROCESS_IDENTITY_UNPROVEN'
    )

    stable_process = {**process, 'owner': 'runner'}
    rows = _samples(_device(0, 'GPU-good', processes=[stable_process]))
    rows[1]['devices'][0]['compute_processes'][0]['pid'] = 13
    unstable = select_attempt_gpu(rows)
    assert unstable['candidate_evaluations'][0]['reason_code'] == (
        'V4_GPU_PROCESS_STATE_UNSTABLE'
    )


def test_no_candidates_is_terminal_exact_reason() -> None:
    result = select_attempt_gpu(
        _samples(_device(0, EXCLUDED_GPU_UUID), _device(1, 'GPU-busy', utilization=1))
    )
    assert result['status'] == 'FAIL'
    assert result['reason_code'] == 'V4_NO_ELIGIBLE_IDLE_GPU'
    assert result['selected_gpu'] is None


def test_probe_failure_never_falls_back_and_creates_no_receipt(tmp_path: Path) -> None:
    attempt = tmp_path / ATTEMPT_ID
    samples = iter(
        _samples(_device(0, 'GPU-a-selected'), _device(1, 'GPU-z-fallback'))
    )

    def fail_probe(model, uuid, index, selected):
        del model, uuid, index, selected
        raise V4ExecutionError('V4_GPU_TORCH_PROBE_FAILED')

    result = create_attempt_hardware_preflight(
        output_path=attempt / 'v4-hardware-preflight.json',
        schedule_sha256=_sha('schedule'),
        inventory_sampler=lambda: next(samples),
        sleeper=lambda seconds: None,
        probe_runner=fail_probe,
    )
    snapshot = json.loads(
        (attempt / 'v4-hardware-preflight.json').read_text(encoding='utf-8')
    )
    assert result['status'] == 'FAIL'
    assert snapshot['selected_gpu']['uuid'] == 'GPU-a-selected'
    assert snapshot['no_fallback_or_switch'] is True
    assert not (attempt / 'v4-execution-receipt.json').exists()
    assert not (attempt / 'v4-execution-authorization.json').exists()


@pytest.mark.parametrize(
    'logical_uuid', ['GPU-wrong-logical-device', None]
)
def test_default_probe_rejects_wrong_or_unavailable_logical_cuda0_uuid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    logical_uuid: str | None,
) -> None:
    selected = _device(4, 'GPU-selected')
    logical_probe = {
        'model_instantiated': False,
        'checkpoint_loaded': False,
        'forward_executed': False,
        'torch_cuda_available': True,
        'torch_device_count': 1,
        'torch_current_device': 0,
        'mapped_device_uuid': logical_uuid,
        'mapped_device_model': selected['model'],
        'mapped_total_memory_bytes': selected['total_memory_bytes'],
    }
    monkeypatch.setattr(
        v4_authorization, '_frozen_python_for_model', lambda model: model
    )
    monkeypatch.setattr(
        v4_authorization,
        '_run_text_command',
        lambda command, env=None: json.dumps(logical_probe),
    )
    monkeypatch.setattr(
        v4_authorization,
        'nvidia_smi_hardware_sample',
        lambda index: {
            'device_uuid': selected['uuid'],
            'device_model': selected['model'],
            'total_memory_bytes': selected['total_memory_bytes'],
            'mig_mode': 'Disabled',
            'ecc_health': 'OK',
            'compute_processes': [],
        },
    )
    rows = iter(_samples(selected))
    attempt = tmp_path / ATTEMPT_ID
    result = create_attempt_hardware_preflight(
        output_path=attempt / 'v4-hardware-preflight.json',
        schedule_sha256=_sha('schedule'),
        inventory_sampler=lambda: next(rows),
        sleeper=lambda seconds: None,
        probe_runner=v4_authorization._default_attempt_torch_probe,
    )

    assert result['status'] == 'FAIL'
    assert result['reason_code'] == 'V4_GPU_TORCH_PROBE_DEVICE_MISMATCH'
    assert not (attempt / 'v4-execution-receipt.json').exists()


def test_attempt_collision_and_cross_attempt_path_are_rejected(tmp_path: Path) -> None:
    attempt = tmp_path / ATTEMPT_ID
    samples = _samples(_device(0, 'GPU-good'))
    sleeps: list[float] = []
    result = create_attempt_hardware_preflight(
        output_path=attempt / 'v4-hardware-preflight.json',
        schedule_sha256=_sha('schedule'),
        inventory_sampler=iter(samples).__next__,
        sleeper=sleeps.append,
        probe_runner=_probe,
    )
    assert result['status'] == 'PASS'
    assert sleeps == [5.0]
    with pytest.raises(V4ExecutionError, match='V4_ATTEMPT_ARTIFACT_COLLISION'):
        create_attempt_hardware_preflight(
            output_path=attempt / 'v4-hardware-preflight.json',
            schedule_sha256=_sha('schedule'),
            inventory_sampler=iter(samples).__next__,
            sleeper=lambda seconds: None,
            probe_runner=_probe,
        )
    with pytest.raises(V4ExecutionError, match='V4_ATTEMPT_PATH_MISMATCH'):
        create_attempt_hardware_preflight(
            output_path=tmp_path / 'attempt-01' / 'v4-hardware-preflight.json',
            schedule_sha256=_sha('schedule'),
            inventory_sampler=iter(samples).__next__,
            sleeper=lambda seconds: None,
            probe_runner=_probe,
        )


def _asset(member: str, label: str) -> AssetEvidence:
    return AssetEvidence(
        member=member,
        sha256=_sha(label),
        source_uri=f'file:///attempt-02-source/{member}',
    )


def _state(scene_id: int, state_id: str):
    inputs: dict[int, AssetEvidence] = {}
    cameras: dict[int, AssetEvidence] = {}
    for view_id in ORDERED_VIEWS:
        if state_id in LIGHTING_STATES:
            member = (
                f'Rectified/scan{scene_id}/'
                f'rect_{view_id:03d}_{DTU_LIGHTING_SEMANTIC_TO_PHYSICAL_TOKEN[state_id]}_r5000.png'
            )
        else:
            member = (
                f'SyntheticFog/scan{scene_id}/{state_id}/view_{view_id:03d}.png'
            )
        inputs[view_id] = _asset(
            member, f'rgb:{scene_id}:{state_id}:{view_id}'
        )
        cameras[view_id] = _asset(
            f'MVS Data/Calibration/cal18/pos_{view_id:03d}.txt',
            f'camera:{scene_id}:{view_id}',
        )
    clean = None
    if state_id.startswith('fog-'):
        clean = {
            view_id: _asset(
                f'Rectified/scan{scene_id}/rect_{view_id:03d}_3_r5000.png',
                f'rgb:{scene_id}:L3:{view_id}',
            )
            for view_id in ORDERED_VIEWS
        }
    return materialize_dtu_state_identity(
        source_root=SOURCE_ROOT,
        scene_id=scene_id,
        state_id=state_id,
        ordered_view_ids=ORDERED_VIEWS,
        rgb_inputs=inputs,
        cameras=cameras,
        gt_point_cloud=_asset(
            f'Points/stl/stl{scene_id:03d}_total.ply', f'gt:{scene_id}'
        ),
        observability_mask=_asset(
            f'MVS Data/ObsMask/ObsMask{scene_id}_10.mat', f'mask:{scene_id}'
        ),
        clean_source_inputs=clean,
    )


@lru_cache(maxsize=1)
def _states():
    return tuple(
        _state(scene_id, state_id)
        for scene_id in TEST_SCENE_IDS
        for state_id in SCIENTIFIC_STATES
    )


def _resource_sources(root: Path) -> Path:
    sources = root / 'production-sources'
    models: dict[str, object] = {}
    environments: dict[str, object] = {}
    for model in SCIENTIFIC_MODELS:
        models[model] = {
            'weights': _file(sources / f'{model}.weights', f'{model}-weights'),
            'config': _file(sources / f'{model}.json', f'{model}-config'),
            'upstream_commit': _sha(f'{model}-upstream')[:40],
        }
        environments[model] = _file(
            sources / f'{model}.lock', f'{model}-environment'
        )
    archives: dict[str, object] = {}
    for name in ('SampleSet', 'Points', 'Rectified'):
        binding = _file(sources / f'{name}.zip', f'{name}-archive')
        archives[name] = {
            **binding,
            'etag': f'etag-{name}',
            'central_directory_sha256': _sha(f'{name}-central-directory'),
            'referenced_members': [
                {'member': f'{name}/member', 'sha256': _sha(f'{name}-member')}
            ],
        }
    split_path = sources / 'v4-split.json'
    _write_json(split_path, {'test_scene_ids': list(TEST_SCENE_IDS)})
    fog_path = sources / 'fog-manifest.json'
    _write_json(
        fog_path,
        {
            'model': 'Koschmieder',
            'severity_family': 'beta-only',
            'state_ids': ['fog-s1', 'fog-s2', 'fog-s3'],
        },
    )
    states_path = sources / 'state-source.json'
    _write_json(states_path, {'states': [state.to_dict() for state in _states()]})
    manifest = {
        'attempt_id': ATTEMPT_ID,
        'model_bindings': models,
        'dtu_archives': archives,
        'v4_split': {'path': str(split_path), 'sha256': _sha_file(split_path)},
        'state_inventory_200': {
            'path': str(states_path),
            'sha256': _sha_file(states_path),
        },
        'fog_manifest': {'path': str(fog_path), 'sha256': _sha_file(fog_path)},
        'environment_locks': environments,
        'science_lock': _file(sources / 'science.lock', 'science-lock'),
    }
    manifest_path = sources / 'resource-source-manifest.json'
    _write_json(manifest_path, manifest)
    return manifest_path


def _materialized(root: Path) -> dict[str, object]:
    manifest = _resource_sources(root)
    output = root / 'artifacts' / 'v4-resource-snapshot.json'
    return materialize_attempt_resources(
        root=root,
        source_manifest_path=manifest,
        output_path=output,
    )


def _passing_preflight(root: Path, schedule_sha: str) -> dict[str, object]:
    rows = iter(_samples(_device(4, 'GPU-attempt-02-selected')))
    return create_attempt_hardware_preflight(
        output_path=root / 'evidence' / 'v4-hardware-preflight.json',
        schedule_sha256=schedule_sha,
        inventory_sampler=lambda: next(rows),
        sleeper=lambda seconds: None,
        probe_runner=_probe,
    )


def _rehash(payload: dict[str, object], field: str) -> None:
    unsigned = {key: value for key, value in payload.items() if key != field}
    payload[field] = hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(',', ':'), default=str
        ).encode('utf-8')
    ).hexdigest()


def test_materialization_produces_exact_200_400_and_deduplicates_l3(
    tmp_path: Path,
) -> None:
    root = tmp_path / ATTEMPT_ID
    result = _materialized(root)
    assert result['state_count'] == 200
    assert result['unit_count'] == 400
    resources = validate_attempt_resources(
        Path(result['resource_snapshot_path'])
    )
    schedule_binding = resources['resources']['scientific_schedule_400']
    schedule_wrapper = json.loads(
        Path(schedule_binding['path']).read_text(encoding='utf-8')
    )
    units = schedule_wrapper['schedule']['units']
    l3 = [unit for unit in units if unit['state_id'] == 'L3']
    assert len(units) == 400
    assert len(l3) == 40
    assert len({(unit['model_id'], unit['scene_id']) for unit in l3}) == 40
    with pytest.raises(V4ExecutionError, match='V4_ATTEMPT_ARTIFACT_COLLISION'):
        materialize_attempt_resources(
            root=root,
            source_manifest_path=_resource_sources(root),
            output_path=Path(result['resource_snapshot_path']),
        )


def test_resource_and_schedule_tamper_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / ATTEMPT_ID
    result = _materialized(root)
    resource_path = Path(result['resource_snapshot_path'])
    resources = validate_attempt_resources(resource_path)
    weights = Path(
        resources['resources']['model_bindings']['VGGT']['weights']['path']
    )
    weights.write_text('tampered', encoding='utf-8')
    with pytest.raises(V4ExecutionError, match='V4_RESOURCE_TAMPER'):
        validate_attempt_resources(resource_path)


@pytest.mark.parametrize(
    'binding_class',
    [
        'model',
        'dtu_archive',
        'environment',
        'science_lock',
        'split',
        'fog',
        'state_source',
        'source_manifest',
    ],
)
@pytest.mark.parametrize(
    'forbidden_source',
    ['pytest_cache', 'attempt-01', 'authorization_artifact'],
)
def test_resource_validation_rejects_rehashed_forbidden_production_paths(
    tmp_path: Path, binding_class: str, forbidden_source: str
) -> None:
    root = tmp_path / ATTEMPT_ID
    result = _materialized(root)
    resource_path = Path(result['resource_snapshot_path'])
    payload = json.loads(resource_path.read_text(encoding='utf-8'))
    if forbidden_source == 'pytest_cache':
        forbidden = root / '.pytest_cache' / 'canonical-source.bin'
    elif forbidden_source == 'attempt-01':
        forbidden = root / 'attempt-01' / 'canonical-source.bin'
    else:
        forbidden = root / 'production-sources' / 'v4-execution-authorization.json'
    binding = _file(forbidden, f'{binding_class}:{forbidden_source}')

    resources = payload['resources']
    if binding_class == 'model':
        resources['model_bindings']['VGGT']['weights'] = binding
    elif binding_class == 'dtu_archive':
        resources['dtu_archives']['SampleSet'].update(binding)
    elif binding_class == 'environment':
        resources['environment_locks']['MASt3R'] = binding
    elif binding_class == 'science_lock':
        resources['science_lock'] = binding
    elif binding_class == 'split':
        resources['v4_split'] = binding
    elif binding_class == 'fog':
        resources['fog_manifest'] = binding
    elif binding_class == 'state_source':
        state_binding = resources['state_inventory_200']
        state_path = Path(state_binding['path'])
        state_payload = json.loads(state_path.read_text(encoding='utf-8'))
        state_payload['source_path'] = binding['path']
        state_payload['source_sha256'] = binding['sha256']
        _write_json(state_path, state_payload)
        state_binding['sha256'] = _sha_file(state_path)
    else:
        payload['source_manifest_path'] = binding['path']
        payload['source_manifest_sha256'] = binding['sha256']
    _rehash(payload, 'resource_snapshot_sha256')
    _write_json(resource_path, payload)

    with pytest.raises(
        V4ExecutionError, match='V4_PRODUCTION_INPUT_SOURCE_FORBIDDEN'
    ):
        validate_attempt_resources(resource_path)


def test_receipt_rejects_stale_attempt_uuid_index_and_payload_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / ATTEMPT_ID
    resources = _materialized(root)
    preflight = _passing_preflight(root, str(resources['schedule_sha256']))
    receipt_path = Path(preflight['receipt_path'])
    original = json.loads(receipt_path.read_text(encoding='utf-8'))
    assert validate_attempt_receipt(receipt_path)['attempt_id'] == ATTEMPT_ID

    for key, value, reason in (
        ('attempt_id', 'attempt-01', 'V4_ATTEMPT_ID_MISMATCH'),
        ('selected_gpu_uuid', 'GPU-drift', 'V4_GPU_RECEIPT_DEVICE_MISMATCH'),
        ('selected_physical_index', 5, 'V4_GPU_RECEIPT_DEVICE_MISMATCH'),
    ):
        payload = deepcopy(original)
        payload[key] = value
        _rehash(payload, 'receipt_payload_sha256')
        _write_json(receipt_path, payload)
        with pytest.raises(V4ExecutionError, match=reason):
            validate_attempt_receipt(receipt_path)
    payload = deepcopy(original)
    payload['selected_gpu_uuid'] = 'GPU-unhashed-tamper'
    _write_json(receipt_path, payload)
    with pytest.raises(V4ExecutionError, match='V4_ATTEMPT_RECEIPT_TAMPER'):
        validate_attempt_receipt(receipt_path)


def test_authorization_rejects_scope_tamper_resource_drift_and_unsafe_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / ATTEMPT_ID
    resources = _materialized(root)
    preflight = _passing_preflight(root, str(resources['schedule_sha256']))
    run_root = root / 'runs'
    artifact_root = root / 'artifacts-live'
    run_root.mkdir(parents=True)
    artifact_root.mkdir(parents=True)
    authorization_path = root / 'evidence' / 'v4-execution-authorization.json'
    result = create_attempt_execution_authorization(
        root=root,
        receipt_path=Path(preflight['receipt_path']),
        resource_snapshot_path=Path(resources['resource_snapshot_path']),
        run_root=run_root,
        artifact_root=artifact_root,
        final_evidence_path=root / 'evidence' / 'final.json',
        output_path=authorization_path,
    )
    assert result['status'] == 'PASS'
    payload = validate_attempt_execution_authorization(authorization_path)
    assert payload['selected_gpu_uuid'] == 'GPU-attempt-02-selected'
    assert payload['selected_physical_index'] == 4

    expanded = deepcopy(payload)
    expanded['authorized_scope']['forbidden'].remove('UAVLight')
    _rehash(expanded, 'authorization_sha256')
    _write_json(authorization_path, expanded)
    with pytest.raises(V4ExecutionError, match='V4_AUTHORIZATION_SCOPE_EXPANDED'):
        validate_attempt_execution_authorization(authorization_path)

    second_root = tmp_path / 'unsafe-root' / ATTEMPT_ID
    second_root.mkdir(parents=True)
    with pytest.raises(V4ExecutionError, match='V4_AUTHORIZATION_PATH_ESCAPE'):
        create_attempt_execution_authorization(
            root=root,
            receipt_path=Path(preflight['receipt_path']),
            resource_snapshot_path=Path(resources['resource_snapshot_path']),
            run_root=second_root,
            artifact_root=artifact_root,
            final_evidence_path=root / 'evidence' / 'final.json',
            output_path=root / 'evidence' / 'second-authorization.json',
        )


def test_preflight_rejects_attempt01_timestamp_hash_reuse(tmp_path: Path) -> None:
    root = tmp_path / ATTEMPT_ID
    history = tmp_path / 'history' / 'attempt-01.json'
    _write_json(
        history,
        {
            'attempt_id': 'attempt-01',
            'timestamp_utc': '2026-08-01T00:00:00Z',
            'hardware_preflight_sha256': _sha('old-snapshot'),
        },
    )
    rows = iter(_samples(_device(0, 'GPU-good')))
    with pytest.raises(
        V4ExecutionError, match='V4_ATTEMPT_HISTORY_REUSE_FORBIDDEN'
    ):
        create_attempt_hardware_preflight(
            output_path=root / 'evidence' / 'v4-hardware-preflight.json',
            schedule_sha256=_sha('schedule'),
            inventory_sampler=lambda: next(rows),
            sleeper=lambda seconds: None,
            probe_runner=_probe,
            historical_evidence_paths=(history,),
        )
    assert not (root / 'evidence' / 'v4-execution-receipt.json').exists()


def test_preflight_rejects_nested_attempt01_timestamp_reuse(
    tmp_path: Path,
) -> None:
    root = tmp_path / ATTEMPT_ID
    history = tmp_path / 'history' / 'attempt-01.json'
    reused_timestamp = '2026-08-01T00:00:00Z'
    _write_json(
        history,
        {
            'attempt_id': 'attempt-01',
            'inventory_samples': [{'timestamp_utc': reused_timestamp}],
        },
    )
    rows = iter(_samples(_device(0, 'GPU-good')))
    with pytest.raises(
        V4ExecutionError, match='V4_ATTEMPT_HISTORY_REUSE_FORBIDDEN'
    ):
        create_attempt_hardware_preflight(
            output_path=root / 'evidence' / 'v4-hardware-preflight.json',
            schedule_sha256=_sha('schedule'),
            inventory_sampler=lambda: next(rows),
            sleeper=lambda seconds: None,
            probe_runner=_probe,
            historical_evidence_paths=(history,),
        )
    assert not (root / 'evidence' / 'v4-execution-receipt.json').exists()


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('max_concurrent_gpus', 2),
        ('sequential_execution', False),
        ('fallback_allowed', True),
        ('device_switch_allowed', True),
        ('retry_allowed', True),
    ],
)
def test_authorization_rejects_parallel_fallback_retry_and_switch(
    tmp_path: Path, field: str, value: object
) -> None:
    root = tmp_path / ATTEMPT_ID
    resources = _materialized(root)
    preflight = _passing_preflight(root, str(resources['schedule_sha256']))
    run_root = root / 'runs'
    artifact_root = root / 'artifacts-live'
    run_root.mkdir(parents=True)
    artifact_root.mkdir(parents=True)
    path = root / 'evidence' / 'v4-execution-authorization.json'
    create_attempt_execution_authorization(
        root=root,
        receipt_path=Path(preflight['receipt_path']),
        resource_snapshot_path=Path(resources['resource_snapshot_path']),
        run_root=run_root,
        artifact_root=artifact_root,
        final_evidence_path=root / 'evidence' / 'final.json',
        output_path=path,
    )
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload[field] = value
    _rehash(payload, 'authorization_sha256')
    _write_json(path, payload)
    with pytest.raises(V4ExecutionError, match='V4_AUTHORIZATION_SCOPE_EXPANDED'):
        validate_attempt_execution_authorization(path)
