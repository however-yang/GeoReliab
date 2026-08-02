from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import georeliab_mve.v4_attempt04_authorization as attempt04
from georeliab_mve.v4_execution import V4ExecutionError


GIB = 1024 * 1024 * 1024


def _sample(number: int, *, eligible: bool = True) -> dict[str, object]:
    device = {
        'index': 0,
        'uuid': 'GPU-flow',
        'pci_bus_id': '00000000:17:00.0',
        'model': 'NVIDIA A100 80GB PCIe',
        'total_memory_bytes': 80 * GIB,
        'free_memory_bytes': 79 * GIB,
        'used_memory_bytes': GIB,
        'utilization_gpu_percent': 0 if eligible else 4,
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
    return {
        'schema_version': attempt04.INVENTORY_SCHEMA,
        'attempt_id': attempt04.ATTEMPT_ID,
        'base_commit': attempt04.BASE_COMMIT,
        'base_tree': attempt04.BASE_TREE,
        'hostname': 'a100-smli',
        'timestamp_utc': f'2026-08-02T00:00:{number * 5:02d}Z',
        'driver_version': '580.95.05',
        'cuda_runtime': '13.0',
        'devices': [deepcopy(device)],
    }


def _resource_payload(tmp_path: Path, root: Path) -> dict[str, object]:
    return {
        'tooling_commit': 'a' * 40,
        'tooling_tree': 'b' * 40,
        'runtime_root': str(tmp_path.resolve()),
        'attempt_root': str(root.resolve()),
        'protocol_id': attempt04.V4_PROTOCOL_ID,
        'protocol_sha256': attempt04.V4_PROTOCOL_SHA256,
    }


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    eligible: bool,
) -> tuple[dict[str, object], Path, Path, dict[str, object]]:
    root = tmp_path / 'authorization-attempts' / 'attempt-04'
    root.mkdir(parents=True)
    resource = root / 'v4-resource-revalidation.json'
    resource.write_text('{}', encoding='utf-8')
    snapshot = root / 'v4-hardware-snapshot.json'
    resources = _resource_payload(tmp_path, root)
    monkeypatch.setattr(
        attempt04,
        'validate_attempt04_resources',
        lambda _path: resources,
    )
    queued = iter(_sample(index, eligible=eligible) for index in range(3))
    payload = attempt04.create_attempt04_gpu_preflight(
        worktree=tmp_path,
        resource_receipt_path=resource,
        output_path=snapshot,
        inventory_sampler=lambda: next(queued),
        sleeper=lambda _seconds: None,
        tooling_commit=str(resources['tooling_commit']),
        tooling_tree=str(resources['tooling_tree']),
    )
    return payload, snapshot, resource, resources


def test_attempt04_pass_writes_snapshot_without_pending_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, snapshot, _, _ = _prepare(
        tmp_path, monkeypatch, eligible=True
    )
    assert payload['status'] == 'PASS'
    assert payload['selected_gpu']['uuid'] == 'GPU-flow'
    assert snapshot.is_file()
    assert not (snapshot.parent / 'v4-gpu-selection-decision.json').exists()
    checked = attempt04.validate_attempt04_gpu_preflight(snapshot)
    assert checked['nvidia_smi_invocations'] == 9
    assert checked['torch_probe_invocations'] == 0
    assert checked['model_forwards'] == 0


def test_attempt04_no_eligible_gpu_writes_terminal_blocked_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, snapshot, _, _ = _prepare(
        tmp_path, monkeypatch, eligible=False
    )
    assert payload['status'] == 'FAIL'
    decision_path = snapshot.parent / 'v4-gpu-selection-decision.json'
    decision = attempt04._load_json(decision_path)
    attempt04._validate_decision_payload(decision)
    assert decision['status'] == 'BLOCKED'
    assert decision['reason_code'] == 'V4_NO_ELIGIBLE_IDLE_GPU'
    assert decision['pass_receipt_generated'] is False
    assert decision['execution_authorization_generated'] is False
    assert not (snapshot.parent / 'v4-gpu-selection-receipt.json').exists()
    assert not (snapshot.parent / 'v4-execution-authorization.json').exists()


def test_attempt04_resource_gate_precedes_sampler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / 'authorization-attempts' / 'attempt-04'
    root.mkdir(parents=True)
    resource = root / 'v4-resource-revalidation.json'
    resource.write_text('{}', encoding='utf-8')
    calls = 0

    def sampler() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _sample(calls)

    monkeypatch.setattr(
        attempt04,
        'validate_attempt04_resources',
        lambda _path: (_ for _ in ()).throw(
            V4ExecutionError('V4_RESOURCE_CLOSURE_REVALIDATION_FAILED')
        ),
    )
    with pytest.raises(V4ExecutionError):
        attempt04.create_attempt04_gpu_preflight(
            worktree=tmp_path,
            resource_receipt_path=resource,
            output_path=root / 'v4-hardware-snapshot.json',
            inventory_sampler=sampler,
            sleeper=lambda _: None,
            tooling_commit='a' * 40,
            tooling_tree='b' * 40,
        )
    assert calls == 0


def test_attempt04_collision_is_checked_before_sampler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / 'authorization-attempts' / 'attempt-04'
    root.mkdir(parents=True)
    resource = root / 'v4-resource-revalidation.json'
    resource.write_text('{}', encoding='utf-8')
    (root / 'v4-gpu-selection-receipt.json').write_text('{}')
    monkeypatch.setattr(
        attempt04,
        'validate_attempt04_resources',
        lambda _path: pytest.fail('guard must run first'),
    )
    with pytest.raises(V4ExecutionError, match='ARTIFACT_COLLISION'):
        attempt04.create_attempt04_gpu_preflight(
            worktree=tmp_path,
            resource_receipt_path=resource,
            output_path=root / 'v4-hardware-snapshot.json',
            inventory_sampler=lambda: pytest.fail('must not sample'),
            sleeper=lambda _: None,
            tooling_commit='a' * 40,
            tooling_tree='b' * 40,
        )


def test_attempt04_authorization_writes_truthful_terminal_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, snapshot, resource, resources = _prepare(
        tmp_path, monkeypatch, eligible=True
    )
    root = snapshot.parent
    monkeypatch.setattr(
        attempt04,
        'validate_attempt04_gpu_preflight',
        lambda _path: payload,
    )
    monkeypatch.setattr(
        attempt04,
        'validate_attempt04_execution_authorization',
        lambda path: attempt04._load_json(path),
    )
    receipt = root / 'v4-gpu-selection-receipt.json'
    authorization = root / 'v4-execution-authorization.json'
    result = attempt04.create_attempt04_execution_authorization(
        worktree=tmp_path,
        runtime_root=tmp_path,
        resource_receipt_path=resource,
        hardware_snapshot_path=snapshot,
        receipt_path=receipt,
        authorization_path=authorization,
        run_root=tmp_path / 'runs' / 'attempt-04',
        artifact_root=tmp_path / 'artifacts' / 'attempt-04',
        gpu_ledger_path=tmp_path / 'ledgers' / 'attempt-04.jsonl',
        final_evidence_path=tmp_path / 'evidence' / 'attempt-04.json',
        tooling_commit=str(resources['tooling_commit']),
        tooling_tree=str(resources['tooling_tree']),
    )
    assert result['status'] == 'V4_MVE_EXECUTION_AUTHORIZED'
    decision = attempt04._load_json(
        root / 'v4-gpu-selection-decision.json'
    )
    attempt04._validate_decision_payload(decision)
    assert decision['status'] == 'PASS'
    assert decision['reason_code'] == 'V4_MVE_EXECUTION_AUTHORIZED'
    assert decision['pass_receipt_generated'] is True
    assert decision['execution_authorization_generated'] is True
    assert decision['gpu_receipt_sha256'] == attempt04._sha256_file(receipt)
    assert decision['execution_authorization_sha256'] == (
        attempt04._sha256_file(authorization)
    )


def test_attempt04_cross_validation_failure_removes_pass_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, snapshot, resource, resources = _prepare(
        tmp_path, monkeypatch, eligible=True
    )
    root = snapshot.parent
    monkeypatch.setattr(
        attempt04,
        'validate_attempt04_gpu_preflight',
        lambda _path: payload,
    )
    monkeypatch.setattr(
        attempt04,
        'validate_attempt04_execution_authorization',
        lambda _path: (_ for _ in ()).throw(
            V4ExecutionError('V4_ATTEMPT04_CROSS_VALIDATION_FAILED')
        ),
    )
    receipt = root / 'v4-gpu-selection-receipt.json'
    authorization = root / 'v4-execution-authorization.json'
    decision = root / 'v4-gpu-selection-decision.json'
    with pytest.raises(V4ExecutionError, match='CROSS_VALIDATION_FAILED'):
        attempt04.create_attempt04_execution_authorization(
            worktree=tmp_path,
            runtime_root=tmp_path,
            resource_receipt_path=resource,
            hardware_snapshot_path=snapshot,
            receipt_path=receipt,
            authorization_path=authorization,
            run_root=tmp_path / 'runs' / 'attempt-04',
            artifact_root=tmp_path / 'artifacts' / 'attempt-04',
            gpu_ledger_path=tmp_path / 'ledgers' / 'attempt-04.jsonl',
            final_evidence_path=tmp_path / 'evidence' / 'attempt-04.json',
            tooling_commit=str(resources['tooling_commit']),
            tooling_tree=str(resources['tooling_tree']),
        )
    assert not receipt.exists()
    assert not authorization.exists()
    assert not decision.exists()

def test_attempt04_decision_rejects_inconsistent_generation_flags(
    tmp_path: Path
) -> None:
    resource = tmp_path / 'resource.json'
    resource.write_text('{}')
    payload = attempt04._decision_payload(
        status='BLOCKED',
        reason_code='V4_NO_ELIGIBLE_IDLE_GPU',
        tooling_commit='a' * 40,
        tooling_tree='b' * 40,
        resource_receipt_path=resource,
        snapshot_path=None,
    )
    payload['pass_receipt_generated'] = True
    payload = attempt04._signed(
        {
            key: value
            for key, value in payload.items()
            if key != 'decision_payload_sha256'
        },
        'decision_payload_sha256',
    )
    with pytest.raises(V4ExecutionError, match='DECISION_TAMPER'):
        attempt04._validate_decision_payload(payload)