from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from georeliab_mve.v4_authorization import (
    AUTHORIZED_GPU_MODEL,
    AUTHORIZED_RESOURCE_KEYS,
    IMPLEMENTATION_ANCHOR_COMMIT,
    IMPLEMENTATION_ANCHOR_TREE,
    V4_EXECUTION_AUTHORIZATION_SCHEMA_VERSION,
    create_execution_authorization,
    create_hardware_preflight,
    load_json,
    sha256_file,
    validate_execution_authorization,
)
from georeliab_mve.v4_counterfactuals import SCIENTIFIC_MODELS
from georeliab_mve.v4_execution import SCIENTIFIC_MVE, V4ExecutionError, V4ExecutionReceipt
from georeliab_mve.v4_science_lock import V4_PROTOCOL_ID, V4_PROTOCOL_SHA256


DEVICE_UUID = "GPU-" + "a" * 32
SCHEDULE_SHA = "b" * 64


def _sample(**updates: object) -> dict[str, object]:
    payload = {
        "host": "pytest-host",
        "timestamp_utc": "2026-08-01T00:00:00Z",
        "requested_physical_index": 1,
        "resolved_physical_index": 1,
        "device_uuid": DEVICE_UUID,
        "device_model": AUTHORIZED_GPU_MODEL,
        "total_memory_bytes": 80 * 1024 * 1024 * 1024,
        "free_memory_bytes": 79 * 1024 * 1024 * 1024,
        "used_memory_bytes": 1 * 1024 * 1024 * 1024,
        "utilization_gpu_percent": 0,
        "temperature_c": 31,
        "driver_version": "555.55",
        "cuda_runtime": "CUDA Version 12.4",
        "mig_mode": "Disabled",
        "ecc_health": "OK",
        "compute_processes": [],
    }
    payload.update(updates)
    return payload


def _pass_probe(model_id: str, index: int, sample: dict[str, object]) -> dict[str, object]:
    return {
        "model_id": model_id,
        "torch_device_count": 1,
        "torch_cuda_available": True,
        "torch_current_device": 0,
        "mapped_device_uuid": sample["device_uuid"],
        "mapped_device_model": sample["device_model"],
        "mapped_total_memory_bytes": sample["total_memory_bytes"],
        "compute_process_count": 0,
    }


@pytest.mark.parametrize(
    ("samples", "reason"),
    [
        ((_sample(), _sample(resolved_physical_index=2)), "V4_GPU_INDEX_RESOLUTION_MISMATCH"),
        ((_sample(), _sample(device_uuid="GPU-drift")), "V4_GPU_UUID_DRIFT"),
        ((_sample(utilization_gpu_percent=99, compute_processes=[{"pid": 123, "owner": "other", "cwd": "/tmp", "cmdline": "python", "used_memory_bytes": 1}]), _sample()), "V4_GPU_RECEIPT_NO_ACTIVE_COMPUTE_PROCESS"),
        ((_sample(mig_mode="Enabled"), _sample()), "V4_GPU_MIG_ENABLED"),
        ((_sample(free_memory_bytes=15 * 1024 * 1024 * 1024), _sample()), "V4_GPU_FREE_MEMORY_INSUFFICIENT"),
        ((_sample(device_model="NVIDIA A100-SXM4-80GB"), _sample()), "V4_GPU_MODEL_NOT_AUTHORIZED"),
        ((_sample(utilization_gpu_percent=1), _sample()), "V4_GPU_UTILIZATION_NONZERO"),
        ((_sample(ecc_health="ERROR"), _sample()), "V4_GPU_HEALTH_ERROR"),
    ],
)
def test_gpu_preflight_basic_failures_publish_snapshot_only(
    tmp_path: Path,
    samples: tuple[dict[str, object], dict[str, object]],
    reason: str,
) -> None:
    calls = {"sample": 0, "probe": 0}

    def sampler(_index: int) -> dict[str, object]:
        value = samples[calls["sample"]]
        calls["sample"] += 1
        return value

    def probe(model_id: str, index: int, sample: dict[str, object]) -> dict[str, object]:
        calls["probe"] += 1
        return _pass_probe(model_id, index, sample)

    output = tmp_path / "hardware.json"
    result = create_hardware_preflight(
        output_path=output,
        requested_physical_index=1,
        schedule_sha256=SCHEDULE_SHA,
        sample_interval_seconds=0,
        sampler=sampler,
        sleeper=lambda _seconds: None,
        probe_runner=probe,
    )

    assert result["status"] == "FAIL"
    assert result["reason_code"] == reason
    assert output.is_file()
    assert not (tmp_path / "v4-execution-receipt.json").exists()
    assert calls["probe"] == 0


def test_gpu_preflight_pass_writes_rich_snapshot_receipt_and_cleans_partials(tmp_path: Path) -> None:
    result = create_hardware_preflight(
        output_path=tmp_path / "hardware.json",
        requested_physical_index=1,
        schedule_sha256=SCHEDULE_SHA,
        sample_interval_seconds=0,
        sampler=lambda _index: _sample(),
        sleeper=lambda _seconds: None,
        probe_runner=_pass_probe,
    )

    assert result["status"] == "PASS"
    snapshot = load_json(tmp_path / "hardware.json")
    assert snapshot["devices"][0]["model"] == AUTHORIZED_GPU_MODEL
    assert len(snapshot["samples"]) == 2
    assert {row["model_id"] for row in snapshot["model_environment_probes"]} == set(SCIENTIFIC_MODELS)
    receipt = load_json(tmp_path / "v4-execution-receipt.json")
    assert receipt["schedule_sha256"] == SCHEDULE_SHA
    assert list(tmp_path.glob("*.partial")) == []


def test_failed_preflight_rerun_removes_old_pass_receipt(tmp_path: Path) -> None:
    output = tmp_path / "hardware.json"
    first = create_hardware_preflight(
        output_path=output,
        requested_physical_index=1,
        schedule_sha256=SCHEDULE_SHA,
        sample_interval_seconds=0,
        sampler=lambda _index: _sample(),
        sleeper=lambda _seconds: None,
        probe_runner=_pass_probe,
    )
    assert first["status"] == "PASS"
    receipt = tmp_path / "v4-execution-receipt.json"
    assert receipt.is_file()

    second = create_hardware_preflight(
        output_path=output,
        requested_physical_index=1,
        schedule_sha256=SCHEDULE_SHA,
        sample_interval_seconds=0,
        sampler=lambda _index: _sample(device_model="NVIDIA A100-SXM4-80GB"),
        sleeper=lambda _seconds: None,
        probe_runner=_pass_probe,
    )

    assert second["status"] == "FAIL"
    assert not receipt.exists()


def test_gpu_preflight_probe_mismatch_fails_without_receipt(tmp_path: Path) -> None:
    def bad_probe(model_id: str, index: int, sample: dict[str, object]) -> dict[str, object]:
        payload = _pass_probe(model_id, index, sample)
        payload["mapped_device_uuid"] = "GPU-wrong"
        return payload

    result = create_hardware_preflight(
        output_path=tmp_path / "hardware.json",
        requested_physical_index=1,
        schedule_sha256=SCHEDULE_SHA,
        sample_interval_seconds=0,
        sampler=lambda _index: _sample(),
        sleeper=lambda _seconds: None,
        probe_runner=bad_probe,
    )

    assert result["status"] == "FAIL"
    assert result["reason_code"] == "V4_GPU_TORCH_PROBE_PHYSICAL_DEVICE_MISMATCH"
    assert not (tmp_path / "v4-execution-receipt.json").exists()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


class _FakeSchedule:
    models = SCIENTIFIC_MODELS
    units = tuple(range(400))


def _patch_schedule_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("georeliab_mve.v4_authorization.parse_scientific_schedule", lambda _text: object())
    monkeypatch.setattr("georeliab_mve.v4_authorization.validate_scientific_schedule", lambda _value: _FakeSchedule())


def _prepare_authorization_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    _patch_schedule_validation(monkeypatch)
    resource_paths = {}
    for key in AUTHORIZED_RESOURCE_KEYS:
        path = tmp_path / "prepared" / f"{key}.json"
        _write_json(path, {"key": key})
        resource_paths[key] = {"path": str(path), "sha256": sha256_file(path)}
    schedule_sha = str(resource_paths["scientific_schedule_400"]["sha256"])
    preflight_result = create_hardware_preflight(
        output_path=tmp_path / "artifacts" / "hardware.json",
        requested_physical_index=1,
        schedule_sha256=schedule_sha,
        sample_interval_seconds=0,
        sampler=lambda _index: _sample(),
        sleeper=lambda _seconds: None,
        probe_runner=_pass_probe,
    )
    assert preflight_result["status"] == "PASS"
    _write_json(tmp_path / "resources.json", resource_paths)
    (tmp_path / "run").mkdir()
    (tmp_path / "artifacts" / "final").mkdir(parents=True, exist_ok=True)
    return {
        "receipt": tmp_path / "artifacts" / "v4-execution-receipt.json",
        "resources": tmp_path / "resources.json",
        "run_root": tmp_path / "run",
        "artifact_root": tmp_path / "artifacts",
        "final_evidence": tmp_path / "artifacts" / "final" / "v4-scientific-bundle.json",
        "authorization": tmp_path / "artifacts" / "authorization.json",
    }


def test_create_and_validate_execution_authorization_binds_receipt_resources_and_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prepare_authorization_inputs(tmp_path, monkeypatch)

    result = create_execution_authorization(
        root=tmp_path,
        receipt_path=paths["receipt"],
        resource_inventory_path=paths["resources"],
        run_root=paths["run_root"],
        artifact_root=paths["artifact_root"],
        final_evidence_path=paths["final_evidence"],
        output_path=paths["authorization"],
    )

    assert result["status"] == "PASS"
    payload = load_json(paths["authorization"])
    assert payload["schema_version"] == V4_EXECUTION_AUTHORIZATION_SCHEMA_VERSION
    assert payload["authorized_scope"]["model_set"] == list(SCIENTIFIC_MODELS)
    assert payload["authorized_scope"]["authorized_stop_gpu_seconds"] == 35 * 3600
    assert payload["authorized_scope"]["hard_ceiling_allocated_bytes"] == 1_000_000_000_000
    assert payload["finalizer"] == "georeliab_mve.v4_execution:finalize_v4_scientific_bundle"
    assert validate_execution_authorization(paths["authorization"])["status"] == "PASS"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["authorized_scope"]["model_set"].append("COLMAP"),
        lambda payload: payload["authorized_scope"].__setitem__("dataset", "UAVLight"),
        lambda payload: payload["authorized_scope"]["fog_states"].append("rain-s1"),
        lambda payload: payload["authorized_scope"].__setitem__("authorized_stop_gpu_seconds", 36 * 3600),
        lambda payload: payload["authorized_scope"].__setitem__("max_concurrency", 2),
        lambda payload: payload["authorized_scope"].__setitem__("fallback_allowed", True),
        lambda payload: payload["authorized_scope"].__setitem__("retry_allowed", True),
    ],
)
def test_validate_authorization_rejects_scope_budget_fallback_parallel_and_retry_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
) -> None:
    paths = _prepare_authorization_inputs(tmp_path, monkeypatch)
    create_execution_authorization(
        root=tmp_path,
        receipt_path=paths["receipt"],
        resource_inventory_path=paths["resources"],
        run_root=paths["run_root"],
        artifact_root=paths["artifact_root"],
        final_evidence_path=paths["final_evidence"],
        output_path=paths["authorization"],
    )
    payload = load_json(paths["authorization"])
    mutate(payload)
    _write_json(paths["authorization"], payload)

    with pytest.raises(V4ExecutionError, match="V4_AUTHORIZATION_TAMPER|V4_AUTHORIZATION_SCOPE_EXPANDED"):
        validate_execution_authorization(paths["authorization"])


def test_create_authorization_rejects_stale_anchor_path_escape_and_schedule_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prepare_authorization_inputs(tmp_path, monkeypatch)
    receipt_payload = load_json(paths["receipt"])
    receipt_payload["project_commit"] = "1" * 40
    _write_json(paths["receipt"], receipt_payload)
    with pytest.raises(V4ExecutionError, match="V4_AUTHORIZATION_STALE_ANCHOR"):
        create_execution_authorization(
            root=tmp_path,
            receipt_path=paths["receipt"],
            resource_inventory_path=paths["resources"],
            run_root=paths["run_root"],
            artifact_root=paths["artifact_root"],
            final_evidence_path=paths["final_evidence"],
            output_path=paths["authorization"],
        )

    paths = _prepare_authorization_inputs(tmp_path / "fresh", monkeypatch)
    with pytest.raises(V4ExecutionError, match="V4_AUTHORIZATION_PATH_ESCAPE"):
        create_execution_authorization(
            root=tmp_path / "fresh",
            receipt_path=paths["receipt"],
            resource_inventory_path=paths["resources"],
            run_root=paths["run_root"],
            artifact_root=paths["artifact_root"],
            final_evidence_path=tmp_path / "escaped.json",
            output_path=paths["authorization"],
        )

    resources = load_json(paths["resources"])
    _write_json(Path(resources["scientific_schedule_400"]["path"]), {"tampered": True})
    with pytest.raises(V4ExecutionError, match="V4_SCHEDULE_TAMPER"):
        create_execution_authorization(
            root=tmp_path / "fresh",
            receipt_path=paths["receipt"],
            resource_inventory_path=paths["resources"],
            run_root=paths["run_root"],
            artifact_root=paths["artifact_root"],
            final_evidence_path=paths["final_evidence"],
            output_path=paths["authorization"],
        )


def test_failed_authorization_rerun_removes_old_pass_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prepare_authorization_inputs(tmp_path, monkeypatch)
    create_execution_authorization(
        root=tmp_path,
        receipt_path=paths["receipt"],
        resource_inventory_path=paths["resources"],
        run_root=paths["run_root"],
        artifact_root=paths["artifact_root"],
        final_evidence_path=paths["final_evidence"],
        output_path=paths["authorization"],
    )
    assert paths["authorization"].is_file()
    receipt = load_json(paths["receipt"])
    receipt["project_commit"] = "1" * 40
    _write_json(paths["receipt"], receipt)

    with pytest.raises(V4ExecutionError, match="V4_AUTHORIZATION_STALE_ANCHOR"):
        create_execution_authorization(
            root=tmp_path,
            receipt_path=paths["receipt"],
            resource_inventory_path=paths["resources"],
            run_root=paths["run_root"],
            artifact_root=paths["artifact_root"],
            final_evidence_path=paths["final_evidence"],
            output_path=paths["authorization"],
        )
    assert not paths["authorization"].exists()


def test_create_authorization_rejects_malicious_receipt_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prepare_authorization_inputs(tmp_path, monkeypatch)
    receipt = load_json(paths["receipt"])
    receipt["fallback_allowed"] = True
    _write_json(paths["receipt"], receipt)

    with pytest.raises(V4ExecutionError, match="V4_GPU_RECEIPT_NO_FALLBACK_REQUIRED"):
        create_execution_authorization(
            root=tmp_path,
            receipt_path=paths["receipt"],
            resource_inventory_path=paths["resources"],
            run_root=paths["run_root"],
            artifact_root=paths["artifact_root"],
            final_evidence_path=paths["final_evidence"],
            output_path=paths["authorization"],
        )


def test_validate_authorization_rejects_receipt_and_resource_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prepare_authorization_inputs(tmp_path, monkeypatch)
    create_execution_authorization(
        root=tmp_path,
        receipt_path=paths["receipt"],
        resource_inventory_path=paths["resources"],
        run_root=paths["run_root"],
        artifact_root=paths["artifact_root"],
        final_evidence_path=paths["final_evidence"],
        output_path=paths["authorization"],
    )
    receipt = load_json(paths["receipt"])
    receipt["retry_allowed"] = True
    _write_json(paths["receipt"], receipt)
    with pytest.raises(V4ExecutionError, match="V4_RECEIPT_TAMPER"):
        validate_execution_authorization(paths["authorization"])

    paths = _prepare_authorization_inputs(tmp_path / "resource", monkeypatch)
    create_execution_authorization(
        root=tmp_path / "resource",
        receipt_path=paths["receipt"],
        resource_inventory_path=paths["resources"],
        run_root=paths["run_root"],
        artifact_root=paths["artifact_root"],
        final_evidence_path=paths["final_evidence"],
        output_path=paths["authorization"],
    )
    payload = load_json(paths["authorization"])
    row = payload["resource_inventory"][0]
    _write_json(Path(row["path"]), {"tampered": True})
    with pytest.raises(V4ExecutionError, match="V4_RESOURCE_TAMPER"):
        validate_execution_authorization(paths["authorization"])


def test_v4_preflight_cli_fails_closed_without_gpu(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "georeliab_mve",
            "v4-gpu-preflight",
            "--output",
            str(tmp_path / "hardware.json"),
            "--sample-interval-seconds",
            "0",
        ],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode == 2
    assert (tmp_path / "hardware.json").is_file()
    payload = json.loads(completed.stdout)
    assert payload["status"] == "FAIL"
    assert "receipt_path" not in payload