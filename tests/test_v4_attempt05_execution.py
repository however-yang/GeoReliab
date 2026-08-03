from __future__ import annotations

import json
from pathlib import Path

import pytest

from georeliab_mve.v4_counterfactuals import (
    SCIENTIFIC_MODELS,
    SCIENTIFIC_STATES,
    TEST_SCENE_IDS,
    ScientificExecutionUnit,
    ScientificSchedule,
    canonical_json_sha256,
)
from georeliab_mve.v4_execution import (
    BYTE_CATASTROPHE,
    BYTE_TARGET,
    GPU_CATASTROPHE_SECONDS,
    GPU_TARGET_SECONDS,
    V4ExecutionError,
    canonical_record_path,
)
from georeliab_mve.v4_science_lock import (
    V4_PROTOCOL_ID,
    V4_PROTOCOL_SHA256,
    V4_PROTOCOL_VERSION,
)

import georeliab_mve.v4_attempt05_execution as attempt05


COMMIT = "1" * 40
TREE = "2" * 40


def _sha(label: str) -> str:
    return canonical_json_sha256({"label": label})


def _input_storage() -> dict[str, object]:
    return {
        "scope": "NEW_CALIBRATION_L3_FOG_PNG_AND_INPUT_CLOSURE_"
        "PAYLOADS_EXCLUDING_MANIFEST",
        "logical_bytes": 0,
        "allocated_bytes": 0,
        "input_closure_sha256": "f" * 64,
    }


def _calibration_schedule() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "model_id": model_id,
            "scene_id": 1000 + scene_index,
            "state_id": "L3",
            "calibration_unit_sha256": _sha(f"cal:{model_id}:{scene_index}"),
        }
        for model_id in SCIENTIFIC_MODELS
        for scene_index in range(20)
    )

def _unit(model_id: str, scene_id: int, state_id: str) -> ScientificExecutionUnit:
    provenance = {
        "schema_version": "georeliab-v4-protocol-provenance-1.0",
        "protocol_id": V4_PROTOCOL_ID,
        "protocol_version": V4_PROTOCOL_VERSION,
        "protocol_sha256": V4_PROTOCOL_SHA256,
    }
    payload = {
        "schema_version": "georeliab-v4-scientific-execution-unit-1.0",
        "protocol_provenance": provenance,
        "dataset": "DTU",
        "model_id": model_id,
        "scene_id": scene_id,
        "state_id": state_id,
        "state_identity_sha256": _sha(f"state:{scene_id}:{state_id}"),
        "pair_identity_sha256": None
        if state_id == "L3"
        else _sha(f"pair:{scene_id}:{state_id}"),
    }
    return ScientificExecutionUnit.from_dict(
        {**payload, "execution_unit_sha256": canonical_json_sha256(payload)}
    )


@pytest.fixture()
def schedule() -> ScientificSchedule:
    units = tuple(
        _unit(model_id, scene_id, state_id)
        for model_id in SCIENTIFIC_MODELS
        for scene_id in TEST_SCENE_IDS
        for state_id in SCIENTIFIC_STATES
    )
    provenance = {
        "schema_version": "georeliab-v4-protocol-provenance-1.0",
        "protocol_id": V4_PROTOCOL_ID,
        "protocol_version": V4_PROTOCOL_VERSION,
        "protocol_sha256": V4_PROTOCOL_SHA256,
    }
    payload = {
        "schema_version": "georeliab-v4-scientific-schedule-1.0",
        "protocol_provenance": provenance,
        "models": list(SCIENTIFIC_MODELS),
        "scene_ids": list(TEST_SCENE_IDS),
        "state_ids": list(SCIENTIFIC_STATES),
        "units": [unit.to_dict() for unit in units],
    }
    return ScientificSchedule(
        protocol_provenance_items=tuple(sorted(provenance.items())),
        models=SCIENTIFIC_MODELS,
        scene_ids=TEST_SCENE_IDS,
        state_ids=SCIENTIFIC_STATES,
        units=units,
        schedule_sha256=canonical_json_sha256(payload),
    )


def _authorization(tmp_path: Path, **updates: object) -> dict[str, object]:
    runtime_root = tmp_path.resolve()
    payload: dict[str, object] = {
        "schema_version": "georeliab-v4-attempt-04-execution-authorization-1.0",
        "attempt_id": "attempt-04",
        "run_id": "a" * 32,
        "status": "V4_MVE_EXECUTION_AUTHORIZED",
        "tooling_commit": COMMIT,
        "tooling_tree": TREE,
        "scientific_anchor_commit": attempt05.SCIENTIFIC_ANCHOR_COMMIT,
        "scientific_anchor_tree": attempt05.SCIENTIFIC_ANCHOR_TREE,
        "protocol_id": V4_PROTOCOL_ID,
        "protocol_sha256": V4_PROTOCOL_SHA256,
        "schedule_file_sha256": attempt05.SCHEDULE_FILE_SHA256,
        "rectified_manifest_sha256": attempt05.MANIFEST_FILE_SHA256,
        "closure_digests": dict(attempt05.ATTEMPT04_CLOSURE_DIGESTS),
        "selected_gpu": {
            "uuid": "GPU-attempt05",
            "pci_bus_id": "00000000:17:00.0",
            "index": 0,
            "model": "NVIDIA A100 80GB PCIe",
        },
        "authorized_scope": {
            "models": list(SCIENTIFIC_MODELS),
            "model_count": 2,
            "dataset": "DTU",
            "paired_lighting_states": ["L1", "L2", "L3", "L4", "L5", "L6", "L7"],
            "rectified_member_count": 960,
            "synthetic_corruption": "Koschmieder fog",
            "synthetic_severity_axis": "beta-only",
            "scientific_unit_count": 400,
            "primary_endpoint": "Pose",
            "supporting_evidence": ["Fusion", "F-score"],
            "single_gpu": True,
            "models_sequential": True,
            "units_sequential": True,
            "fallback_allowed": False,
            "device_switch_allowed": False,
            "forbidden": [
                "UAVLight",
                "v4.1",
                "third model",
                "second corruption family",
                "seed expansion",
                "severity expansion",
                "scene expansion",
                "view expansion",
                "model matrix expansion",
            ],
        },
        "budget": {
            "authorization_gpu_seconds": GPU_TARGET_SECONDS,
            "authorization_storage_bytes": BYTE_TARGET,
            "catastrophe_gpu_seconds": GPU_CATASTROPHE_SECONDS,
            "catastrophe_storage_bytes": BYTE_CATASTROPHE,
        },
        "runtime_root": str(runtime_root),
        "runtime_paths": {
            "run_root": str(runtime_root / "runs" / "attempt-04"),
            "artifact_root": str(runtime_root / "artifacts" / "attempt-04"),
            "gpu_ledger_path": str(runtime_root / "ledgers" / "attempt-04.jsonl"),
            "final_evidence_path": str(runtime_root / "evidence" / "attempt-04.json"),
        },
        "finalizer": "georeliab_mve.v4_execution:finalize_v4_scientific_bundle",
        "execution_lock_created": False,
        "gpu_ledger_created": False,
        "dispatcher_called": False,
        "torch_probe_invocations": 0,
        "model_loads": 0,
        "model_forwards": 0,
        "gpu_inference_seconds": 0,
        "scientific_result": "NO_SCIENTIFIC_RESULT",
    }
    payload.update(updates)
    return payload


def _patch_auth(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> Path:
    path = Path(str(payload["runtime_root"])) / "authorization.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "georeliab_mve.v4_attempt05_execution.validate_attempt04_execution_authorization",
        lambda auth_path: payload,
    )
    return path




def _revision(*, commit: str = COMMIT, tree: str = TREE, clean: bool = True) -> dict[str, object]:
    return {"commit": commit, "tree": tree, "clean": clean}


def _sample(*, uuid: str = "GPU-attempt05", pci: str = "00000000:17:00.0", index: int = 0, free_gib: int = 80, util: int = 0, processes: int = 0) -> dict[str, object]:
    return {
        "devices": [
            {
                "uuid": uuid,
                "pci_bus_id": pci,
                "index": index,
                "total_memory_bytes": 80 * 1024 * 1024 * 1024,
                "free_memory_bytes": free_gib * 1024 * 1024 * 1024,
                "utilization_gpu_percent": util,
                "compute_process_count": processes,
            }
        ]
    }


def _create_preflight(authorization_path: Path) -> dict[str, object]:
    queued = iter(_sample() for _ in range(3))
    return attempt05.create_attempt05_execution_preflight(
        authorization_path=authorization_path,
            attempt05_tooling_commit=COMMIT,
            attempt05_tooling_tree=TREE,
        nvidia_smi_sampler=lambda: next(queued),
        sleeper=lambda _seconds: None,
        revision_checker=_revision,
        cuda_mapping_probe=lambda: {
            "visible_device_count": 1,
            "mapped_uuid": "GPU-attempt05",
            "mapped_pci_bus_id": "00000000:17:00.0",
            "checkpoint_loads": 0,
            "model_loads": 0,
            "model_forwards": 0,
        },
    )



def test_execution_preflight_samples_exact_authorized_gpu_and_cuda_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization_path = _patch_auth(monkeypatch, _authorization(tmp_path))
    sleeps: list[float] = []
    queued = iter(_sample() for _ in range(3))
    preflight = attempt05.create_attempt05_execution_preflight(
        authorization_path=authorization_path,
        attempt05_tooling_commit=COMMIT,
        attempt05_tooling_tree=TREE,
        nvidia_smi_sampler=lambda: next(queued),
        sleeper=lambda seconds: sleeps.append(seconds),
        revision_checker=_revision,
        cuda_mapping_probe=lambda: {
            "visible_device_count": 1,
            "mapped_uuid": "GPU-attempt05",
            "mapped_pci_bus_id": "00000000:17:00.0",
            "checkpoint_loads": 0,
            "model_loads": 0,
            "model_forwards": 0,
        },
    )
    assert sleeps == [5.0, 5.0]
    assert preflight["sample_count"] == 3
    assert preflight["dispatcher_called"] == 0
    assert preflight["model_loads"] == 0
    assert preflight["model_forwards"] == 0
    assert Path(preflight["preflight_path"]).is_file()


def test_execution_preflight_checks_revision_before_gpu_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization_path = _patch_auth(monkeypatch, _authorization(tmp_path))
    samples = {"count": 0}

    def sampler() -> dict[str, object]:
        samples["count"] += 1
        return _sample()

    with pytest.raises(V4ExecutionError, match="V4_ATTEMPT05_TOOLING_REVISION_MISMATCH"):
        attempt05.create_attempt05_execution_preflight(
            authorization_path=authorization_path,
            attempt05_tooling_commit=COMMIT,
            attempt05_tooling_tree=TREE,
            nvidia_smi_sampler=sampler,
            sleeper=lambda _seconds: None,
            revision_checker=lambda: _revision(commit="3" * 40),
        )
    assert samples["count"] == 0


def test_execution_preflight_checks_clean_tree_before_gpu_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization_path = _patch_auth(monkeypatch, _authorization(tmp_path))
    samples = {"count": 0}

    def sampler() -> dict[str, object]:
        samples["count"] += 1
        return _sample()

    with pytest.raises(V4ExecutionError, match="V4_ATTEMPT05_TOOLING_TREE_DIRTY"):
        attempt05.create_attempt05_execution_preflight(
            authorization_path=authorization_path,
            attempt05_tooling_commit=COMMIT,
            attempt05_tooling_tree=TREE,
            nvidia_smi_sampler=sampler,
            sleeper=lambda _seconds: None,
            revision_checker=lambda: _revision(clean=False),
        )
    assert samples["count"] == 0

def test_execution_preflight_rejects_gpu_mismatch_and_busy_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization_path = _patch_auth(monkeypatch, _authorization(tmp_path))
    queued = iter([_sample(uuid="GPU-other"), _sample(), _sample()])
    with pytest.raises(V4ExecutionError, match="V4_ATTEMPT05_PREFLIGHT_GPU_IDENTITY_MISMATCH"):
        attempt05.create_attempt05_execution_preflight(
            authorization_path=authorization_path,
            attempt05_tooling_commit=COMMIT,
            attempt05_tooling_tree=TREE,
            nvidia_smi_sampler=lambda: next(queued),
            sleeper=lambda _seconds: None,
            revision_checker=_revision,
        )
    authorization_path = _patch_auth(monkeypatch, _authorization(tmp_path / "busy"))
    queued = iter([_sample(processes=1), _sample(), _sample()])
    with pytest.raises(V4ExecutionError, match="V4_ATTEMPT05_PREFLIGHT_COMPUTE_PROCESS_PRESENT"):
        attempt05.create_attempt05_execution_preflight(
            authorization_path=authorization_path,
            attempt05_tooling_commit=COMMIT,
            attempt05_tooling_tree=TREE,
            nvidia_smi_sampler=lambda: next(queued),
            sleeper=lambda _seconds: None,
            revision_checker=_revision,
        )
def test_start_receipt_consumes_attempt04_rebinds_paths_and_closes_cpu_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schedule: ScientificSchedule,
) -> None:
    authorization_path = _patch_auth(monkeypatch, _authorization(tmp_path))
    _create_preflight(authorization_path)
    receipt = attempt05.create_attempt05_start_receipt(
        authorization_path=authorization_path,
        schedule=schedule,
        model_independent_states=tuple(object() for _ in range(200)),
        split_assignment=object(),
        calibration_schedule=_calibration_schedule(),
            attempt05_tooling_commit=COMMIT,
            attempt05_tooling_tree=TREE,
        input_storage=_input_storage(),
    )
    assert receipt["schema_version"] == attempt05.START_RECEIPT_SCHEMA
    assert receipt["attempt_id"] == "attempt-05"
    assert receipt["attempt04_consumed"] is True
    assert Path(receipt["runtime_paths"]["run_root"]).parts[-2:] == ("runs", "v4-mve-attempt-05")
    assert receipt["cpu_input_closure"] == {
        "calibration_l3_units": 40,
        "scientific_units": 400,
        "rectified_non_l3_members": 960,
        "test_l3_units": 40,
        "calibration_l3_units_against_native_warning": 40,
        "fog_units_bound_to_l3": 120,
        "model_independent_states": 200,
        "calibration_schedule_units": 40,
    }
    assert receipt["single_gpu_serial"] == {
        "max_concurrent_gpus": 1,
        "models_sequential": True,
        "units_sequential": True,
        "fallback_allowed": False,
        "device_switch_allowed": False,
        "retry_allowed": False,
    }
    assert (tmp_path / "runs" / "v4-mve-attempt-05" / "v4-attempt05-start-receipt.json").is_file()
    with pytest.raises(V4ExecutionError, match="V4_ATTEMPT05_START_RECEIPT_IMMUTABLE"):
        attempt05.create_attempt05_start_receipt(
            authorization_path=authorization_path,
            schedule=schedule,
            model_independent_states=tuple(object() for _ in range(200)),
            split_assignment=object(),
            calibration_schedule=_calibration_schedule(),
            attempt05_tooling_commit=COMMIT,
            attempt05_tooling_tree=TREE,
            input_storage=_input_storage(),
        )


def test_authorization_rejects_path_escape_and_scope_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_paths = _authorization(tmp_path)
    bad_paths["runtime_paths"] = dict(bad_paths["runtime_paths"])
    bad_paths["runtime_paths"]["gpu_ledger_path"] = str(tmp_path.parent / "escape.jsonl")
    with pytest.raises(V4ExecutionError, match="V4_ATTEMPT05_RUNTIME_PATH_ESCAPE"):
        attempt05._authorized_context_from_payload(bad_paths, Path("auth.json"))

    bad_scope = _authorization(tmp_path)
    bad_scope["authorized_scope"] = dict(bad_scope["authorized_scope"])
    bad_scope["authorized_scope"]["scientific_unit_count"] = 401
    _patch_auth(monkeypatch, bad_scope)
    with pytest.raises(V4ExecutionError, match="V4_ATTEMPT05_AUTHORIZATION_SCOPE_MISMATCH"):
        attempt05.load_attempt05_authorized_context(
            authorization_path=tmp_path / "authorization.json"
        )


def test_resume_requires_clean_boundary_and_blocks_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schedule: ScientificSchedule,
) -> None:
    authorization_path = _patch_auth(monkeypatch, _authorization(tmp_path))
    _create_preflight(authorization_path)
    attempt05.create_attempt05_start_receipt(
        authorization_path=authorization_path,
        schedule=schedule,
        model_independent_states=tuple(object() for _ in range(200)),
        split_assignment=object(),
        calibration_schedule=_calibration_schedule(),
            attempt05_tooling_commit=COMMIT,
            attempt05_tooling_tree=TREE,
        input_storage=_input_storage(),
    )
    context = attempt05.load_attempt05_authorized_context(
        authorization_path=authorization_path
    )
    first_path = canonical_record_path(context.run_root, schedule.units[0])
    first_path.parent.mkdir(parents=True)
    first_path.with_name(first_path.name + ".partial").write_text("{}", encoding="utf-8")
    with pytest.raises(V4ExecutionError, match="V4_ATTEMPT05_RESUME_PARTIAL_BLOCKED"):
        attempt05.authorize_attempt05_next_dispatch(
            authorization_path=authorization_path,
            schedule=schedule,
            resume=True,
        )

    first_path.with_name(first_path.name + ".partial").unlink()
    partial_dir = context.run_root / "calibration" / "VGGT" / "scan1" / "L3.partial"
    partial_dir.mkdir(parents=True)
    with pytest.raises(V4ExecutionError, match="V4_ATTEMPT05_RESUME_PARTIAL_BLOCKED"):
        attempt05.authorize_attempt05_next_dispatch(
            authorization_path=authorization_path,
            schedule=schedule,
            resume=True,
        )


def test_hash_chain_ledger_rejects_tamper_and_appends_next_hash(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    first = attempt05.append_attempt05_ledger_event(
        ledger_path=ledger,
        event_type="START",
        payload={"value": 1},
    )
    second = attempt05.append_attempt05_ledger_event(
        ledger_path=ledger,
        event_type="UNIT_FINALIZED",
        payload={"value": 2},
    )
    assert first["event_sha256"] == second["previous_event_sha256"]
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    rows[0]["payload"]["value"] = 999
    ledger.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(V4ExecutionError, match="V4_ATTEMPT05_LEDGER_CHAIN_TAMPER"):
        attempt05.append_attempt05_ledger_event(
            ledger_path=ledger,
            event_type="UNIT_FINALIZED",
            payload={"value": 3},
        )


def test_start_receipt_rejects_fitted_calibration_outputs_and_requires_tooling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schedule: ScientificSchedule,
) -> None:
    authorization_path = _patch_auth(monkeypatch, _authorization(tmp_path))
    _create_preflight(authorization_path)
    with pytest.raises(V4ExecutionError, match="V4_ATTEMPT05_TOOLING_REVISION_REQUIRED"):
        attempt05.create_attempt05_start_receipt(
            authorization_path=authorization_path,
            schedule=schedule,
            model_independent_states=tuple(object() for _ in range(200)),
            split_assignment=object(),
            calibration_schedule=_calibration_schedule(),
        )
    with pytest.raises(V4ExecutionError, match="V4_ATTEMPT05_CALIBRATION_SCHEDULE_REQUIRED"):
        attempt05.create_attempt05_start_receipt(
            authorization_path=authorization_path,
            schedule=schedule,
            model_independent_states=tuple(object() for _ in range(200)),
            split_assignment=object(),
            calibration_schedule=(object(), object()),
            attempt05_tooling_commit=COMMIT,
            attempt05_tooling_tree=TREE,
        )


def test_q90_freeze_is_separate_immutable_post_calibration_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schedule: ScientificSchedule,
) -> None:
    authorization_path = _patch_auth(monkeypatch, _authorization(tmp_path))
    _create_preflight(authorization_path)
    start = attempt05.create_attempt05_start_receipt(
        authorization_path=authorization_path,
        schedule=schedule,
        model_independent_states=tuple(object() for _ in range(200)),
        split_assignment=object(),
        calibration_schedule=_calibration_schedule(),
            attempt05_tooling_commit=COMMIT,
            attempt05_tooling_tree=TREE,
        input_storage=_input_storage(),
    )
    assert start["q90_freeze_artifact_sha256"] is None
    freeze = attempt05.create_attempt05_q90_freeze_artifact(
        authorization_path=authorization_path,
        calibration_schedule_sha256=start["calibration_schedule_sha256"],
        q90_calibrations=(
            {"model_id": "VGGT", "state_id": "L3", "scene_count": 20, "quantile_probability": 0.90, "quantile_method": "linear", "alarm_threshold": 1.25, "calibration_identifier": _sha("q90:vggt")},
            {"model_id": "MASt3R", "state_id": "L3", "scene_count": 20, "quantile_probability": 0.90, "quantile_method": "linear", "alarm_threshold": 2.5, "calibration_identifier": _sha("q90:mast3r")},
        ),
            attempt05_tooling_commit=COMMIT,
            attempt05_tooling_tree=TREE,
    )
    checked = attempt05.validate_attempt05_q90_freeze_artifact(
        freeze["q90_freeze_artifact_path"],
        authorization_path=authorization_path,
    )
    assert checked["q90_freeze_artifact_sha256"] == freeze["q90_freeze_artifact_sha256"]
    reused = attempt05.create_attempt05_q90_freeze_artifact(
        authorization_path=authorization_path,
        calibration_schedule_sha256=start["calibration_schedule_sha256"],
        q90_calibrations=freeze["q90_calibrations"],
        attempt05_tooling_commit=COMMIT,
        attempt05_tooling_tree=TREE,
    )
    assert reused["q90_freeze_artifact_sha256"] == freeze["q90_freeze_artifact_sha256"]
    drifted = [dict(row) for row in freeze["q90_calibrations"]]
    drifted[0]["alarm_threshold"] = 99.0
    with pytest.raises(V4ExecutionError, match="V4_ATTEMPT05_Q90_FREEZE_MISMATCH"):
        attempt05.create_attempt05_q90_freeze_artifact(
            authorization_path=authorization_path,
            calibration_schedule_sha256=start["calibration_schedule_sha256"],
            q90_calibrations=drifted,
            attempt05_tooling_commit=COMMIT,
            attempt05_tooling_tree=TREE,
        )

def test_recovery_revision_preserves_old_evidence_and_allows_truthful_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schedule: ScientificSchedule,
) -> None:
    authorization_path = _patch_auth(monkeypatch, _authorization(tmp_path))
    _create_preflight(authorization_path)
    start = attempt05.create_attempt05_start_receipt(
        authorization_path=authorization_path,
        schedule=schedule,
        model_independent_states=tuple(object() for _ in range(200)),
        split_assignment=object(),
        calibration_schedule=_calibration_schedule(),
        attempt05_tooling_commit=COMMIT,
        attempt05_tooling_tree=TREE,
        input_storage=_input_storage(),
    )
    q90_rows = (
        {
            "model_id": "VGGT",
            "state_id": "L3",
            "scene_count": 20,
            "quantile_probability": 0.90,
            "quantile_method": "linear",
            "alarm_threshold": 1.25,
            "calibration_identifier": _sha("recovery:q90:vggt"),
        },
        {
            "model_id": "MASt3R",
            "state_id": "L3",
            "scene_count": 20,
            "quantile_probability": 0.90,
            "quantile_method": "linear",
            "alarm_threshold": 2.5,
            "calibration_identifier": _sha("recovery:q90:mast3r"),
        },
    )
    freeze = attempt05.create_attempt05_q90_freeze_artifact(
        authorization_path=authorization_path,
        calibration_schedule_sha256=start["calibration_schedule_sha256"],
        q90_calibrations=q90_rows,
        attempt05_tooling_commit=COMMIT,
        attempt05_tooling_tree=TREE,
    )
    context = attempt05.load_attempt05_authorized_context(
        authorization_path=authorization_path
    )
    attempt05.append_attempt05_ledger_event(
        ledger_path=context.gpu_ledger_path,
        event_type="MVE_RUN_STARTED",
        payload={"gpu_inference_seconds_total": 0.0},
    )
    for index, row in enumerate(_calibration_schedule(), start=1):
        attempt05.append_attempt05_ledger_event(
            ledger_path=context.gpu_ledger_path,
            event_type="CALIBRATION_UNIT_COMPLETE",
            payload={
                "model_id": row["model_id"],
                "scene_id": row["scene_id"],
                "state_id": "L3",
                "gpu_inference_seconds_total": float(index),
                "logical_bytes_total": index,
                "allocated_bytes_total": index,
            },
        )
    first = schedule.units[0]
    attempt05.append_attempt05_ledger_event(
        ledger_path=context.gpu_ledger_path,
        event_type="SCIENTIFIC_UNIT_COMPLETE",
        payload={
            "model_id": first.model_id,
            "scene_id": first.scene_id,
            "state_id": first.state_id,
            "status": "VALID_COMPLETE",
            "gpu_inference_seconds_total": 41.0,
            "logical_bytes_total": 41,
            "allocated_bytes_total": 41,
        },
    )
    legacy = (
        context.run_root
        / "stage"
        / "SCIENTIFIC_MVE"
        / "records"
        / first.model_id
        / f"scan{first.scene_id:03d}"
    )
    legacy.mkdir(parents=True)
    (legacy / "task_audit_record.json").write_text(
        "{\"immutable\":true}\n", encoding="utf-8"
    )
    old_start = Path(context.run_root / "v4-attempt05-start-receipt.json").read_bytes()
    old_q90 = Path(freeze["q90_freeze_artifact_path"]).read_bytes()
    new_commit = "3" * 40
    new_tree = "4" * 40

    recovery = attempt05.create_attempt05_recovery_revision(
        authorization_path=authorization_path,
        from_tooling_commit=COMMIT,
        from_tooling_tree=TREE,
        to_tooling_commit=new_commit,
        to_tooling_tree=new_tree,
        nvidia_smi_sampler=lambda: _sample(),
        sleeper=lambda _seconds: None,
        cuda_mapping_probe=lambda: {
            "visible_device_count": 1,
            "mapped_uuid": "GPU-attempt05",
            "mapped_pci_bus_id": "00000000:17:00.0",
            "checkpoint_loads": 0,
            "model_loads": 0,
            "model_forwards": 0,
        },
        revision_checker=lambda: _revision(
            commit=new_commit,
            tree=new_tree,
        ),
    )
    assert recovery["scientific_units_completed"] == 1
    assert recovery["existing_artifacts_are_read_only"] is True
    assert attempt05.create_attempt05_recovery_revision(
        authorization_path=authorization_path,
        from_tooling_commit=COMMIT,
        from_tooling_tree=TREE,
        to_tooling_commit=new_commit,
        to_tooling_tree=new_tree,
    )["recovery_revision_sha256"] == recovery["recovery_revision_sha256"]
    resumed = attempt05.create_attempt05_start_receipt(
        authorization_path=authorization_path,
        schedule=schedule,
        model_independent_states=tuple(object() for _ in range(200)),
        split_assignment=object(),
        calibration_schedule=_calibration_schedule(),
        resume=True,
        attempt05_tooling_commit=new_commit,
        attempt05_tooling_tree=new_tree,
        input_storage=_input_storage(),
    )
    assert resumed["attempt05_tooling_commit"] == COMMIT
    assert attempt05.create_attempt05_q90_freeze_artifact(
        authorization_path=authorization_path,
        calibration_schedule_sha256=start["calibration_schedule_sha256"],
        q90_calibrations=q90_rows,
        attempt05_tooling_commit=new_commit,
        attempt05_tooling_tree=new_tree,
    )["attempt05_tooling_commit"] == COMMIT
    (legacy / f"{first.state_id}.json").write_text(
        "{\"projection\":true}\n", encoding="utf-8"
    )
    attempt05.validate_attempt05_recovery_revision(
        authorization_path=authorization_path,
        expected_tooling_commit=new_commit,
        expected_tooling_tree=new_tree,
    )
    assert Path(context.run_root / "v4-attempt05-start-receipt.json").read_bytes() == old_start
    assert Path(freeze["q90_freeze_artifact_path"]).read_bytes() == old_q90


def test_recovery_revision_rejects_partial_and_wrong_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization_path = _patch_auth(monkeypatch, _authorization(tmp_path))
    context = attempt05.load_attempt05_authorized_context(
        authorization_path=authorization_path
    )
    partial = context.run_root / "blocked.partial"
    partial.parent.mkdir(parents=True)
    partial.write_text("partial", encoding="utf-8")
    with pytest.raises(
        V4ExecutionError, match="V4_ATTEMPT05_RESUME_PARTIAL_BLOCKED"
    ):
        attempt05.create_attempt05_recovery_revision(
            authorization_path=authorization_path,
            from_tooling_commit=COMMIT,
            from_tooling_tree=TREE,
            to_tooling_commit="3" * 40,
            to_tooling_tree="4" * 40,
        )


def test_ledger_resume_normalizes_legacy_artifact_only_totals(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    attempt05.append_attempt05_ledger_event(
        ledger_path=ledger,
        event_type="MVE_RUN_STARTED",
        payload={
            "gpu_inference_seconds_total": 0.0,
            "logical_bytes_total": 100,
            "allocated_bytes_total": 200,
        },
    )
    attempt05.append_attempt05_ledger_event(
        ledger_path=ledger,
        event_type="CALIBRATION_UNIT_COMPLETE",
        payload={
            "model_id": "VGGT",
            "scene_id": 1,
            "state_id": "L3",
            "gpu_inference_seconds_total": 1.0,
            "logical_bytes_total": 10,
            "allocated_bytes_total": 20,
        },
    )
    attempt05.append_attempt05_ledger_event(
        ledger_path=ledger,
        event_type="TOOLING_RECOVERY_REVISION",
        payload={
            "gpu_inference_seconds_total": 1.0,
            "logical_bytes_total": 130,
            "allocated_bytes_total": 240,
            "resource_accounting_mode": "TOTAL",
        },
    )

    totals = attempt05.rehydrate_attempt05_ledger_totals(ledger)

    assert totals.resource_accounting_mode == "TOTAL"
    assert totals.logical_bytes == 130
    assert totals.allocated_bytes == 240


def test_resource_gate_enforces_targets_and_catastrophe_fuses() -> None:
    assert attempt05.evaluate_attempt05_resource_gate(
        gpu_inference_seconds=GPU_TARGET_SECONDS,
        wall_runtime_seconds=1,
        new_logical_bytes=1,
        new_allocated_bytes=1,
    ).reason_code == "V4_REAUTHORIZE_GPUH_TARGET_REACHED"
    assert attempt05.evaluate_attempt05_resource_gate(
        gpu_inference_seconds=1,
        wall_runtime_seconds=1,
        new_logical_bytes=BYTE_TARGET,
        new_allocated_bytes=1,
    ).reason_code == "V4_REAUTHORIZE_LOGICAL_BYTES_TARGET_REACHED"
    assert attempt05.evaluate_attempt05_resource_gate(
        gpu_inference_seconds=GPU_CATASTROPHE_SECONDS,
        wall_runtime_seconds=1,
        new_logical_bytes=1,
        new_allocated_bytes=1,
    ).reason_code == "V4_CATASTROPHE_GPUH_FUSE"
    assert attempt05.evaluate_attempt05_resource_gate(
        gpu_inference_seconds=1,
        wall_runtime_seconds=1,
        new_logical_bytes=1,
        new_allocated_bytes=BYTE_CATASTROPHE,
    ).reason_code == "V4_CATASTROPHE_STORAGE_FUSE"


def test_dispatch_requires_start_receipt_and_returns_first_unit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schedule: ScientificSchedule,
) -> None:
    authorization_path = _patch_auth(monkeypatch, _authorization(tmp_path))
    _create_preflight(authorization_path)
    with pytest.raises(V4ExecutionError, match="V4_ATTEMPT05_START_RECEIPT_REQUIRED"):
        attempt05.authorize_attempt05_next_dispatch(
            authorization_path=authorization_path,
            schedule=schedule,
        )
    attempt05.create_attempt05_start_receipt(
        authorization_path=authorization_path,
        schedule=schedule,
        model_independent_states=tuple(object() for _ in range(200)),
        split_assignment=object(),
        calibration_schedule=_calibration_schedule(),
            attempt05_tooling_commit=COMMIT,
            attempt05_tooling_tree=TREE,
        input_storage=_input_storage(),
    )
    decision = attempt05.authorize_attempt05_next_dispatch(
        authorization_path=authorization_path,
        schedule=schedule,
    )
    assert decision.status == "PASS"
    assert decision.reason_code == "V4_ATTEMPT05_NEXT_UNIT_AUTHORIZED"
    assert decision.unit == schedule.units[0]


def test_ledger_resume_totals_rehydrates_hash_chain_budget(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    attempt05.append_attempt05_ledger_event(
        ledger_path=ledger,
        event_type="CALIBRATION_UNIT_COMPLETE",
        payload={
            "gpu_inference_seconds_total": 1.5,
            "wall_runtime_seconds_total": 2.5,
            "logical_bytes_total": 3,
            "allocated_bytes_total": 4,
            "peak_memory_mb_total": 5.5,
            "model_id": "vggt",
            "scene_id": 1,
            "state_id": "L3",
        },
    )
    attempt05.append_attempt05_ledger_event(
        ledger_path=ledger,
        event_type="SCIENTIFIC_UNIT_COMPLETE",
        payload={
            "status": "INVALID_FAILURE_RECORDED",
            "gpu_inference_seconds_total": 6.5,
            "wall_runtime_seconds_total": 7.5,
            "logical_bytes_total": 8,
            "allocated_bytes_total": 9,
            "peak_memory_mb_total": 10.5,
            "model_id": "vggt",
            "scene_id": 1,
            "state_id": "L1",
        },
    )

    totals = attempt05.rehydrate_attempt05_ledger_totals(ledger)

    assert totals.gpu_inference_seconds == 6.5
    assert totals.wall_runtime_seconds == 7.5
    assert totals.logical_bytes == 8
    assert totals.allocated_bytes == 9
    assert totals.completed_units == 2
    assert totals.invalid_units == 1
    assert totals.peak_memory_mb == 10.5
    assert totals.projection_unit_keys == frozenset()
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["payload"]["retry_count"] == 0


def test_projection_event_is_accounted_without_incrementing_scientific_units(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "projection-ledger.jsonl"
    attempt05.append_attempt05_ledger_event(
        ledger_path=ledger,
        event_type="CANONICAL_RECORD_PROJECTION",
        payload={
            "model_id": "VGGT",
            "scene_id": 1,
            "state_id": "L1",
            "task_record_sha256": "a" * 64,
            "gpu_inference_seconds_total": 0.0,
            "wall_runtime_seconds_total": 0.0,
            "logical_bytes_total": 5,
            "allocated_bytes_total": 8,
            "peak_memory_mb_total": 0.0,
        },
    )

    totals = attempt05.rehydrate_attempt05_ledger_totals(ledger)

    assert totals.projection_unit_keys == frozenset({("VGGT", 1, "L1")})
    assert totals.scientific_units_completed == 0
    assert totals.logical_bytes == 5
    assert totals.allocated_bytes == 8


def test_finalizer_requires_exact_400_records_and_uses_authorized_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schedule: ScientificSchedule,
) -> None:
    authorization_path = _patch_auth(monkeypatch, _authorization(tmp_path))
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        "georeliab_mve.v4_attempt05_execution.finalize_v4_scientific_bundle",
        lambda **kwargs: {"kwargs": calls.setdefault("kwargs", kwargs)},
    )
    with pytest.raises(V4ExecutionError, match="V4_RECORD_COUNT_NOT_400"):
        attempt05.finalize_attempt05_scientific_bundle(
            authorization_path=authorization_path,
            record_paths=[],
            scientific_schedule=schedule,
            model_independent_states=tuple(object() for _ in range(200)),
            split_assignment=object(),
            native_warning_calibrations=(object(), object()),
        )
    result = attempt05.finalize_attempt05_scientific_bundle(
        authorization_path=authorization_path,
        record_paths=[tmp_path / f"{index}.json" for index in range(400)],
        scientific_schedule=schedule,
        model_independent_states=tuple(object() for _ in range(200)),
        split_assignment=object(),
            native_warning_calibrations=(object(), object()),
    )
    assert result["kwargs"]["output_dir"] == tmp_path / "artifacts" / "v4-mve-attempt-05" / "final-evidence"
    assert result["kwargs"]["record_paths"][0] == tmp_path / "0.json"





def test_dtu_projection_decomposition_recovers_c2w_and_preserves_view_order() -> None:
    np = __import__("numpy")
    k = np.array(
        [
            [1200.0, 0.0, 320.0],
            [0.0, 1180.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    center = np.array([1.0, 2.0, 3.0])
    projection = k @ np.column_stack([rotation, -rotation @ center])

    result = attempt05.decompose_dtu_projection_to_camera_to_world(projection)

    assert result.max_reprojection_abs_error < 1e-7
    assert result.camera_center_world == pytest.approx(center, abs=1e-8)
    assert result.camera_to_world[0][1] == pytest.approx(1.0)
    shifted = projection.copy()
    shifted[0, 3] += 1e-9
    ordered = attempt05.decompose_ordered_dtu_projections(
        {
            8: projection,
            3: shifted,
            5: projection,
            1: projection,
            2: projection,
            4: projection,
            6: projection,
            7: projection,
        },
        ordered_view_ids=(3, 1, 2, 4, 5, 6, 7, 8),
    )
    assert ordered[0].projection != ordered[1].projection
    with pytest.raises(V4ExecutionError, match="V4_ATTEMPT05_DTU_VIEW_ORDER_INVALID"):
        attempt05.decompose_ordered_dtu_projections(
            {1: projection},
            ordered_view_ids=(1, 1, 2, 3, 4, 5, 6, 7),
        )





