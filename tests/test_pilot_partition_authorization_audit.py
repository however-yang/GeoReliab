"""RED contracts for Pilot partition freeze and explicit authorization.

These tests are CPU-only and use synthetic manifests.  They authorize no real
GPU work, create no Pilot input, and must remain outside ``georeliab_mve``.
Exactly twelve contracts define the boundary between a successful formal Gate
2 admission report and the later Pilot input-materialization preflight.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from georeliab_mve.v4_counterfactuals import (
    SCIENTIFIC_MODELS,
    SCIENTIFIC_STATES,
)
from georeliab_mve.v4_scoped import build_pilot_partition
from tests import pilot_partition_authorization_audit as audit


ADMISSION_READY = "V4_PILOT_ADMISSION_READY_FOR_EXPLICIT_AUTHORIZATION"
ADMISSION_SCHEMA = "georeliab-v4-gate2-pilot-readiness-audit-1.0"
SCHEDULE_SHA = "1" * 64
PROTOCOL_BYTES = b"frozen georeliab v4 protocol fixture\n"
PROTOCOL_SHA = hashlib.sha256(PROTOCOL_BYTES).hexdigest()
SOURCE_COMMIT = "3" * 40
SOURCE_TREE = "4" * 40
GPU_UUID = "GPU-" + "5" * 32


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_value(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _admission_payload(formal_closure: Path) -> dict[str, object]:
    return {
        "schema_version": ADMISSION_SCHEMA,
        "status": ADMISSION_READY,
        "validation_class": "GATE2_TO_PILOT_READINESS_AUDIT",
        "scientific_result": audit.NO_SCIENTIFIC_RESULT,
        "gates": {
            "g0": {
                "status": "G0_SOURCE_TOOLCHAIN_PASS",
                "source_commit": SOURCE_COMMIT,
                "source_tree": SOURCE_TREE,
                "worktree_clean": True,
                "scientific_result": audit.NO_SCIENTIFIC_RESULT,
            },
            "g1": {
                "status": "G1_CPU_FAULT_MATRIX_PASS",
                "scientific_result": audit.NO_SCIENTIFIC_RESULT,
            },
            "g2": {
                "status": "G2_FORMAL_GPU_SMOKE_PASS",
                "formal_gate2_equivalent": True,
                "scientific_result": audit.NO_SCIENTIFIC_RESULT,
                "input_binding": {
                    "topology": "FORMAL_HOME_DUAL_ROOT",
                    "formal_closure_path": str(formal_closure.resolve()),
                    "formal_closure_sha256": _sha(formal_closure),
                    "attempt05_predictions_read": False,
                    "prediction_outputs_reused": False,
                },
            },
        },
        "blockers": [],
        "can_request_pilot_execution_authorization": True,
        "pilot_execution_authorized": False,
        "pilot_partition_frozen": False,
        "pilot_started": False,
        "confirmation_started": False,
        "automatic_progression_allowed": False,
        "next_action": (
            "OBTAIN_EXPLICIT_PILOT_GPU_BUDGET_AUTHORIZATION_THEN_FREEZE_PARTITION"
        ),
        "auditor": {
            "source_path": "tests/gate2_pilot_readiness_audit.py",
            "source_sha256": "6" * 64,
            "source_commit": SOURCE_COMMIT,
            "source_tree": SOURCE_TREE,
            "source_tracked": True,
            "worktree_clean": True,
        },
    }


def _resource_payload() -> dict[str, object]:
    models = [
        {
            "model_id": "VGGT",
            "source_commit": "7" * 40,
            "checkpoint_sha256": "8" * 64,
            "adapter_id": "georeliab-v4-vggt-adapter",
            "adapter_sha256": "9" * 64,
        },
        {
            "model_id": "MASt3R",
            "source_commit": "a" * 40,
            "checkpoint_sha256": "b" * 64,
            "adapter_id": "georeliab-v4-mast3r-adapter",
            "adapter_sha256": "c" * 64,
        },
    ]
    return {
        "schema_version": "georeliab-v4-pilot-resource-candidate-1.0",
        "validation_class": audit.DEVELOPMENT_EVIDENCE_ONLY,
        "scientific_result": audit.NO_SCIENTIFIC_RESULT,
        "model_order": list(SCIENTIFIC_MODELS),
        "models": models,
        "model_bindings_sha256": _sha_value(models),
        "pilot_inputs_materialized": False,
        "pilot_started": False,
    }


def _case(tmp_path: Path, *, suffix: str = "01") -> dict[str, Path]:
    home = tmp_path / f"home-{suffix}"
    formal_closure = home / "georeliab-gate2-formal/manifests/formal-gate2-input-closure.json"
    _write_json(
        formal_closure,
        {
            "schema_version": "georeliab-v4-formal-gate2-input-closure-1.0",
            "status": "FORMAL_GATE2_INPUT_CLOSURE_READY",
            "schedule_identity_sha256": SCHEDULE_SHA,
            "production_source_commit": SOURCE_COMMIT,
            "production_source_tree": SOURCE_TREE,
            "attempt05_predictions_read": False,
            "prediction_outputs_reused": False,
            "scientific_result": audit.NO_SCIENTIFIC_RESULT,
        },
    )
    admission = home / "georeliab-v4-pilot/readiness/formal-admission/report.json"
    _write_json(admission, _admission_payload(formal_closure))
    protocol = home / "georeliab-v4-pilot/readiness/protocol/GEORELIAB_V4_PROTOCOL.md"
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_bytes(PROTOCOL_BYTES)
    resource = home / "georeliab-v4-pilot/readiness/resources/pilot-resource-candidate.json"
    _write_json(resource, _resource_payload())
    run_root = home / f"georeliab-v4-pilot/runs/development-pilot-{suffix}"
    request = home / f"georeliab-v4-pilot/requests/pilot-authorization-{suffix}.json"
    _write_json(
        request,
        {
            "schema_version": "georeliab-v4-pilot-authorization-request-1.0",
            "status": "USER_APPROVED_PILOT_GPU_BUDGET_REQUEST",
            "validation_class": audit.DEVELOPMENT_EVIDENCE_ONLY,
            "scientific_result": audit.NO_SCIENTIFIC_RESULT,
            "user_approved": True,
            "authorization_note": "user-approved-development-pilot-contract-fixture",
            "requested_scope": "PRIMARY_3_SCENE_60_UNIT_PILOT_ONLY",
            "admission_report_path": str(admission.resolve()),
            "admission_report_sha256": _sha(admission),
            "resource_manifest_path": str(resource.resolve()),
            "resource_manifest_sha256": _sha(resource),
            "schedule_identity_sha256": SCHEDULE_SHA,
            "protocol_path": str(protocol.resolve()),
            "protocol_sha256": PROTOCOL_SHA,
            "production_source_commit": SOURCE_COMMIT,
            "production_source_tree": SOURCE_TREE,
            "model_bindings_sha256": _resource_payload()[
                "model_bindings_sha256"
            ],
            "gpu_uuid": GPU_UUID,
            "physical_gpu_index": 0,
            "physical_gpu_count": 1,
            "model_order": list(SCIENTIFIC_MODELS),
            "sequential_model_execution": True,
            "sequential_unit_execution": True,
            "fallback_allowed": False,
            "auto_retry_allowed": False,
            "device_switch_allowed": False,
            "grid_reduction_allowed": False,
            "downstream_advance_allowed": False,
            "extension_authorized": False,
            "confirmation_authorized": False,
            "max_gpu_seconds": 21_600,
            "max_wall_seconds": 43_200,
            "max_storage_bytes": 25 * 1024**3,
            "run_root": str(run_root.resolve()),
            "forbidden_provenance": [],
            "pilot_inputs_materialized": False,
            "pilot_started": False,
        },
    )
    return {
        "home": home,
        "formal_closure": formal_closure,
        "admission": admission,
        "protocol": protocol,
        "resource": resource,
        "request": request,
        "run_root": run_root,
        "output": home
        / f"georeliab-v4-pilot/readiness/pilot-freeze-authorization-{suffix}",
    }


def _prepare(case: dict[str, Path]) -> dict[str, object]:
    return audit.prepare_pilot_partition_authorization(
        admission_report_path=case["admission"],
        approval_request_path=case["request"],
        output_root=case["output"],
        home_root=case["home"],
    )


def test_ready_admission_freezes_authorized_partition_without_starting_pilot(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)

    result = _prepare(case)

    assert result["status"] == audit.READY_STATUS
    assert result["pilot_execution_authorized"] is True
    assert result["pilot_partition_frozen"] is True
    assert result["pilot_inputs_materialized"] is False
    assert result["pilot_started"] is False
    assert result["automatic_progression_allowed"] is False
    assert not case["run_root"].exists()


def test_admission_state_mutations_fail_closed_before_any_output(
    tmp_path: Path,
) -> None:
    mutations = (
        ("status", "V4_PILOT_ADMISSION_BLOCKED"),
        ("blockers", ["G2_NOT_QUALIFIED"]),
        ("can_request_pilot_execution_authorization", False),
        ("pilot_execution_authorized", True),
        ("pilot_partition_frozen", True),
        ("pilot_started", True),
        ("confirmation_started", True),
        ("automatic_progression_allowed", True),
        ("next_action", "START_PILOT"),
    )
    for index, (field, bad_value) in enumerate(mutations):
        case = _case(tmp_path, suffix=f"admission-{index}")
        payload = _read_json(case["admission"])
        payload[field] = bad_value
        _write_json(case["admission"], payload)
        request = _read_json(case["request"])
        request["admission_report_sha256"] = _sha(case["admission"])
        _write_json(case["request"], request)

        with pytest.raises(audit.PilotFreezeAuthorizationError, match="ADMISSION"):
            _prepare(case)
        assert not case["output"].exists()


def test_admission_digest_and_auditor_lineage_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    stale = _case(tmp_path, suffix="stale")
    stale_payload = _read_json(stale["admission"])
    stale_payload["auditor"]["worktree_clean"] = False
    _write_json(stale["admission"], stale_payload)

    with pytest.raises(audit.PilotFreezeAuthorizationError, match="ADMISSION"):
        _prepare(stale)

    rehashed = _case(tmp_path, suffix="rehashed")
    rehashed_payload = _read_json(rehashed["admission"])
    rehashed_payload["auditor"]["source_tracked"] = False
    _write_json(rehashed["admission"], rehashed_payload)
    request = _read_json(rehashed["request"])
    request["admission_report_sha256"] = _sha(rehashed["admission"])
    _write_json(rehashed["request"], request)

    with pytest.raises(audit.PilotFreezeAuthorizationError, match="AUDITOR"):
        _prepare(rehashed)

    protocol_tamper = _case(tmp_path, suffix="protocol-tamper")
    protocol_tamper["protocol"].write_bytes(PROTOCOL_BYTES + b"tamper\n")
    with pytest.raises(audit.PilotFreezeAuthorizationError, match="PROTOCOL"):
        _prepare(protocol_tamper)


def test_partition_is_deterministic_and_binds_schedule_protocol_and_models(
    tmp_path: Path,
) -> None:
    first = _case(tmp_path, suffix="deterministic-a")
    second = _case(tmp_path, suffix="deterministic-b")

    _prepare(first)
    _prepare(second)
    left = _read_json(first["output"] / "manifests/pilot-partition-manifest.json")
    right = _read_json(second["output"] / "manifests/pilot-partition-manifest.json")

    for field in (
        "partition_sha256",
        "selector_payload_sha256",
        "primary_scene_ids",
        "primary_unit_ids",
    ):
        assert left[field] == right[field]
    assert left["schedule_identity_sha256"] == SCHEDULE_SHA
    assert left["protocol_sha256"] == PROTOCOL_SHA
    assert left["model_bindings_sha256"] == _resource_payload()[
        "model_bindings_sha256"
    ]


def test_partition_inventory_is_exact_ordered_sixty_and_disjoint(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    _prepare(case)
    payload = _read_json(case["output"] / "manifests/pilot-partition-manifest.json")
    expected = build_pilot_partition(SCHEDULE_SHA, protocol_sha256=PROTOCOL_SHA)

    assert payload["primary_scene_ids"] == list(expected.primary_scene_ids)
    assert payload["primary_unit_ids"] == list(expected.primary_unit_ids)
    assert len(payload["primary_scene_ids"]) == 3
    assert len(payload["primary_unit_ids"]) == 60
    assert len(set(payload["primary_unit_ids"])) == 60
    assert payload["state_ids"] == list(SCIENTIFIC_STATES)
    assert payload["model_order"] == list(SCIENTIFIC_MODELS)
    assert payload["disjointness_proof"]["primary_vs_non_primary_disjoint"] is True
    assert payload["disjointness_proof"]["all_twenty_scenes_covered"] is True


def test_partition_tamper_reorder_and_scope_expansion_fail_closed(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    _prepare(case)
    original = _read_json(
        case["output"] / "manifests/pilot-partition-manifest.json"
    )
    mutations = []
    changed_scene = deepcopy(original)
    changed_scene["primary_scene_ids"][0] = 118
    mutations.append(changed_scene)
    reordered = deepcopy(original)
    reordered["primary_unit_ids"][:2] = reversed(reordered["primary_unit_ids"][:2])
    mutations.append(reordered)
    expanded = deepcopy(original)
    expanded["primary_unit_ids"].append(expanded["extension_unit_ids"][0])
    mutations.append(expanded)

    for payload in mutations:
        with pytest.raises(audit.PilotFreezeAuthorizationError, match="PARTITION"):
            audit.validate_pilot_partition_manifest(
                payload,
                expected_schedule_identity_sha256=SCHEDULE_SHA,
                expected_protocol_sha256=PROTOCOL_SHA,
                expected_model_bindings_sha256=_resource_payload()[
                    "model_bindings_sha256"
                ],
            )


def test_attempt05_and_gate2_prediction_receipt_ledger_provenance_is_forbidden(
    tmp_path: Path,
) -> None:
    forbidden_values = (
        "/archive/attempt-05/predictions",
        "/home/hryang/georeliab-gate2-formal/runs/formal-gate2-run-01",
        "local_gate2_recovery_receipt",
    )
    for index, forbidden in enumerate(forbidden_values):
        case = _case(tmp_path, suffix=f"provenance-{index}")
        request = _read_json(case["request"])
        request["forbidden_provenance"] = [
            {"nested": {"prediction_or_receipt_root": forbidden}}
        ]
        _write_json(case["request"], request)

        with pytest.raises(audit.PilotFreezeAuthorizationError, match="PROVENANCE"):
            _prepare(case)
        assert not case["output"].exists()


def test_authorization_binds_user_gpu_budget_partition_and_fresh_run_root(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    _prepare(case)
    authorization = _read_json(
        case["output"] / "manifests/pilot-authorization.json"
    )
    partition = _read_json(
        case["output"] / "manifests/pilot-partition-manifest.json"
    )

    assert authorization["user_approved"] is True
    assert authorization["authorization_note"]
    assert authorization["admission_report_sha256"] == _sha(case["admission"])
    assert authorization["partition_sha256"] == partition["partition_sha256"]
    assert authorization["protocol_path"] == str(case["protocol"].resolve())
    assert authorization["protocol_sha256"] == PROTOCOL_SHA
    assert authorization["gpu_uuid"] == GPU_UUID
    assert authorization["physical_gpu_count"] == 1
    assert authorization["max_gpu_seconds"] == 21_600
    assert authorization["max_wall_seconds"] == 43_200
    assert authorization["max_storage_bytes"] == 25 * 1024**3
    assert authorization["run_root"] == str(case["run_root"].resolve())
    assert authorization["pilot_started"] is False


def test_authorization_relaxation_budget_or_scope_mutation_fails_closed(
    tmp_path: Path,
) -> None:
    mutations = (
        ("user_approved", False),
        ("authorization_note", ""),
        ("physical_gpu_count", 2),
        ("sequential_unit_execution", False),
        ("fallback_allowed", True),
        ("auto_retry_allowed", True),
        ("device_switch_allowed", True),
        ("grid_reduction_allowed", True),
        ("downstream_advance_allowed", True),
        ("extension_authorized", True),
        ("confirmation_authorized", True),
        ("max_gpu_seconds", 21_601),
        ("max_wall_seconds", 43_201),
        ("max_storage_bytes", 25 * 1024**3 + 1),
        ("pilot_started", True),
    )
    for index, (field, bad_value) in enumerate(mutations):
        case = _case(tmp_path, suffix=f"authorization-{index}")
        request = _read_json(case["request"])
        request[field] = bad_value
        _write_json(case["request"], request)

        with pytest.raises(
            audit.PilotFreezeAuthorizationError,
            match="AUTHORIZATION|BUDGET|SCOPE",
        ):
            _prepare(case)
        assert not case["output"].exists()


def test_roots_must_be_home_owned_fresh_and_separate_from_prior_evidence(
    tmp_path: Path,
) -> None:
    cases = []
    outside = _case(tmp_path, suffix="outside")
    request = _read_json(outside["request"])
    request["run_root"] = str((tmp_path / "outside-home/run").resolve())
    _write_json(outside["request"], request)
    cases.append(outside)

    alias = _case(tmp_path, suffix="alias")
    request = _read_json(alias["request"])
    request["run_root"] = str(alias["formal_closure"].parent.parent.resolve())
    _write_json(alias["request"], request)
    cases.append(alias)

    occupied = _case(tmp_path, suffix="occupied")
    occupied["run_root"].mkdir(parents=True)
    cases.append(occupied)

    for case in cases:
        with pytest.raises(audit.PilotFreezeAuthorizationError, match="ROOT"):
            _prepare(case)
        assert not case["output"].exists()


def test_freeze_is_no_clobber_and_cannot_dispatch_gpu_or_materialize_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)

    def forbidden(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess:
        raise AssertionError("partition freeze must not invoke a subprocess")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    _prepare(case)

    assert not case["run_root"].exists()
    with pytest.raises(audit.PilotFreezeAuthorizationError, match="EXISTS"):
        _prepare(case)


def test_bundle_manifest_and_preflight_are_digest_closed_without_progression(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    _prepare(case)

    verified = audit.verify_pilot_freeze_bundle(case["output"])
    preflight = _read_json(case["output"] / "pilot-freeze-audit.json")
    assert verified["status"] == audit.READY_STATUS
    assert preflight["pilot_execution_authorized"] is True
    assert preflight["pilot_partition_frozen"] is True
    assert preflight["pilot_inputs_materialized"] is False
    assert preflight["pilot_started"] is False
    assert preflight["confirmation_started"] is False
    assert preflight["automatic_progression_allowed"] is False
    assert preflight["next_action"] == "MATERIALIZE_AND_AUDIT_FRESH_PILOT_INPUTS"

    authorization = case["output"] / "manifests/pilot-authorization.json"
    authorization.write_text(
        authorization.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(audit.PilotFreezeAuthorizationError, match="MANIFEST|DIGEST"):
        audit.verify_pilot_freeze_bundle(case["output"])
