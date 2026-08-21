"""Fourteen CPU-only contracts for fresh Pilot inputs and resources."""

from __future__ import annotations

from itertools import combinations
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from georeliab_mve.v4_counterfactuals import (
    DTU_LIGHTING_SEMANTIC_TO_PHYSICAL_TOKEN,
    SCIENTIFIC_MODELS,
    SCIENTIFIC_STATES,
)
from georeliab_mve.v4_science_lock import V4_PROTOCOL_SHA256
from tests import pilot_input_resource_audit as audit
from tests import pilot_partition_authorization_audit as freeze


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SOURCE = PROJECT_ROOT / "configs/georeliab_v4_protocol.toml"
SCHEDULE_SHA = "d22bf8fc2cef9f20e92d364a577fca68849f9a6b520ea675578dd61f009f31e7"
SOURCE_COMMIT = "1" * 40
SOURCE_TREE = "2" * 40
GPU_UUID = "GPU-" + "3" * 32
ORDERED_VIEWS = (1, 39, 45, 49, 6, 16, 42, 27)
POSE_PAIRS = tuple(combinations(ORDERED_VIEWS, 2))


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_value(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_asset(path: Path, label: str) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((label + "\n").encode("utf-8"))
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _tree_sha(root: Path) -> str:
    rows = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": _sha(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]
    return _sha_value(rows)


def _resource_case(tmp_path: Path, *, suffix: str = "01") -> dict[str, Path]:
    home = tmp_path / f"home-{suffix}"
    root = home / f"georeliab-v4-pilot/resources/candidate-{suffix}"
    models = []
    for index, model_id in enumerate(SCIENTIFIC_MODELS):
        slug = model_id.casefold()
        source_root = root / slug / "source"
        _write_asset(source_root / "model.py", f"{model_id}:source:model")
        _write_asset(source_root / "runtime.py", f"{model_id}:source:runtime")
        checkpoint = root / slug / "checkpoints/model.pt"
        adapter = root / slug / "adapter.py"
        config = root / slug / "config.json"
        environment = root / slug / "environment.txt"
        _write_asset(checkpoint, f"{model_id}:checkpoint")
        _write_asset(adapter, f"{model_id}:adapter")
        _write_asset(config, f"{model_id}:config")
        _write_asset(environment, f"{model_id}:environment")
        models.append(
            {
                "model_id": model_id,
                "source_root": str(source_root.resolve()),
                "source_commit": f"{index + 4:x}" * 40,
                "source_tree_sha256": _tree_sha(source_root),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha(checkpoint),
                "adapter_id": f"georeliab-v4-{slug}-adapter",
                "adapter_path": str(adapter.resolve()),
                "adapter_sha256": _sha(adapter),
                "config_path": str(config.resolve()),
                "config_sha256": _sha(config),
                "environment_path": str(environment.resolve()),
                "environment_sha256": _sha(environment),
            }
        )
    request = home / f"georeliab-v4-pilot/requests/resource-request-{suffix}.json"
    _write_json(
        request,
        {
            "schema_version": "georeliab-v4-pilot-resource-audit-request-1.0",
            "status": "PILOT_RESOURCE_AUDIT_REQUESTED",
            "validation_class": audit.DEVELOPMENT_EVIDENCE_ONLY,
            "scientific_result": audit.NO_SCIENTIFIC_RESULT,
            "resource_root": str(root.resolve()),
            "model_order": list(SCIENTIFIC_MODELS),
            "models": models,
            "source_evidence_class": "FRESH_PILOT_RESOURCE_CANDIDATE",
            "historical_resource_audit_reused": False,
            "download_allowed": False,
            "gpu_probe_allowed": False,
            "pilot_started": False,
            "automatic_progression_allowed": False,
        },
    )
    return {
        "home": home,
        "root": root,
        "request": request,
        "output": home
        / f"georeliab-v4-pilot/readiness/resources/pilot-resource-candidate-{suffix}.json",
    }


def _prepare_resource(case: dict[str, Path]) -> dict[str, object]:
    return audit.prepare_pilot_resource_candidate(
        request_path=case["request"],
        output_path=case["output"],
        home_root=case["home"],
    )


def _admission_payload(formal_closure: Path) -> dict[str, object]:
    return {
        "schema_version": freeze.ADMISSION_SCHEMA_VERSION,
        "status": freeze.ADMISSION_READY_STATUS,
        "validation_class": freeze.ADMISSION_VALIDATION_CLASS,
        "scientific_result": freeze.NO_SCIENTIFIC_RESULT,
        "gates": {
            "g0": {
                "status": "G0_SOURCE_TOOLCHAIN_PASS",
                "source_commit": SOURCE_COMMIT,
                "source_tree": SOURCE_TREE,
                "worktree_clean": True,
                "scientific_result": freeze.NO_SCIENTIFIC_RESULT,
            },
            "g1": {
                "status": "G1_CPU_FAULT_MATRIX_PASS",
                "scientific_result": freeze.NO_SCIENTIFIC_RESULT,
            },
            "g2": {
                "status": "G2_FORMAL_GPU_SMOKE_PASS",
                "formal_gate2_equivalent": True,
                "scientific_result": freeze.NO_SCIENTIFIC_RESULT,
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
        "next_action": freeze.ADMISSION_NEXT_ACTION,
        "auditor": {
            "source_path": "tests/gate2_pilot_readiness_audit.py",
            "source_sha256": "6" * 64,
            "source_commit": SOURCE_COMMIT,
            "source_tree": SOURCE_TREE,
            "source_tracked": True,
            "worktree_clean": True,
        },
    }


def _freeze_case(
    tmp_path: Path,
    resource_manifest: Path,
    *,
    suffix: str = "01",
) -> dict[str, Path]:
    home = resource_manifest.parents[3]
    formal = home / "georeliab-gate2-formal/manifests/formal-gate2-input-closure.json"
    _write_json(
        formal,
        {
            "schema_version": "georeliab-v4-formal-gate2-input-closure-1.0",
            "schedule_identity_sha256": SCHEDULE_SHA,
            "attempt05_predictions_read": False,
            "prediction_outputs_reused": False,
            "scientific_result": freeze.NO_SCIENTIFIC_RESULT,
        },
    )
    admission = home / "georeliab-v4-pilot/readiness/formal-admission/report.json"
    _write_json(admission, _admission_payload(formal))
    protocol = home / "georeliab-v4-pilot/readiness/protocol/georeliab_v4_protocol.toml"
    protocol.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROTOCOL_SOURCE, protocol)
    resource = _read_json(resource_manifest)
    run_root = home / f"georeliab-v4-pilot/runs/development-pilot-{suffix}"
    request = home / f"georeliab-v4-pilot/requests/pilot-authorization-{suffix}.json"
    _write_json(
        request,
        {
            "schema_version": freeze.REQUEST_SCHEMA_VERSION,
            "status": freeze.REQUEST_STATUS,
            "validation_class": freeze.DEVELOPMENT_EVIDENCE_ONLY,
            "scientific_result": freeze.NO_SCIENTIFIC_RESULT,
            "user_approved": True,
            "authorization_note": "user-approved-test-fixture-only",
            "requested_scope": freeze.REQUESTED_SCOPE,
            "admission_report_path": str(admission.resolve()),
            "admission_report_sha256": _sha(admission),
            "resource_manifest_path": str(resource_manifest.resolve()),
            "resource_manifest_sha256": _sha(resource_manifest),
            "schedule_identity_sha256": SCHEDULE_SHA,
            "protocol_path": str(protocol.resolve()),
            "protocol_sha256": _sha(protocol),
            "production_source_commit": SOURCE_COMMIT,
            "production_source_tree": SOURCE_TREE,
            "model_bindings_sha256": resource["model_bindings_sha256"],
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
    output = home / f"georeliab-v4-pilot/readiness/pilot-freeze-{suffix}"
    freeze.prepare_pilot_partition_authorization(
        admission_report_path=admission,
        approval_request_path=request,
        output_root=output,
        home_root=home,
    )
    return {
        "home": home,
        "formal": formal,
        "admission": admission,
        "protocol": protocol,
        "request": request,
        "run_root": run_root,
        "output": output,
    }


def _evidence(path: Path, member: str) -> dict[str, object]:
    return {"member": member, **_write_asset(path, member)}


def _staged_inventory(
    case: dict[str, Path],
    resource_manifest: Path,
    *,
    suffix: str = "01",
) -> Path:
    partition = _read_json(
        case["output"] / "manifests/pilot-partition-manifest.json"
    )
    root = case["home"] / f"georeliab-v4-pilot/inputs/staged-{suffix}"
    schedule_path = (
        case["home"]
        / f"georeliab-v4-pilot/readiness/schedules/pilot-schedule-views-{suffix}.json"
    )
    _write_json(
        schedule_path,
        {
            "schema_version": "georeliab-v4-pilot-schedule-views-1.0",
            "schedule_identity_sha256": SCHEDULE_SHA,
            "scene_ids": partition["primary_scene_ids"],
            "ordered_views_by_scene": {
                str(scene): list(ORDERED_VIEWS)
                for scene in partition["primary_scene_ids"]
            },
        },
    )
    calibration = root / "fog-calibration.json"
    _write_json(
        calibration,
        {
            "schema_version": "georeliab-v4-fog-calibration-1.0",
            "levels": ["fog-s1", "fog-s2", "fog-s3"],
            "recipe_sha256": "7" * 64,
        },
    )
    cameras = {
        view: _evidence(
            root / f"MVS Data/Calibration/cal18/pos_{view:03d}.txt",
            f"MVS Data/Calibration/cal18/pos_{view:03d}.txt",
        )
        for view in ORDERED_VIEWS
    }
    scenes = []
    for scene in partition["primary_scene_ids"]:
        gt = _evidence(
            root / f"Points/stl/stl{scene:03d}_total.ply",
            f"Points/stl/stl{scene:03d}_total.ply",
        )
        mask = _evidence(
            root / f"MVS Data/ObsMask/ObsMask{scene}_10.mat",
            f"MVS Data/ObsMask/ObsMask{scene}_10.mat",
        )
        states = []
        l3_by_view: dict[int, dict[str, object]] = {}
        for state in SCIENTIFIC_STATES:
            views = []
            for view in ORDERED_VIEWS:
                if state.startswith("L"):
                    token = DTU_LIGHTING_SEMANTIC_TO_PHYSICAL_TOKEN[state]
                    member = (
                        f"Rectified/scan{scene}/rect_{view:03d}_{token}_r5000.png"
                    )
                else:
                    member = (
                        f"SyntheticFog/scan{scene}/{state}/"
                        f"rect_{view:03d}_3_r5000.png"
                    )
                row = _evidence(root / member, member)
                if state == "L3":
                    l3_by_view[view] = row
                if state.startswith("fog-"):
                    row["fog_generation"] = {
                        "source_state_id": "L3",
                        "source_sha256": l3_by_view[view]["sha256"],
                        "calibration_sha256": _sha(calibration),
                        "recipe_sha256": "7" * 64,
                    }
                views.append(row)
            states.append({"state_id": state, "rgb_inputs": views})
        scenes.append(
            {
                "scene_id": scene,
                "ordered_view_ids": list(ORDERED_VIEWS),
                "cameras": [cameras[view] for view in ORDERED_VIEWS],
                "gt_point_cloud": gt,
                "observability_mask": mask,
                "states": states,
            }
        )
    inventory = case["home"] / f"georeliab-v4-pilot/requests/staged-inputs-{suffix}.json"
    _write_json(
        inventory,
        {
            "schema_version": "georeliab-v4-pilot-staged-input-inventory-1.0",
            "status": "PILOT_FRESH_INPUTS_STAGED",
            "validation_class": audit.DEVELOPMENT_EVIDENCE_ONLY,
            "scientific_result": audit.NO_SCIENTIFIC_RESULT,
            "freeze_root": str(case["output"].resolve()),
            "resource_manifest_path": str(resource_manifest.resolve()),
            "resource_manifest_sha256": _sha(resource_manifest),
            "schedule_manifest_path": str(schedule_path.resolve()),
            "schedule_manifest_sha256": _sha(schedule_path),
            "schedule_identity_sha256": SCHEDULE_SHA,
            "protocol_sha256": V4_PROTOCOL_SHA256,
            "input_root": str(root.resolve()),
            "fog_calibration_path": str(calibration.resolve()),
            "fog_calibration_sha256": _sha(calibration),
            "scenes": scenes,
            "attempt05_predictions_read": False,
            "gate2_predictions_read": False,
            "prediction_outputs_reused": False,
            "receipt_or_ledger_reused": False,
            "historical_provenance": [],
            "pilot_started": False,
            "automatic_progression_allowed": False,
        },
    )
    return inventory


def _ready_case(tmp_path: Path, *, suffix: str = "01") -> dict[str, Path]:
    resources = _resource_case(tmp_path, suffix=suffix)
    _prepare_resource(resources)
    frozen = _freeze_case(tmp_path, resources["output"], suffix=suffix)
    inventory = _staged_inventory(frozen, resources["output"], suffix=suffix)
    return {
        **frozen,
        "resource_request": resources["request"],
        "resource_manifest": resources["output"],
        "inventory": inventory,
        "preflight_output": frozen["home"]
        / f"georeliab-v4-pilot/readiness/pilot-input-preflight-{suffix}",
    }


def _prepare_inputs(case: dict[str, Path]) -> dict[str, object]:
    return audit.prepare_pilot_input_resource_preflight(
        freeze_root=case["output"],
        staged_inventory_path=case["inventory"],
        output_root=case["preflight_output"],
        home_root=case["home"],
        source_root=PROJECT_ROOT,
    )


FOG_CALIBRATION_SCENES = (
    2,
    3,
    5,
    7,
    8,
    14,
    17,
    18,
    20,
    21,
    22,
    25,
    26,
    28,
    30,
    31,
    35,
    36,
    37,
    38,
)


def _strengthen_fog_bindings(
    case: dict[str, Path],
    *,
    calibration_scene_ids: tuple[int, ...] = FOG_CALIBRATION_SCENES,
) -> None:
    inventory = _read_json(case["inventory"])
    calibration_path = Path(inventory["fog_calibration_path"])
    calibration = {
        "schema_version": "georeliab-v4-pilot-fog-calibration-1.0",
        "status": "V4_PILOT_FOG_CALIBRATION_FROZEN",
        "validation_class": audit.DEVELOPMENT_EVIDENCE_ONLY,
        "scientific_result": audit.NO_SCIENTIFIC_RESULT,
        "source_split_role": "CALIBRATION",
        "source_scene_ids": list(calibration_scene_ids),
        "pilot_scene_ids": [9, 34, 118],
        "scene_disjoint": not bool(
            set(calibration_scene_ids) & {9, 34, 118}
        ),
        "split_fingerprint_sha256": "8" * 64,
        "inventory_sha256": "9" * 64,
        "d_ref": 12.5,
        "airlight": [0.75, 0.8, 0.85],
        "fog_betas": [0.01, 0.02, 0.03],
        "implementation_version": "georeliab-corruptions-v1",
        "pilot_started": False,
    }
    _write_json(calibration_path, calibration)
    calibration_sha = _sha(calibration_path)
    inventory["fog_calibration_sha256"] = calibration_sha

    for scene in inventory["scenes"]:
        scene_id = scene["scene_id"]
        cameras = {
            view: row
            for view, row in zip(
                scene["ordered_view_ids"], scene["cameras"], strict=True
            )
        }
        states = {row["state_id"]: row for row in scene["states"]}
        l3 = {
            view: row
            for view, row in zip(
                scene["ordered_view_ids"],
                states["L3"]["rgb_inputs"],
                strict=True,
            )
        }
        for state_id in ("fog-s1", "fog-s2", "fog-s3"):
            severity = int(state_id[-1])
            beta = calibration["fog_betas"][severity - 1]
            recipe = {
                "schema_version": "georeliab-v4-pilot-fog-recipe-1.0",
                "renderer": "Koschmieder",
                "state_id": state_id,
                "severity": severity,
                "beta": beta,
                "d_ref": calibration["d_ref"],
                "airlight": calibration["airlight"],
                "implementation_version": calibration[
                    "implementation_version"
                ],
                "corruption_calibration_sha256": calibration_sha,
            }
            recipe_sha = _sha_value(recipe)
            for view, fog_row in zip(
                scene["ordered_view_ids"],
                states[state_id]["rgb_inputs"],
                strict=True,
            ):
                binding = {
                    "schema_version": (
                        "georeliab-v4-pilot-fog-generation-binding-1.0"
                    ),
                    "scene_id": scene_id,
                    "state_id": state_id,
                    "severity": severity,
                    "view_id": view,
                    "fog_member": fog_row["member"],
                    "fog_sha256": fog_row["sha256"],
                    "source_state_id": "L3",
                    "source_member": l3[view]["member"],
                    "source_sha256": l3[view]["sha256"],
                    "gt_sha256": scene["gt_point_cloud"]["sha256"],
                    "camera_sha256": cameras[view]["sha256"],
                    "corruption_calibration_sha256": calibration_sha,
                    "fog_recipe_sha256": recipe_sha,
                    "renderer": "Koschmieder",
                    "implementation_version": calibration[
                        "implementation_version"
                    ],
                    "beta": beta,
                    "d_ref": calibration["d_ref"],
                    "airlight": calibration["airlight"],
                }
                fog_row["fog_generation"] = {
                    **binding,
                    "fog_binding_sha256": _sha_value(binding),
                }
    _write_json(case["inventory"], inventory)


# Resource identity and isolation: 5 contracts.
def test_independent_resource_audit_binds_two_frozen_models(tmp_path: Path) -> None:
    case = _resource_case(tmp_path)
    result = _prepare_resource(case)

    assert result["status"] == audit.RESOURCE_READY
    assert result["model_order"] == list(SCIENTIFIC_MODELS)
    assert [row["model_id"] for row in result["models"]] == list(
        SCIENTIFIC_MODELS
    )
    assert result["historical_resource_audit_reused"] is False
    assert result["pilot_started"] is False
    assert result["automatic_progression_allowed"] is False


def test_resource_source_checkpoint_adapter_config_and_environment_tamper_fails(
    tmp_path: Path,
) -> None:
    fields = (
        "source_root",
        "checkpoint_path",
        "adapter_path",
        "config_path",
        "environment_path",
    )
    for index, field in enumerate(fields):
        case = _resource_case(tmp_path, suffix=f"tamper-{index}")
        request = _read_json(case["request"])
        model = request["models"][0]
        path = Path(model[field])
        target = path / "model.py" if field == "source_root" else path
        target.write_bytes(target.read_bytes() + b"tamper\n")
        with pytest.raises(audit.PilotInputResourceError, match="RESOURCE|DIGEST"):
            _prepare_resource(case)
        assert not case["output"].exists()


def test_attempt05_or_gate2_resource_promotion_is_rejected(tmp_path: Path) -> None:
    mutations = (
        ("source_evidence_class", "LOCAL_GATE2_DEVELOPMENT_VALIDATION"),
        ("historical_resource_audit_reused", True),
    )
    for index, (field, value) in enumerate(mutations):
        case = _resource_case(tmp_path, suffix=f"historical-{index}")
        request = _read_json(case["request"])
        request[field] = value
        _write_json(case["request"], request)
        with pytest.raises(
            audit.PilotInputResourceError, match="HISTORICAL|RESOURCE"
        ):
            _prepare_resource(case)


def test_resource_paths_symlinks_escape_and_unlisted_files_fail_closed(
    tmp_path: Path,
) -> None:
    symlink = _resource_case(tmp_path, suffix="symlink")
    request = _read_json(symlink["request"])
    source = Path(request["models"][0]["source_root"])
    target = source / "model.py"
    target.unlink()
    target.symlink_to(source / "runtime.py")

    extra = _resource_case(tmp_path, suffix="extra")
    request_extra = _read_json(extra["request"])
    Path(request_extra["models"][0]["source_root"], "unlisted.py").write_text(
        "unlisted\n", encoding="utf-8"
    )

    outside = _resource_case(tmp_path, suffix="outside")
    request_outside = _read_json(outside["request"])
    request_outside["resource_root"] = str((tmp_path / "outside").resolve())
    _write_json(outside["request"], request_outside)

    for case in (symlink, extra, outside):
        with pytest.raises(
            audit.PilotInputResourceError, match="RESOURCE|ROOT|SYMLINK|DIGEST"
        ):
            _prepare_resource(case)
        assert not case["output"].exists()


def test_resource_audit_is_no_clobber_cpu_only_and_side_effect_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _resource_case(tmp_path)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("resource audit cannot invoke subprocesses")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    _prepare_resource(case)
    with pytest.raises(audit.PilotInputResourceError, match="EXISTS"):
        _prepare_resource(case)
    assert not (case["home"] / "georeliab-v4-pilot/runs").exists()


# Fresh input identity and coverage: 5 contracts.
def test_valid_frozen_chain_emits_cpu_only_input_preflight_ready(tmp_path: Path) -> None:
    case = _ready_case(tmp_path)
    result = _prepare_inputs(case)

    assert result["status"] == audit.INPUT_PREFLIGHT_READY
    assert result["primary_scene_count"] == 3
    assert result["state_identity_count"] == 30
    assert result["execution_unit_count"] == 60
    assert result["pilot_execution_authorized"] is True
    assert result["pilot_inputs_materialized"] is True
    assert result["pilot_started"] is False
    assert result["gpu_preflight_started"] is False
    assert result["automatic_progression_allowed"] is False
    assert not case["run_root"].exists()


def test_input_closure_has_exact_scenes_states_views_pairs_and_units(
    tmp_path: Path,
) -> None:
    case = _ready_case(tmp_path)
    _prepare_inputs(case)
    root = case["preflight_output"]
    closure = _read_json(root / "manifests/pilot-input-closure.json")
    states = json.loads(
        (root / "manifests/pilot-model-independent-states.json").read_text()
    )
    units = json.loads((root / "manifests/pilot-unit-records.json").read_text())
    partition = _read_json(
        case["output"] / "manifests/pilot-partition-manifest.json"
    )

    assert closure["scene_ids"] == partition["primary_scene_ids"]
    assert closure["state_ids"] == list(SCIENTIFIC_STATES)
    assert len(states) == 30
    assert len(units) == 60
    assert [row["unit_id"] for row in units] == partition["primary_unit_ids"]
    assert all(row["ordered_view_ids"] == list(ORDERED_VIEWS) for row in units)
    assert all(row["pose_pairs"] == [list(pair) for pair in POSE_PAIRS] for row in units)


def test_lighting_names_and_fog_l3_generation_binding_are_frozen(
    tmp_path: Path,
) -> None:
    case = _ready_case(tmp_path)
    _prepare_inputs(case)
    states = json.loads(
        (
            case["preflight_output"]
            / "manifests/pilot-model-independent-states.json"
        ).read_text()
    )
    for row in states:
        if row["state_id"].startswith("L"):
            assert row["source_state_id"] == row["state_id"]
            assert row["source_input_sha256_by_view"] == row["input_sha256_by_view"]
        else:
            assert row["source_state_id"] == "L3"
            assert row["source_input_sha256_by_view"] != row["input_sha256_by_view"]


def test_missing_state_view_asset_tamper_or_fog_source_drift_fails_closed(
    tmp_path: Path,
) -> None:
    mutations = ("state", "view", "asset", "fog-source")
    for index, mutation in enumerate(mutations):
        case = _ready_case(tmp_path, suffix=f"input-drift-{index}")
        inventory = _read_json(case["inventory"])
        if mutation == "state":
            inventory["scenes"][0]["states"].pop()
        elif mutation == "view":
            inventory["scenes"][0]["ordered_view_ids"][:2] = reversed(
                inventory["scenes"][0]["ordered_view_ids"][:2]
            )
        elif mutation == "asset":
            path = Path(
                inventory["scenes"][0]["states"][0]["rgb_inputs"][0]["path"]
            )
            path.write_bytes(path.read_bytes() + b"tamper\n")
        else:
            inventory["scenes"][0]["states"][-1]["rgb_inputs"][0][
                "fog_generation"
            ]["source_sha256"] = "f" * 64
        _write_json(case["inventory"], inventory)
        with pytest.raises(
            audit.PilotInputResourceError, match="INPUT|STATE|VIEW|ASSET|FOG"
        ):
            _prepare_inputs(case)
        assert not case["preflight_output"].exists()


def test_attempt05_gate2_prediction_receipt_ledger_provenance_is_rejected(
    tmp_path: Path,
) -> None:
    mutations = (
        ("attempt05_predictions_read", True),
        ("gate2_predictions_read", True),
        ("prediction_outputs_reused", True),
        ("receipt_or_ledger_reused", True),
        ("historical_provenance", [{"ledger": "attempt-05"}]),
    )
    for index, (field, value) in enumerate(mutations):
        case = _ready_case(tmp_path, suffix=f"provenance-{index}")
        inventory = _read_json(case["inventory"])
        inventory[field] = value
        _write_json(case["inventory"], inventory)
        with pytest.raises(audit.PilotInputResourceError, match="PROVENANCE"):
            _prepare_inputs(case)


# Closure integrity, budget, and preflight: 4 contracts.
def test_input_and_output_roots_are_fresh_home_owned_no_symlink_and_no_clobber(
    tmp_path: Path,
) -> None:
    outside = _ready_case(tmp_path, suffix="outside-root")
    outside["preflight_output"] = tmp_path / "outside-preflight"

    reused = _ready_case(tmp_path, suffix="gate2-root")
    inventory = _read_json(reused["inventory"])
    inventory["input_root"] = str(
        (reused["home"] / "georeliab-gate2-local/data").resolve()
    )
    _write_json(reused["inventory"], inventory)

    symlink = _ready_case(tmp_path, suffix="input-symlink")
    inventory = _read_json(symlink["inventory"])
    target = Path(inventory["scenes"][0]["states"][0]["rgb_inputs"][0]["path"])
    source = Path(inventory["scenes"][0]["states"][0]["rgb_inputs"][1]["path"])
    target.unlink()
    target.symlink_to(source)

    for case in (outside, reused, symlink):
        with pytest.raises(
            audit.PilotInputResourceError, match="ROOT|FRESH|SYMLINK|INPUT"
        ):
            _prepare_inputs(case)

    occupied = _ready_case(tmp_path, suffix="occupied")
    occupied["preflight_output"].mkdir(parents=True)
    with pytest.raises(audit.PilotInputResourceError, match="EXISTS"):
        _prepare_inputs(occupied)


def test_freeze_authorization_resource_schedule_and_protocol_tamper_fails(
    tmp_path: Path,
) -> None:
    mutations = (
        ("schedule_identity_sha256", "8" * 64),
        ("protocol_sha256", "9" * 64),
        ("resource_manifest_sha256", "a" * 64),
        ("freeze_root", "/home/hryang/forged-freeze"),
    )
    for index, (field, value) in enumerate(mutations):
        case = _ready_case(tmp_path, suffix=f"binding-{index}")
        inventory = _read_json(case["inventory"])
        inventory[field] = value
        _write_json(case["inventory"], inventory)
        with pytest.raises(audit.PilotInputResourceError, match="BINDING|DIGEST"):
            _prepare_inputs(case)


def test_storage_accounting_manifest_missing_tamper_and_unlisted_fail_closed(
    tmp_path: Path,
) -> None:
    case = _ready_case(tmp_path)
    _prepare_inputs(case)
    root = case["preflight_output"]
    closure = _read_json(root / "manifests/pilot-input-closure.json")
    assert closure["storage_accounting"]["logical_bytes"] > 0
    assert closure["storage_accounting"]["regular_file_count"] > 0
    audit.verify_pilot_input_resource_bundle(root)

    target = root / "manifests/pilot-unit-records.json"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(audit.PilotInputResourceError, match="MANIFEST|DIGEST"):
        audit.verify_pilot_input_resource_bundle(root)

    extra = _ready_case(tmp_path, suffix="unlisted-bundle")
    _prepare_inputs(extra)
    (extra["preflight_output"] / "unexpected.txt").write_text(
        "unlisted\n", encoding="utf-8"
    )
    with pytest.raises(audit.PilotInputResourceError, match="MANIFEST|UNLISTED"):
        audit.verify_pilot_input_resource_bundle(extra["preflight_output"])


def test_cpu_preflight_cannot_dispatch_gpu_start_pilot_or_auto_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _ready_case(tmp_path)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("CPU preflight cannot invoke subprocesses")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    result = _prepare_inputs(case)

    assert result["gpu_preflight_started"] is False
    assert result["pilot_started"] is False
    assert result["confirmation_started"] is False
    assert result["automatic_progression_allowed"] is False
    assert result["next_action"] == "RUN_SEPARATE_GPU_PREFLIGHT_AFTER_USER_REVIEW"
    assert not case["run_root"].exists()


# Real fog generation and calibration closure: 4 additional contracts.
def test_strong_fog_receipts_bind_every_output_input_and_recipe(
    tmp_path: Path,
) -> None:
    case = _ready_case(tmp_path, suffix="strong-fog")
    _strengthen_fog_bindings(case)

    _prepare_inputs(case)
    root = case["preflight_output"]
    bindings = json.loads(
        (root / "manifests/pilot-fog-generation-bindings.json").read_text()
    )
    closure = _read_json(root / "manifests/pilot-input-closure.json")

    assert len(bindings) == 72
    assert len({row["fog_binding_sha256"] for row in bindings}) == 72
    assert {row["renderer"] for row in bindings} == {"Koschmieder"}
    assert {row["implementation_version"] for row in bindings} == {
        "georeliab-corruptions-v1"
    }
    assert closure["fog_generation_binding_count"] == 72
    assert closure["fog_generation_inventory_sha256"] == _sha(
        root / "manifests/pilot-fog-generation-bindings.json"
    )


def test_fog_receipt_or_declared_output_digest_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    valid = _ready_case(tmp_path, suffix="fog-valid-before-tamper")
    _strengthen_fog_bindings(valid)
    _prepare_inputs(valid)

    for index, mutation in enumerate(("receipt", "declared-output")):
        case = _ready_case(tmp_path, suffix=f"fog-binding-tamper-{index}")
        _strengthen_fog_bindings(case)
        inventory = _read_json(case["inventory"])
        fog = inventory["scenes"][0]["states"][-1]["rgb_inputs"][0]
        if mutation == "receipt":
            fog["fog_generation"]["camera_sha256"] = "a" * 64
        else:
            path = Path(fog["path"])
            path.write_bytes(path.read_bytes() + b"re-encoded\n")
            fog["sha256"] = _sha(path)
        _write_json(case["inventory"], inventory)
        with pytest.raises(
            audit.PilotInputResourceError,
            match="FOG.*(BINDING|DIGEST)|INPUT.*ASSET",
        ):
            _prepare_inputs(case)


def test_fog_calibration_requires_frozen_disjoint_nonpilot_split(
    tmp_path: Path,
) -> None:
    valid = _ready_case(tmp_path, suffix="fog-calibration-valid")
    _strengthen_fog_bindings(valid)
    _prepare_inputs(valid)

    overlap = _ready_case(tmp_path, suffix="fog-calibration-overlap")
    _strengthen_fog_bindings(
        overlap,
        calibration_scene_ids=(*FOG_CALIBRATION_SCENES[:-1], 9),
    )
    with pytest.raises(
        audit.PilotInputResourceError,
        match="FOG_CALIBRATION.*(SPLIT|OVERLAP|DISJOINT)",
    ):
        _prepare_inputs(overlap)


def test_fog_binding_manifest_missing_duplicate_and_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    case = _ready_case(tmp_path, suffix="fog-binding-manifest")
    _strengthen_fog_bindings(case)
    _prepare_inputs(case)
    root = case["preflight_output"]
    binding_path = root / "manifests/pilot-fog-generation-bindings.json"

    rows = json.loads(binding_path.read_text())
    assert len(rows) == 72
    rows[-1] = rows[0]
    _write_json(binding_path, rows)
    manifest = root / "MANIFEST.sha256"
    text = manifest.read_text(encoding="ascii")
    old = next(
        line.split("  ", 1)[0]
        for line in text.splitlines()
        if line.endswith("pilot-fog-generation-bindings.json")
    )
    manifest.write_text(text.replace(old, _sha(binding_path)), encoding="ascii")

    with pytest.raises(
        audit.PilotInputResourceError,
        match="FOG.*(DUPLICATE|INVENTORY)|MANIFEST.*FOG",
    ):
        audit.verify_pilot_input_resource_bundle(root)
