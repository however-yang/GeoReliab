"""RED contracts for a fresh home-owned formal Gate 2 qualification."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from georeliab_mve.v4_attempt05_recovery import NO_SCIENTIFIC_RESULT
from tests import gate2_gpu_smoke_harness as harness
from tests import local_gate2_prepare as local


PRODUCTION_COMMIT = "6de08f7a89f88de1de79cef09de74b4e909f27b0"
PRODUCTION_TREE = "111078e2f3031061ea8aec8cf7cbf9bea77ebbb7"
TEST_ONLY_COMMIT = "8699ab0ae22c1b2d603b20b248f8924eaeac30e0"
GPU_UUID = "GPU-6ae218e6-3d51-b748-e308-1f0509e87886"


def _required_api(module: object, name: str):
    assert hasattr(module, name), f"RED_MISSING_TEST_ONLY_API:{name}"
    return getattr(module, name)


def _source_closure() -> dict[str, object]:
    smoke = local.local_smoke_manifest()
    bindings = [
        {
            "scene_id": scene_id,
            "state_id": "L3",
            "ordered_view_ids": list(local.LOCAL_ORDERED_VIEW_IDS),
            "views": [
                {
                    "view_id": view_id,
                    "path": f"/home/user/local/data/scene-{scene_id}-{view_id}.png",
                    "sha256": f"{scene_id * 100 + view_id:064x}"[-64:],
                    "source_sha256": f"{scene_id * 100 + view_id:064x}"[-64:],
                    "width": 1600,
                    "height": 1200,
                }
                for view_id in local.LOCAL_ORDERED_VIEW_IDS
            ],
        }
        for scene_id in smoke["scene_ids"]
    ]
    return {
        "schema_version": local.SCHEMA_VERSION,
        "status": local.LOCAL_STATUS,
        "validation_class": "LOCAL_GATE2_DEVELOPMENT_VALIDATION",
        "formal_gate2_equivalent": False,
        "root": "/home/user/local",
        "schedule_identity_sha256": local.LOCAL_SCHEDULE_IDENTITY,
        "scene_ids": smoke["scene_ids"],
        "ordered_view_ids": list(local.LOCAL_ORDERED_VIEW_IDS),
        "smoke_manifest": smoke,
        "bindings": bindings,
        "member_count": 48,
        "attempt05_predictions_read": False,
        "scientific_result": NO_SCIENTIFIC_RESULT,
    }


def _resource_audit() -> dict[str, object]:
    return {
        "schema_version": "georeliab-v4-local-gate2-resource-audit-1.0",
        "status": "LOCAL_GATE2_DEVELOPMENT_RESOURCES_READY",
        "validation_class": "LOCAL_GATE2_DEVELOPMENT_VALIDATION",
        "formal_gate2_equivalent": False,
        "overlay_path": "/home/user/local/manifests/local-gate2-overlay.toml",
        "overlay_sha256": "c" * 64,
        "models": [
            {"model": "VGGT", "checkpoint_sha256": "d" * 64},
            {"model": "MASt3R", "checkpoint_sha256": "e" * 64},
        ],
        "scientific_result": NO_SCIENTIFIC_RESULT,
    }


def _formal_payload(tmp_path: Path) -> dict[str, object]:
    builder = _required_api(local, "build_formal_home_closure_payload")
    formal_root = tmp_path / "formal"
    return builder(
        source_closure=_source_closure(),
        source_closure_path=Path("/home/user/local/manifests/local-gate2-input-closure.json"),
        source_closure_sha256="a" * 64,
        resource_audit=_resource_audit(),
        resource_audit_path=Path("/home/user/local/manifests/local-gate2-resource-audit.json"),
        resource_audit_sha256="b" * 64,
        overlay_path=Path("/home/user/local/manifests/local-gate2-overlay.toml"),
        overlay_sha256="c" * 64,
        formal_root=formal_root,
        production_source_commit=PRODUCTION_COMMIT,
        production_source_tree=PRODUCTION_TREE,
        test_only_source_commit=TEST_ONLY_COMMIT,
    )


def _authorization(payload: dict[str, object], tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": getattr(
            local,
            "FORMAL_GATE2_AUTH_SCHEMA_VERSION",
            "georeliab-v4-formal-gate2-authorization-1.0",
        ),
        "status": getattr(
            local,
            "FORMAL_GATE2_AUTH_STATUS",
            "V4_FORMAL_GATE2_EXECUTION_AUTHORIZED",
        ),
        "validation_class": "FORMAL_GATE2",
        "user_approved": True,
        "authorization_note": "user-approved-fresh-formal-gate2-only",
        "formal_closure_sha256": "f" * 64,
        "production_source_commit": PRODUCTION_COMMIT,
        "test_only_source_commit": TEST_ONLY_COMMIT,
        "gpu_uuid": GPU_UUID,
        "physical_gpu_index": 0,
        "output_root": str(tmp_path / "formal" / "runs" / "formal-gate2-run-01"),
        "max_gpu_seconds": 21_600,
        "max_wall_seconds": 43_200,
        "max_storage_bytes": 25 * 1024**3,
        "gate2_started": False,
        "pilot_started": False,
        "attempt06_started": False,
        "scientific_result": NO_SCIENTIFIC_RESULT,
        "schedule_identity_sha256": payload["schedule_identity_sha256"],
    }


def test_formal_home_identity_is_distinct_from_local_development() -> None:
    assert _required_api(local, "FORMAL_HOME_SCHEMA_VERSION") != local.SCHEMA_VERSION
    assert _required_api(local, "FORMAL_HOME_STATUS") == (
        "V4_FORMAL_HOME_GATE2_INPUT_CLOSURE_READY"
    )
    assert _required_api(local, "FORMAL_HOME_VALIDATION_CLASS") == (
        "FORMAL_GATE2_INPUT_CLOSURE"
    )


def test_formal_payload_revalidates_bits_without_promoting_local_result(
    tmp_path: Path,
) -> None:
    payload = _formal_payload(tmp_path)

    assert payload["schema_version"] == local.FORMAL_HOME_SCHEMA_VERSION
    assert payload["status"] == local.FORMAL_HOME_STATUS
    assert payload["validation_class"] == local.FORMAL_HOME_VALIDATION_CLASS
    assert payload["formal_gate2_equivalent"] is True
    assert payload["source_validation_class"] == "LOCAL_GATE2_DEVELOPMENT_VALIDATION"
    assert payload["input_bits_revalidated"] is True
    assert payload["resource_bits_revalidated"] is True
    assert payload["prediction_outputs_reused"] is False
    assert payload["attempt05_predictions_read"] is False
    assert payload["gate2_started"] is False
    assert payload["pilot_started"] is False
    assert payload["execution_authorized"] is False
    assert payload["scientific_result"] == NO_SCIENTIFIC_RESULT


def test_formal_payload_binds_exact_source_and_all_twelve_units(tmp_path: Path) -> None:
    payload = _formal_payload(tmp_path)

    assert payload["production_source_commit"] == PRODUCTION_COMMIT
    assert payload["production_source_tree"] == PRODUCTION_TREE
    assert payload["test_only_source_commit"] == TEST_ONLY_COMMIT
    assert payload["schedule_identity_sha256"] == local.LOCAL_SCHEDULE_IDENTITY
    assert len(payload["smoke_manifest"]["unit_keys"]) == 12
    assert len(payload["bindings"]) == 6
    assert sum(len(row["views"]) for row in payload["bindings"]) == 48


def test_freeze_formal_is_no_clobber_and_uses_a_separate_home_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freezer = _required_api(local, "write_formal_home_closure")
    monkeypatch.setattr(local, "require_home_owned_root", lambda path: path.resolve())
    formal_root = tmp_path / "formal"
    payload = _formal_payload(tmp_path)

    path = freezer(formal_root, payload)

    assert path == formal_root / "manifests" / "formal-gate2-input-closure.json"
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    with pytest.raises(Exception, match="COLLISION|EXISTS|NO_CLOBBER"):
        freezer(formal_root, payload)


def test_local_closure_cannot_be_promoted_by_flipping_one_field(
    tmp_path: Path,
) -> None:
    validator = _required_api(local, "validate_formal_home_closure_payload")
    relabel = deepcopy(_source_closure())
    relabel["formal_gate2_equivalent"] = True
    relabel["validation_class"] = "FORMAL_GATE2_INPUT_CLOSURE"

    with pytest.raises(Exception, match="SCHEMA|PROVENANCE|REVALID|IDENTITY"):
        validator(relabel, expected_formal_root=tmp_path / "formal")


@pytest.mark.parametrize(
    ("field", "bad_value", "reason"),
    (
        ("user_approved", False, "USER|AUTH"),
        ("formal_closure_sha256", "0" * 64, "CLOSURE|DIGEST"),
        ("gpu_uuid", "GPU-wrong", "GPU"),
        ("physical_gpu_index", 1, "GPU"),
        ("pilot_started", True, "PILOT|DOWNSTREAM"),
        ("scientific_result", "SCIENTIFIC", "SCIENTIFIC"),
    ),
)
def test_machine_readable_authorization_fails_closed(
    tmp_path: Path,
    field: str,
    bad_value: object,
    reason: str,
) -> None:
    validator = _required_api(local, "validate_formal_gate2_authorization")
    payload = _formal_payload(tmp_path)
    authorization = _authorization(payload, tmp_path)
    authorization[field] = bad_value

    with pytest.raises(Exception, match=reason):
        validator(
            authorization,
            expected_closure_sha256="f" * 64,
            expected_output_root=tmp_path
            / "formal"
            / "runs"
            / "formal-gate2-run-01",
        )


def test_formal_home_parser_requires_authorization_manifest(tmp_path: Path) -> None:
    args = harness.build_parser().parse_args(
        [
            "formal-home-run",
            "--input-closure-dir",
            str(tmp_path / "formal" / "manifests"),
            "--overlay-config",
            str(tmp_path / "local" / "manifests" / "local-gate2-overlay.toml"),
            "--output-root",
            str(tmp_path / "formal" / "runs" / "formal-gate2-run-01"),
            "--authorization-manifest",
            str(tmp_path / "formal" / "manifests" / "formal-gate2-authorization.json"),
        ]
    )

    assert args.home_owned_formal is True
    assert args.local_development is False
    assert args.authorization_manifest.name == "formal-gate2-authorization.json"
    assert not hasattr(args, "authorization_note")


def test_formal_home_worker_is_formal_but_uses_frozen_home_inputs(
    tmp_path: Path,
) -> None:
    command = harness._worker_command(
        SimpleNamespace(
            home_owned_formal=True,
            local_development=False,
            output_root=tmp_path / "formal" / "runs" / "one",
            input_closure_dir=tmp_path / "formal" / "manifests",
            overlay_config=tmp_path / "local" / "manifests" / "overlay.toml",
            model_timeout_seconds=7200.0,
        ),
        tmp_path / "smoke-manifest.json",
        "VGGT|12|L3",
        None,
    )

    assert "_formal-home-unit-worker" in command
    assert "_local-unit-worker" not in command
    assert "_unit-worker" not in command


def test_formal_home_output_guard_accepts_only_fresh_descendant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = _required_api(harness, "_require_formal_home_output")
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    accepted = home / "georeliab-gate2-formal" / "runs" / "one"

    guard(accepted)

    with pytest.raises(Exception, match="HOME|ROOT|OUTPUT"):
        guard(home)
    accepted.mkdir(parents=True)
    with pytest.raises(Exception, match="EXISTS|FRESH|CLOBBER"):
        guard(accepted)


def test_formal_home_keeps_strict_gpu_exclusivity() -> None:
    assert harness._non_target_gpu_processes_forbidden(local_development=False) is True


def test_new_formal_source_anchor_is_exact_and_production_zero_drift() -> None:
    repo = Path(__file__).resolve().parents[1]
    manifest = harness._source_manifest(repo)

    assert manifest["canonical_commit"] == PRODUCTION_COMMIT
    assert manifest["canonical_tree"] == PRODUCTION_TREE
    assert manifest["production_source_zero_drift"] is True
    assert manifest["scientific_result"] == NO_SCIENTIFIC_RESULT
