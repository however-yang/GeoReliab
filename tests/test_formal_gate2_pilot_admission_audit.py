"""RED contracts for admitting a formal-home Gate 2 dual-root bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import subprocess
from typing import Any

import pytest

from georeliab_mve.v4_attempt05_recovery import (
    AUTHORIZED_GPU_UUID,
    AUTHORIZED_PHYSICAL_GPU_INDEX,
    NO_SCIENTIFIC_RESULT,
    RecoverySmokeManifest,
)
from tests import gate2_pilot_readiness_audit as audit
from tests import local_gate2_prepare as local


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(local._canonical_bytes(value))


def _write_manifest(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            rows.append(f"{_sha(path)}  {path.relative_to(root).as_posix()}")
    (root / "MANIFEST.sha256").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def _git(repo: Path, revision: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repo), "rev-parse", revision),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _png(width: int = 1600, height: int = 1200) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\x0dIHDR"
        + struct.pack(">II", width, height)
    )


def _fake_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    class Index:
        content_length = local.RECTIFIED_BYTES
        etag = local.RECTIFIED_ETAG
        central_directory_sha256 = "a" * 64

    monkeypatch.setattr(local, "index_remote_zip", lambda _url: Index())

    def extract(_url, _index, members, destination):
        rows = {}
        for member in members:
            path = destination / member
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = _png()
            path.write_bytes(payload)
            rows[member] = {
                "raw_sha256": hashlib.sha256(payload).hexdigest(),
                "disposition": "written",
            }
        return rows

    monkeypatch.setattr(local, "extract_range_members_evidence", extract)


def _resource_audit(local_root: Path, overlay: Path) -> dict[str, object]:
    return {
        "schema_version": "georeliab-v4-local-gate2-resource-audit-1.0",
        "status": "LOCAL_GATE2_DEVELOPMENT_RESOURCES_READY",
        "validation_class": "LOCAL_GATE2_DEVELOPMENT_VALIDATION",
        "formal_gate2_equivalent": False,
        "overlay_path": str(overlay.resolve()),
        "overlay_sha256": _sha(overlay),
        "models": [
            {
                "model": "VGGT",
                "source_commit": local.VGGT_SOURCE_COMMIT,
                "checkpoint_sha256": local.VGGT_CHECKPOINT_SHA256,
            },
            {
                "model": "MASt3R",
                "source_commit": local.MAST3R_SOURCE_COMMIT,
                "checkpoint_sha256": local.MAST3R_CHECKPOINT_SHA256,
                "config_sha256": local.MAST3R_CONFIG_SHA256,
                "dust3r_source_commit": local.DUST3R_SOURCE_COMMIT,
                "croco_source_commit": local.CROCO_SOURCE_COMMIT,
            },
        ],
        "scientific_result": NO_SCIENTIFIC_RESULT,
        "root": str(local_root.resolve()),
    }


def _observation_rows(smoke: RecoverySmokeManifest) -> list[dict[str, object]]:
    return [
        {
            "unit_key": unit_key,
            "inference_start_count": smoke.expected_inference_starts[unit_key],
            "completion_count": 1,
            "projection_count": 1,
            "overwrite_count": 0,
            "interruption_phase": smoke.interruption_plan.get(unit_key),
            "gpu_uuid": smoke.gpu_uuid,
            "physical_gpu_index": smoke.physical_gpu_index,
            "canonical_present": True,
            "ledger_committed": True,
            "scientific_marker": NO_SCIENTIFIC_RESULT,
        }
        for unit_key in smoke.unit_keys
    ]


def _formal_home_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    _fake_remote(monkeypatch)

    repo = Path(__file__).resolve().parents[1]
    commit = _git(repo, "HEAD")
    tree = _git(repo, "HEAD^{tree}")
    local_root = home / "georeliab-gate2-local"
    formal_root = home / "georeliab-gate2-formal"
    source_path = local.materialize(local_root)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    overlay = local_root / "manifests" / "local-gate2-overlay.toml"
    resource_path = local_root / "manifests" / "local-gate2-resource-audit.json"
    resource = _resource_audit(local_root, overlay)
    _write_json(resource_path, resource)

    formal_path = formal_root / "manifests" / "formal-gate2-input-closure.json"
    formal = local.build_formal_home_closure_payload(
        source_closure=source,
        source_closure_path=source_path,
        source_closure_sha256=_sha(source_path),
        resource_audit=resource,
        resource_audit_path=resource_path,
        resource_audit_sha256=_sha(resource_path),
        overlay_path=overlay,
        overlay_sha256=_sha(overlay),
        formal_root=formal_root,
        production_source_commit=commit,
        production_source_tree=tree,
        test_only_source_commit=commit,
    )
    _write_json(formal_path, formal)

    run = formal_root / "runs" / "formal-gate2-run-01"
    authorization_path = formal_root / "manifests/formal-gate2-authorization.json"
    authorization = {
        "schema_version": local.FORMAL_GATE2_AUTH_SCHEMA_VERSION,
        "status": local.FORMAL_GATE2_AUTH_STATUS,
        "validation_class": "FORMAL_GATE2",
        "user_approved": True,
        "authorization_note": "user-approved-formal-gate2-fixture-only",
        "formal_closure_sha256": _sha(formal_path),
        "production_source_commit": commit,
        "test_only_source_commit": commit,
        "gpu_uuid": AUTHORIZED_GPU_UUID,
        "physical_gpu_index": AUTHORIZED_PHYSICAL_GPU_INDEX,
        "output_root": str(run.resolve()),
        "max_gpu_seconds": 21_600,
        "max_wall_seconds": 43_200,
        "max_storage_bytes": 25 * 1024**3,
        "gate2_started": False,
        "pilot_started": False,
        "attempt06_started": False,
        "scientific_result": NO_SCIENTIFIC_RESULT,
        "schedule_identity_sha256": formal["schedule_identity_sha256"],
    }
    _write_json(authorization_path, authorization)

    smoke = RecoverySmokeManifest.from_mapping(formal["smoke_manifest"])
    observations = run / "observations"
    for index, row in enumerate(_observation_rows(smoke)):
        _write_json(observations / f"{index:02d}.json", row)
    input_manifest_path = run / "input-manifest.json"
    _write_json(
        input_manifest_path,
        {
            "input_closure_path": str(formal_path.resolve()),
            "input_closure_sha256": _sha(formal_path),
            "runtime_binding_path": str(formal_path.resolve()),
            "runtime_binding_sha256": _sha(formal_path),
            "overlay_config": str(overlay.resolve()),
            "overlay_sha256": _sha(overlay),
            "authorization_manifest": str(authorization_path.resolve()),
            "authorization_sha256": _sha(authorization_path),
            "attempt05_predictions_read": False,
            "prediction_outputs_reused": False,
            "formal_gate2_equivalent": True,
            "validation_class": "FORMAL_GATE2",
            "scientific_result": NO_SCIENTIFIC_RESULT,
        },
    )
    _write_json(run / "smoke-manifest.json", smoke.to_dict())
    _write_json(
        run / "source-manifest.json",
        {
            "canonical_commit": commit,
            "canonical_tree": tree,
            "production_source_zero_drift": True,
            "worktree_status": "",
            "scientific_result": NO_SCIENTIFIC_RESULT,
        },
    )
    qualification = {
        "status": audit.FORMAL_GATE2_STATUS,
        "validation_class": audit.FORMAL_GATE2_CLASS,
        "formal_gate2_equivalent": True,
        "expected_unit_count": 12,
        "unit_count": 12,
        "recovery_runtime_status": audit.GATE2_RUNTIME_STATUS,
        "gate2_started": True,
        "pilot_started": False,
        "attempt06_started": False,
        "scientific_result": NO_SCIENTIFIC_RESULT,
    }
    for field in audit._EMPTY_VIOLATION_FIELDS:
        qualification[field] = []
    _write_json(run / "qualification.json", qualification)
    _write_manifest(run)
    return {
        "repo": repo,
        "commit": commit,
        "local_root": local_root,
        "formal_root": formal_root,
        "source_path": source_path,
        "resource_path": resource_path,
        "overlay_path": overlay,
        "formal_path": formal_path,
        "authorization_path": authorization_path,
        "input_manifest_path": input_manifest_path,
        "run": run,
    }


def _rewrite_run_manifest(bundle: dict[str, Any]) -> None:
    _write_manifest(bundle["run"])


def _refresh_formal_chain(bundle: dict[str, Any]) -> None:
    formal_path = bundle["formal_path"]
    authorization_path = bundle["authorization_path"]
    input_path = bundle["input_manifest_path"]
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["formal_closure_sha256"] = _sha(formal_path)
    _write_json(authorization_path, authorization)
    inputs = json.loads(input_path.read_text(encoding="utf-8"))
    inputs["input_closure_sha256"] = _sha(formal_path)
    inputs["runtime_binding_sha256"] = _sha(formal_path)
    inputs["authorization_sha256"] = _sha(authorization_path)
    _write_json(input_path, inputs)
    _rewrite_run_manifest(bundle)


def _mutate_bundle(bundle: dict[str, Any], mutation: str) -> None:
    if mutation == "source_digest":
        path = bundle["source_path"]
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return
    if mutation == "resource_digest":
        path = bundle["resource_path"]
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return
    if mutation == "overlay_digest":
        path = bundle["overlay_path"]
        path.write_text(path.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8")
        return
    if mutation == "authorization_digest":
        path = bundle["authorization_path"]
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return
    if mutation == "formal_closure_digest":
        path = bundle["formal_path"]
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return
    if mutation == "resource_promoted":
        resource_path = bundle["resource_path"]
        resource = json.loads(resource_path.read_text(encoding="utf-8"))
        resource["formal_gate2_equivalent"] = True
        _write_json(resource_path, resource)
        formal_path = bundle["formal_path"]
        formal = json.loads(formal_path.read_text(encoding="utf-8"))
        formal["resource_audit_sha256"] = _sha(resource_path)
        _write_json(formal_path, formal)
        _refresh_formal_chain(bundle)
        return
    if mutation == "source_root":
        formal_path = bundle["formal_path"]
        formal = json.loads(formal_path.read_text(encoding="utf-8"))
        formal["source_root"] = str(
            bundle["local_root"].with_name("unbound-local-root").resolve()
        )
        _write_json(formal_path, formal)
        _refresh_formal_chain(bundle)
        return
    if mutation == "authorization_provenance":
        authorization_path = bundle["authorization_path"]
        authorization = json.loads(
            authorization_path.read_text(encoding="utf-8")
        )
        authorization["test_only_source_commit"] = "f" * 40
        _write_json(authorization_path, authorization)
        inputs = json.loads(
            bundle["input_manifest_path"].read_text(encoding="utf-8")
        )
        inputs["authorization_sha256"] = _sha(authorization_path)
        _write_json(bundle["input_manifest_path"], inputs)
        _rewrite_run_manifest(bundle)
        return
    if mutation == "prediction_reuse":
        inputs = json.loads(
            bundle["input_manifest_path"].read_text(encoding="utf-8")
        )
        inputs["prediction_outputs_reused"] = True
        _write_json(bundle["input_manifest_path"], inputs)
        _rewrite_run_manifest(bundle)
        return
    raise AssertionError(f"unknown mutation: {mutation}")


def _audit_g2(bundle: dict[str, Any]) -> dict[str, object]:
    return audit.audit_g2(
        repo=bundle["repo"],
        gate2_root=bundle["formal_root"],
        run_root=bundle["run"],
        target_commit=bundle["commit"],
    )


def test_formal_home_dual_root_is_admission_ready_but_never_authorized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _formal_home_bundle(tmp_path, monkeypatch)

    g2 = _audit_g2(bundle)
    report = audit.build_readiness_report(
        g0={"status": audit.G0_PASS},
        g1={"status": audit.G1_PASS},
        g2=g2,
    )

    assert g2["status"] == audit.G2_FORMAL_PASS
    assert g2["input_binding"]["topology"] == "FORMAL_HOME_DUAL_ROOT"
    assert g2["input_binding"]["formal_root"] == str(
        bundle["formal_root"].resolve()
    )
    assert g2["input_binding"]["source_root"] == str(
        bundle["local_root"].resolve()
    )
    assert g2["input_binding"]["view_count"] == 48
    assert g2["input_binding"]["local_resource_formal_equivalent"] is False
    assert report["status"] == audit.ADMISSION_READY
    assert report["can_request_pilot_execution_authorization"] is True
    assert report["pilot_execution_authorized"] is False
    assert report["automatic_progression_allowed"] is False


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("source_digest", "G2_FORMAL_SOURCE_CLOSURE_DIGEST_MISMATCH"),
        ("resource_digest", "G2_FORMAL_RESOURCE_AUDIT_DIGEST_MISMATCH"),
        ("overlay_digest", "G2_FORMAL_OVERLAY_DIGEST_MISMATCH"),
        ("authorization_digest", "G2_FORMAL_AUTHORIZATION_DIGEST_MISMATCH"),
        ("formal_closure_digest", "G2_INPUT_CLOSURE_DIGEST_MISMATCH"),
        ("resource_promoted", "G2_FORMAL_LOCAL_RESOURCE_PROMOTION_FORBIDDEN"),
        ("source_root", "G2_FORMAL_SOURCE_ROOT_IDENTITY_MISMATCH"),
        (
            "authorization_provenance",
            "G2_FORMAL_AUTHORIZATION_PROVENANCE_MISMATCH",
        ),
        ("prediction_reuse", "G2_PREDICTION_OUTPUT_REUSE_FORBIDDEN"),
    ),
)
def test_formal_home_dual_root_mutations_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    reason: str,
) -> None:
    bundle = _formal_home_bundle(tmp_path, monkeypatch)
    _mutate_bundle(bundle, mutation)

    with pytest.raises(audit.ReadinessAuditError, match=reason):
        _audit_g2(bundle)
