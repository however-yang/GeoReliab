from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from georeliab_mve import toml_compat as tomllib
from georeliab_mve.v4_attempt05_recovery import NO_SCIENTIFIC_RESULT
from tests import local_gate2_prepare as local


def _png(width: int = 16, height: int = 12) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\x0dIHDR"
        + struct.pack(">II", width, height)
    )


def _allow_test_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local, "require_home_owned_root", lambda value: value.resolve())


def _fake_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    index = SimpleNamespace(
        content_length=local.RECTIFIED_BYTES,
        etag=local.RECTIFIED_ETAG,
        central_directory_sha256="a" * 64,
    )
    monkeypatch.setattr(local, "index_remote_zip", lambda _url: index)

    def extract(_url, _index, members, destination):
        rows = {}
        for member in members:
            path = destination / member
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = _png()
            path.write_bytes(payload)
            rows[member] = {
                "raw_sha256": hashlib.sha256(payload).hexdigest(),
                "disposition": "downloaded",
            }
        return rows

    monkeypatch.setattr(local, "extract_range_members_evidence", extract)


def test_local_schedule_and_selection_are_deterministic() -> None:
    first = local.local_smoke_manifest()
    second = local.local_smoke_manifest()

    assert first == second
    assert len(first["scene_ids"]) == 6
    assert len(first["unit_keys"]) == 12
    assert len(local.LOCAL_SCHEDULE_IDENTITY) == 64
    assert len(local.selected_member_paths()) == 48
    assert len(set(local.selected_member_paths())) == 48


def test_member_paths_are_only_official_l3_pngs() -> None:
    for member in local.selected_member_paths():
        assert member.startswith("Rectified/scan")
        assert member.endswith("_3_r5000.png")
        assert ".." not in Path(member).parts


def test_root_guard_accepts_only_descendant_of_current_home() -> None:
    accepted = local.require_home_owned_root(Path.home() / "georeliab-gate2-local")
    assert Path.home().resolve() in accepted.parents

    with pytest.raises(
        local.LocalGate2PreparationError, match="LOCAL_GATE2_ROOT_OUTSIDE_HOME"
    ):
        local.require_home_owned_root(Path("/tmp/georeliab-gate2-local"))


def test_plan_records_non_formal_scope_and_no_full_archive(
    tmp_path, monkeypatch
) -> None:
    _allow_test_root(monkeypatch)
    plan = local.build_plan(tmp_path / "local")

    assert plan["formal_gate2_equivalent"] is False
    assert plan["archive"]["full_archive_download_forbidden"] is True
    assert plan["member_count"] == 48
    assert plan["scientific_result"] == NO_SCIENTIFIC_RESULT


def test_generated_overlay_keeps_every_path_below_local_root(
    tmp_path, monkeypatch
) -> None:
    _allow_test_root(monkeypatch)
    root = (tmp_path / "local").resolve()
    payload = tomllib.loads(local.local_overlay_text(root))

    for key, value in payload["runtime"].items():
        if key.endswith(("source", "env", "site")) or key == "root":
            assert (
                root in Path(value).resolve().parents or Path(value).resolve() == root
            )
    for key in ("vggt_checkpoint", "mast3r_checkpoint", "mast3r_config"):
        assert root in Path(payload["resources"][key]).resolve().parents
    assert payload["local_development"]["formal_gate2_equivalent"] is False
    assert payload["local_development"]["scientific_result"] == NO_SCIENTIFIC_RESULT


def test_materialize_writes_auditable_48_member_closure(tmp_path, monkeypatch) -> None:
    _allow_test_root(monkeypatch)
    _fake_remote(monkeypatch)
    root = tmp_path / "local"

    closure_path = local.materialize(root)
    payload = local.validate_local_closure(closure_path)

    assert payload["status"] == local.LOCAL_STATUS
    assert payload["formal_gate2_equivalent"] is False
    assert payload["member_count"] == 48
    assert len(payload["bindings"]) == 6
    assert all(len(row["views"]) == 8 for row in payload["bindings"])
    assert (root / "logs" / "preparation.log").is_file()
    assert (root / "logs" / "preparation-events.jsonl").is_file()
    events = [
        json.loads(line)
        for line in (root / "logs" / "preparation-events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["event"] for row in events] == [
        "LOCAL_PREPARATION_START",
        "REMOTE_INDEX_VERIFIED",
        "LOCAL_PREPARATION_COMPLETE",
    ]
    assert all(row["scientific_result"] == NO_SCIENTIFIC_RESULT for row in events)


def test_audit_fails_after_selected_png_tamper(tmp_path, monkeypatch) -> None:
    _allow_test_root(monkeypatch)
    _fake_remote(monkeypatch)
    closure_path = local.materialize(tmp_path / "local")
    payload = json.loads(closure_path.read_text(encoding="utf-8"))
    Path(payload["members"][0]["path"]).write_bytes(_png() + b"tampered")

    with pytest.raises(
        local.LocalGate2PreparationError,
        match="LOCAL_GATE2_MEMBER_DIGEST_MISMATCH",
    ):
        local.validate_local_closure(closure_path)


def test_audit_rejects_formal_or_scientific_claim(tmp_path, monkeypatch) -> None:
    _allow_test_root(monkeypatch)
    _fake_remote(monkeypatch)
    closure_path = local.materialize(tmp_path / "local")
    payload = json.loads(closure_path.read_text(encoding="utf-8"))
    payload["formal_gate2_equivalent"] = True
    closure_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        local.LocalGate2PreparationError,
        match="LOCAL_GATE2_FORMAL_CLAIM_FORBIDDEN",
    ):
        local.validate_local_closure(closure_path)


def test_resource_audit_writes_frozen_evidence(tmp_path, monkeypatch) -> None:
    _allow_test_root(monkeypatch)
    root = tmp_path / "local"
    local.write_plan(root)
    monkeypatch.setattr(
        local,
        "_frozen_runtime_from_config",
        lambda model, _context: SimpleNamespace(model=model),
    )
    monkeypatch.setattr(
        local,
        "verify_frozen_runtime",
        lambda model, _runtime: SimpleNamespace(
            source_commit=(
                local.VGGT_SOURCE_COMMIT
                if model == "VGGT"
                else local.MAST3R_SOURCE_COMMIT
            ),
            checkpoint_sha256=(
                local.VGGT_CHECKPOINT_SHA256
                if model == "VGGT"
                else local.MAST3R_CHECKPOINT_SHA256
            ),
            config_sha256=(None if model == "VGGT" else local.MAST3R_CONFIG_SHA256),
            dust3r_source_commit=(
                None if model == "VGGT" else local.DUST3R_SOURCE_COMMIT
            ),
            croco_source_commit=(
                None if model == "VGGT" else local.CROCO_SOURCE_COMMIT
            ),
            environment={
                "path": f"/test/{model.lower()}",
                "python": "3.10.20",
                "torch": ("2.3.1+cu121" if model == "VGGT" else "2.5.1+cu121"),
                "typing_extensions_version": "4.15.0",
            },
        ),
    )

    path = local.audit_resources(root)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["status"] == "LOCAL_GATE2_DEVELOPMENT_RESOURCES_READY"
    assert payload["formal_gate2_equivalent"] is False
    assert [row["model"] for row in payload["models"]] == ["VGGT", "MASt3R"]
    assert payload["models"][0]["python_version"] == "3.10.20"
    assert payload["models"][1]["torch_version"] == "2.5.1+cu121"
    assert payload["models"][1]["dust3r_source_commit"] == local.DUST3R_SOURCE_COMMIT


def test_cli_help_is_collectable() -> None:
    parser = local.build_parser()
    assert parser.parse_args(["plan", "--root", "/home/user/local"]).command == "plan"
    assert (
        parser.parse_args(["materialize", "--root", "/home/user/local"]).command
        == "materialize"
    )
    assert parser.parse_args(["audit", "--root", "/home/user/local"]).command == "audit"
    assert (
        parser.parse_args(["audit-resources", "--root", "/home/user/local"]).command
        == "audit-resources"
    )
