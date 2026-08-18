"""RED contracts for CPU-only freezing of the formal home Gate 2 closure."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import struct

import pytest

from georeliab_mve.v4_attempt05_recovery import NO_SCIENTIFIC_RESULT, sha256_file
from tests import local_gate2_prepare as local


PRODUCTION_COMMIT = "6de08f7a89f88de1de79cef09de74b4e909f27b0"
PRODUCTION_TREE = "111078e2f3031061ea8aec8cf7cbf9bea77ebbb7"
TEST_ONLY_COMMIT = "101aec8da452eea444e1c213148aac33952a9c98"
AUDIT_PASS = "V4_FORMAL_HOME_GATE2_INPUT_CLOSURE_AUDIT_PASS"


def _required_api(name: str):
    assert hasattr(local, name), f"RED_MISSING_TEST_ONLY_API:{name}"
    return getattr(local, name)


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


def _write_resource_audit(source_root: Path, overlay: Path) -> Path:
    payload = {
        "schema_version": "georeliab-v4-local-gate2-resource-audit-1.0",
        "status": "LOCAL_GATE2_DEVELOPMENT_RESOURCES_READY",
        "validation_class": "LOCAL_GATE2_DEVELOPMENT_VALIDATION",
        "formal_gate2_equivalent": False,
        "overlay_path": str(overlay.resolve()),
        "overlay_sha256": sha256_file(overlay),
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
    }
    path = source_root / "manifests" / "local-gate2-resource-audit.json"
    path.write_bytes(local._canonical_bytes(payload))
    return path


def _source_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    _fake_remote(monkeypatch)
    source_root = home / "georeliab-gate2-local"
    local.materialize(source_root)
    overlay = local.write_local_overlay(source_root)
    _write_resource_audit(source_root, overlay)
    return source_root, home / "georeliab-gate2-formal"


def _freeze(source_root: Path, formal_root: Path) -> Path:
    freezer = _required_api("freeze_formal_home_closure")
    return freezer(
        source_root=source_root,
        formal_root=formal_root,
        production_source_commit=PRODUCTION_COMMIT,
        production_source_tree=PRODUCTION_TREE,
        test_only_source_commit=TEST_ONLY_COMMIT,
    )


def _audit(formal_root: Path) -> dict[str, object]:
    auditor = _required_api("audit_formal_home_closure")
    return auditor(
        formal_root=formal_root,
        expected_production_source_commit=PRODUCTION_COMMIT,
        expected_production_source_tree=PRODUCTION_TREE,
        expected_test_only_source_commit=TEST_ONLY_COMMIT,
    )


def test_freeze_cli_requires_explicit_roots_and_source_identities(tmp_path: Path) -> None:
    args = local.build_parser().parse_args(
        [
            "freeze-formal-home",
            "--source-root",
            str(tmp_path / "local"),
            "--formal-root",
            str(tmp_path / "formal"),
            "--production-source-commit",
            PRODUCTION_COMMIT,
            "--production-source-tree",
            PRODUCTION_TREE,
            "--test-only-source-commit",
            TEST_ONLY_COMMIT,
        ]
    )

    assert args.command == "freeze-formal-home"
    assert args.source_root == tmp_path / "local"
    assert args.formal_root == tmp_path / "formal"
    assert args.production_source_commit == PRODUCTION_COMMIT
    assert args.production_source_tree == PRODUCTION_TREE
    assert args.test_only_source_commit == TEST_ONLY_COMMIT
    assert not hasattr(args, "authorization_note")


def test_audit_cli_requires_all_three_expected_source_identities(tmp_path: Path) -> None:
    args = local.build_parser().parse_args(
        [
            "audit-formal-home",
            "--formal-root",
            str(tmp_path / "formal"),
            "--expected-production-source-commit",
            PRODUCTION_COMMIT,
            "--expected-production-source-tree",
            PRODUCTION_TREE,
            "--expected-test-only-source-commit",
            TEST_ONLY_COMMIT,
        ]
    )

    assert args.command == "audit-formal-home"
    assert args.formal_root == tmp_path / "formal"
    assert args.expected_production_source_commit == PRODUCTION_COMMIT
    assert args.expected_production_source_tree == PRODUCTION_TREE
    assert args.expected_test_only_source_commit == TEST_ONLY_COMMIT


def test_freeze_revalidates_bits_and_writes_a_distinct_formal_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, formal_root = _source_fixture(tmp_path, monkeypatch)

    closure_path = _freeze(source_root, formal_root)
    payload = json.loads(closure_path.read_text(encoding="utf-8"))

    assert closure_path == formal_root / "manifests/formal-gate2-input-closure.json"
    assert payload["schema_version"] == local.FORMAL_HOME_SCHEMA_VERSION
    assert payload["status"] == local.FORMAL_HOME_STATUS
    assert payload["validation_class"] == local.FORMAL_HOME_VALIDATION_CLASS
    assert payload["source_validation_class"] == "LOCAL_GATE2_DEVELOPMENT_VALIDATION"
    assert payload["input_bits_revalidated"] is True
    assert payload["resource_bits_revalidated"] is True
    assert payload["prediction_outputs_reused"] is False
    assert payload["attempt05_predictions_read"] is False
    assert payload["production_source_commit"] == PRODUCTION_COMMIT
    assert payload["production_source_tree"] == PRODUCTION_TREE
    assert payload["test_only_source_commit"] == TEST_ONLY_COMMIT
    assert payload["gate2_started"] is False
    assert payload["pilot_started"] is False
    assert payload["execution_authorized"] is False
    assert payload["scientific_result"] == NO_SCIENTIFIC_RESULT


def test_freeze_writes_auditable_logs_but_no_authorization_or_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, formal_root = _source_fixture(tmp_path, monkeypatch)

    _freeze(source_root, formal_root)

    events_path = formal_root / "logs/formal-home-freeze-events.jsonl"
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in events] == [
        "SOURCE_CLOSURE_REVALIDATED",
        "RESOURCE_AUDIT_REVALIDATED",
        "FORMAL_HOME_CLOSURE_FROZEN",
    ]
    assert all(row["scientific_result"] == NO_SCIENTIFIC_RESULT for row in events)
    assert (formal_root / "logs/formal-home-freeze.log").is_file()
    assert (formal_root / "FORMAL_FREEZE_MANIFEST.sha256").is_file()
    assert not (formal_root / "manifests/formal-gate2-authorization.json").exists()
    assert not (formal_root / "runs").exists()


def test_freeze_is_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, formal_root = _source_fixture(tmp_path, monkeypatch)
    _freeze(source_root, formal_root)

    with pytest.raises(Exception, match="EXISTS|COLLISION|CLOBBER|FRESH"):
        _freeze(source_root, formal_root)


def test_freeze_rejects_tampered_input_before_creating_formal_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, formal_root = _source_fixture(tmp_path, monkeypatch)
    source = json.loads(
        (source_root / "manifests/local-gate2-input-closure.json").read_text(
            encoding="utf-8"
        )
    )
    Path(source["members"][0]["path"]).write_bytes(_png() + b"tampered")

    with pytest.raises(Exception, match="MEMBER_DIGEST|DIGEST_MISMATCH"):
        _freeze(source_root, formal_root)
    assert not formal_root.exists()


def test_freeze_rejects_invalid_resource_audit_before_creating_formal_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, formal_root = _source_fixture(tmp_path, monkeypatch)
    resource_path = source_root / "manifests/local-gate2-resource-audit.json"
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    resource["status"] = "LOCAL_RESOURCE_NOT_READY"
    resource_path.write_text(json.dumps(resource), encoding="utf-8")

    with pytest.raises(Exception, match="RESOURCE.*IDENTITY|RESOURCE.*READY"):
        _freeze(source_root, formal_root)
    assert not formal_root.exists()


def test_freeze_requires_the_dedicated_fresh_formal_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, formal_root = _source_fixture(tmp_path, monkeypatch)

    with pytest.raises(Exception, match="FORMAL.*ROOT|ROOT.*FORMAL"):
        _freeze(source_root, formal_root.with_name("some-other-formal-root"))


def test_audit_rechecks_source_digests_and_has_no_execution_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, formal_root = _source_fixture(tmp_path, monkeypatch)
    _freeze(source_root, formal_root)

    result = _audit(formal_root)

    assert result["status"] == AUDIT_PASS
    assert result["checked_member_count"] == 48
    assert result["checked_binding_count"] == 6
    assert result["checked_unit_count"] == 12
    assert result["authorization_present"] is False
    assert result["run_root_present"] is False
    assert result["gate2_started"] is False
    assert result["pilot_started"] is False
    assert result["scientific_result"] == NO_SCIENTIFIC_RESULT


def test_audit_rejects_source_closure_digest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, formal_root = _source_fixture(tmp_path, monkeypatch)
    _freeze(source_root, formal_root)
    source_path = source_root / "manifests/local-gate2-input-closure.json"
    source_path.write_text(source_path.read_text(encoding="utf-8") + "\n")

    with pytest.raises(Exception, match="SOURCE.*DIGEST|DIGEST.*SOURCE"):
        _audit(formal_root)


def test_audit_rejects_overlay_digest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, formal_root = _source_fixture(tmp_path, monkeypatch)
    _freeze(source_root, formal_root)
    overlay = source_root / "manifests/local-gate2-overlay.toml"
    overlay.write_text(overlay.read_text(encoding="utf-8") + "# tampered\n")

    with pytest.raises(Exception, match="OVERLAY.*DIGEST|DIGEST.*OVERLAY"):
        _audit(formal_root)


def test_audit_rejects_wrong_expected_test_only_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, formal_root = _source_fixture(tmp_path, monkeypatch)
    _freeze(source_root, formal_root)
    auditor = _required_api("audit_formal_home_closure")

    with pytest.raises(Exception, match="SOURCE|COMMIT|IDENTITY"):
        auditor(
            formal_root=formal_root,
            expected_production_source_commit=PRODUCTION_COMMIT,
            expected_production_source_tree=PRODUCTION_TREE,
            expected_test_only_source_commit="f" * 40,
        )


def test_freezer_source_contains_no_gpu_or_downstream_dispatch() -> None:
    freezer = _required_api("freeze_formal_home_closure")
    source = inspect.getsource(freezer).lower()

    assert "nvidia-smi" not in source
    assert "cuda" not in source
    assert "_run_supervisor" not in source
    assert "formal-home-run" not in source
    assert "pilot" not in source
