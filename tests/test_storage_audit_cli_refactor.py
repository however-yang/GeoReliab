from __future__ import annotations

import hashlib
import json
from pathlib import Path

from georeliab_mve.cli import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_storage_audit_cli_writes_digest_bound_dry_run(
    tmp_path: Path, capsys
) -> None:
    runtime_root = tmp_path / "runtime"
    output_dir = runtime_root / "artifacts"
    runtime_root.mkdir()
    (runtime_root / "manifests").mkdir()
    (runtime_root / "manifests" / "frozen.json").write_text(
        '{"status":"PASS"}\n', encoding="utf-8"
    )

    exit_code = main(
        [
            "storage-audit",
            "--root",
            str(runtime_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    stdout = json.loads(capsys.readouterr().out)
    plan_path = Path(stdout["storage_plan"])
    assert Path(stdout["storage_before"]).is_file()
    assert plan_path.is_file()
    assert stdout["plan_file_sha256"] == hashlib.sha256(
        plan_path.read_bytes()
    ).hexdigest()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["mutation_enabled"] is True


def test_storage_audit_cli_requires_expected_digest_for_apply(
    tmp_path: Path, capsys
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    plan_path = runtime_root / "plan.json"
    plan_path.write_text("{}\n", encoding="utf-8")

    exit_code = main(
        [
            "storage-audit",
            "--root",
            str(runtime_root),
            "--apply-plan",
            str(plan_path),
        ]
    )

    assert exit_code == 2
    assert "--apply-plan requires --expected-plan-sha256" in capsys.readouterr().err


def test_storage_audit_cli_uses_repository_science_lock() -> None:
    assert PROJECT_ROOT.joinpath("configs", "dual_mve_protocol.toml").is_file()
