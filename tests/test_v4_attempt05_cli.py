from __future__ import annotations

import json
from pathlib import Path

import pytest

from georeliab_mve.cli import (
    ATTEMPT05_COMMANDS, _attempt05_authorized_gpu_inventory, build_parser, main,
)
from georeliab_mve.v4_attempt05_execution import V4ExecutionError


def test_attempt05_cli_commands_are_registered() -> None:
    parser = build_parser()
    for command in ATTEMPT05_COMMANDS:
        assert command in parser._subparsers._group_actions[0].choices


def test_attempt05_cli_argument_errors_are_fail_closed(capsys) -> None:
    code = main(["v4-attempt05-preflight"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 2
    assert payload == {
        "attempt_id": "attempt-05",
        "error_type": "ArgumentParserError",
        "reason_code": "V4_ATTEMPT05_CLI_ARGUMENT_ERROR",
        "scientific_result": "NO_SCIENTIFIC_RESULT",
        "status": "FAIL",
    }


def test_attempt05_gpu_inventory_projects_only_the_authorized_device() -> None:
    class Context:
        selected_gpu = {
            "uuid": "GPU-authorized",
            "pci_bus_id": "00000000:4D:00.0",
            "index": 0,
        }

    host_inventory = {
        "hostname": "test-host",
        "devices": [
            {
                "uuid": "GPU-authorized",
                "pci_bus_id": "00000000:4D:00.0",
                "index": 0,
                "compute_process_count": 0,
            },
            {
                "uuid": "GPU-other",
                "pci_bus_id": "00000000:B2:00.0",
                "index": 1,
                "compute_process_count": 1,
            },
        ],
    }

    projected = _attempt05_authorized_gpu_inventory(
        Context(), sampler=lambda: host_inventory
    )

    assert projected["hostname"] == "test-host"
    assert [row["uuid"] for row in projected["devices"]] == ["GPU-authorized"]
    assert len(host_inventory["devices"]) == 2


@pytest.mark.parametrize(
    "devices",
    [
        [],
        [
            {
                "uuid": "GPU-authorized",
                "pci_bus_id": "00000000:4D:00.0",
                "index": 1,
            }
        ],
        [
            {
                "uuid": "GPU-authorized",
                "pci_bus_id": "00000000:4D:00.0",
                "index": 0,
            },
            {
                "uuid": "GPU-authorized",
                "pci_bus_id": "00000000:4D:00.0",
                "index": 0,
            },
        ],
    ],
)
def test_attempt05_gpu_inventory_requires_one_exact_authorized_match(devices) -> None:
    class Context:
        selected_gpu = {
            "uuid": "GPU-authorized",
            "pci_bus_id": "00000000:4D:00.0",
            "index": 0,
        }

    with pytest.raises(
        V4ExecutionError,
        match="V4_ATTEMPT05_PREFLIGHT_GPU_IDENTITY_MISMATCH",
    ):
        _attempt05_authorized_gpu_inventory(
            Context(), sampler=lambda: {"devices": devices}
        )


def test_attempt05_run_invokes_bound_serial_pipeline(monkeypatch, tmp_path, capsys) -> None:
    class Decision:
        status = "PASS"
        reason_code = "NEXT_UNIT_READY"

    class Schedule:
        units = []
        schedule_sha256 = "a" * 64

    class Context:
        gpu_ledger_path = tmp_path / "ledger.jsonl"

    class BindingSet:
        calibration_bindings = {("VGGT", 1): object()}
        scientific_bindings = {("VGGT", 1, "L3"): object()}
        adapter_provider = None

    class Result:
        status = "V4_MVE_COMPLETED"
        calibration_units_completed = 40
        scientific_units_completed = 400
        invalid_scientific_units = 0
        gpu_inference_seconds = 1.0
        wall_runtime_seconds = 2.0
        logical_bytes = 3
        allocated_bytes = 4
        peak_memory_mb = 5.0
        finalizer_result = {"status": "PASS"}

    calls = {"bindings": 0, "executors": 0, "pipeline": 0}
    monkeypatch.setattr("georeliab_mve.cli._attempt05_states", lambda _path: [], raising=False)
    monkeypatch.setattr("georeliab_mve.cli._attempt05_schedule", lambda _path: Schedule(), raising=False)
    monkeypatch.setattr("georeliab_mve.cli._attempt05_split", lambda _path: object(), raising=False)
    monkeypatch.setattr("georeliab_mve.cli._attempt05_calibration_schedule", lambda _path: [], raising=False)
    monkeypatch.setattr(
        "georeliab_mve.cli.authorize_attempt05_next_dispatch",
        lambda **_kwargs: Decision(),
    )

    def fake_bindings(**kwargs):
        calls["bindings"] += 1
        assert kwargs["attempt05_tooling_commit"] == "b" * 40
        assert kwargs["attempt05_tooling_tree"] == "c" * 40
        return BindingSet()

    def fake_executors(**_kwargs):
        calls["executors"] += 1
        return (lambda job: job), (lambda unit, calibration: unit)

    def fake_pipeline(**_kwargs):
        calls["pipeline"] += 1
        return Result()

    monkeypatch.setattr("georeliab_mve.cli.construct_attempt05_runtime_bindings", fake_bindings)
    monkeypatch.setattr("georeliab_mve.cli._attempt05_overlay_config_path", lambda _context: tmp_path / "overlay.toml")
    monkeypatch.setattr("georeliab_mve.cli.make_attempt05_runtime_executors", fake_executors)
    monkeypatch.setattr("georeliab_mve.cli.run_attempt05_pipeline", fake_pipeline)
    monkeypatch.setattr("georeliab_mve.cli.load_attempt05_authorized_context", lambda **_kwargs: Context())
    auth = tmp_path / "authorization.json"
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    closure = input_dir / "v4-attempt05-input-closure.json"
    auth.write_text("{}", encoding="utf-8")
    closure.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "georeliab_mve.cli.validate_attempt05_input_closure",
        lambda _path: {
            "budgeted_input_storage": {
                "scope": "TEST",
                "logical_bytes": 0,
                "allocated_bytes": 0,
            }
        },
    )

    code = main([
        "v4-attempt05-run",
        "--authorization",
        str(auth),
        "--input-closure-dir",
        str(input_dir),
        "--tooling-commit",
        "b" * 40,
        "--tooling-tree",
        "c" * 40,
        "--resume",
    ])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "V4_MVE_COMPLETED"
    assert payload["scientific_result"] == "SCIENTIFIC_RESULT_AVAILABLE"
    assert payload["retry_count"] == 0
    assert calls == {"bindings": 1, "executors": 1, "pipeline": 1}


def test_attempt05_launcher_binds_future_tooling_revision_fail_closed() -> None:
    script = Path("scripts/a100/run_v4_attempt05.sh").read_text(encoding="ascii")

    assert "b7faf490280c50bc821ab157e035a68bc64a3090" not in script
    assert "EXPECTED_TOOLING_COMMIT EXPECTED_TOOLING_TREE" in script
    assert "if [[ $# -ne 7 ]]" in script
    assert "actual_commit=\"$(git rev-parse HEAD)\"" in script
    assert "actual_tree=\"$(git rev-parse 'HEAD^{tree}')\"" in script
    assert "V4_ATTEMPT05_TOOLING_REVISION_REQUIRED" in script
    assert "V4_ATTEMPT05_TOOLING_COMMIT_MISMATCH" in script
    assert "V4_ATTEMPT05_TOOLING_TREE_MISMATCH" in script
    assert "--tooling-commit \"$expected_tooling_commit\"" in script
    assert "--tooling-tree \"$expected_tooling_tree\"" in script
    assert "CUDA_VISIBLE_DEVICES=0" in script
    assert "GEORELIAB_PHYSICAL_GPU_UUID=GPU-6ae218e6-3d51-b748-e308-1f0509e87886" in script
    assert "GEORELIAB_PHYSICAL_GPU_PCI_BUS_ID=00000000:4D:00.0" in script






