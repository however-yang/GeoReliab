from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import pytest

from georeliab_mve.v4_counterfactuals import (
    SCIENTIFIC_MODELS,
    SCIENTIFIC_STATES,
    TEST_SCENE_IDS,
    ScientificExecutionUnit,
    canonical_json_sha256,
)
from georeliab_mve.v4_metrics import CalibrationWarningSample, linear_quantile
from georeliab_mve.v4_science_lock import (
    V4_PROTOCOL_ID,
    V4_PROTOCOL_PROVENANCE_SCHEMA_VERSION,
    V4_PROTOCOL_SHA256,
    V4_PROTOCOL_VERSION,
)

import georeliab_mve.v4_attempt05_orchestrator as orch


@dataclass(frozen=True)
class FakeSplit:
    calibration: tuple[int, ...]
    fingerprint_sha256: str = "4" * 64
    inventory_sha256: str = "5" * 64


@dataclass(frozen=True)
class FakeTaskRecord:
    model_id: str
    scene_id: int
    state_id: str
    record_sha256: str


def _unit(model_id: str, scene_id: int, state_id: str) -> ScientificExecutionUnit:
    provenance = {
        "schema_version": V4_PROTOCOL_PROVENANCE_SCHEMA_VERSION,
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
        "state_identity_sha256": canonical_json_sha256({"state": [scene_id, state_id]}),
        "pair_identity_sha256": None
        if state_id == "L3"
        else canonical_json_sha256({"pair": [scene_id, state_id]}),
    }
    return ScientificExecutionUnit.from_dict(
        {**payload, "execution_unit_sha256": canonical_json_sha256(payload)}
    )


def _schedule() -> Any:
    units = tuple(
        _unit(model, scene, state)
        for model in SCIENTIFIC_MODELS
        for scene in TEST_SCENE_IDS
        for state in SCIENTIFIC_STATES
    )
    return SimpleNamespace(units=units, schedule_sha256="6" * 64)


def _calibration_schedule(split: FakeSplit) -> tuple[dict[str, Any], ...]:
    rows = []
    for model in SCIENTIFIC_MODELS:
        for scene in split.calibration:
            rows.append(
                {
                    "model_id": model,
                    "scene_id": scene,
                    "state_id": "L3",
                    "calibration_unit_sha256": canonical_json_sha256(
                        {"model": model, "scene": scene, "state": "L3"}
                    ),
                }
            )
    return tuple(rows)


def _sample(model: str, scene: int, split: FakeSplit, score: float) -> CalibrationWarningSample:
    return CalibrationWarningSample(
        model_id=model,
        scene_id=scene,
        state_id="L3",
        warning_score=score,
        split_fingerprint_sha256=split.fingerprint_sha256,
        inventory_sha256=split.inventory_sha256,
    )


def test_orchestrator_runs_frozen_serial_order_and_finalizes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    schedule = _schedule()
    split = FakeSplit((2, 3, 5, 6, 7, 8, 14, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 30, 31))
    events: list[tuple[str, str, int, str]] = []
    record_paths: list[Path] = []
    q90_rows: list[dict[str, Any]] = []

    monkeypatch.setattr(
        orch,
        "create_attempt05_start_receipt",
        lambda **kwargs: {
            "start_receipt_sha256": "a" * 64,
            "calibration_schedule_sha256": canonical_json_sha256(kwargs["calibration_schedule"]),
            "budgeted_input_storage": {"logical_bytes": 0, "allocated_bytes": 0},
        },
    )
    monkeypatch.setattr(
        orch,
        "create_attempt05_q90_freeze_artifact",
        lambda **kwargs: q90_rows.extend(kwargs["q90_calibrations"]) or {"q90_freeze_artifact_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        orch,
        "append_attempt05_ledger_event",
        lambda **kwargs: {"event_sha256": "c" * 64},
    )
    monkeypatch.setattr(orch, "validate_v4_split_assignment", lambda value: None)
    monkeypatch.setattr(orch, "validate_scientific_schedule", lambda value: value)
    def calibration_fitter(samples: Any, split_assignment: Any) -> Any:
        rows = tuple(samples)
        return SimpleNamespace(
            model_id=rows[0].model_id,
            alarm_threshold=linear_quantile(tuple(sorted(float(row.warning_score) for row in rows)), 0.90),
            calibration_identifier=canonical_json_sha256({"model": rows[0].model_id}),
        )

    def calibration_executor(job: orch.Attempt05CalibrationUnit) -> orch.Attempt05CalibrationResult:
        events.append(("calibration", job.model_id, job.scene_id, job.state_id))
        return orch.Attempt05CalibrationResult(
            sample=_sample(job.model_id, job.scene_id, split, float(job.scene_id)),
            gpu_inference_seconds=1.0,
            wall_runtime_seconds=2.0,
            logical_bytes=3,
            allocated_bytes=4,
            peak_memory_mb=5.0,
            artifact_sha256="d" * 64,
        )

    def scientific_executor(unit: ScientificExecutionUnit, calibration: Any) -> orch.Attempt05ScientificResult:
        events.append(("scientific", unit.model_id, unit.scene_id, unit.state_id))
        path = tmp_path / unit.model_id / f"{unit.scene_id}-{unit.state_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        record_paths.append(path)
        return orch.Attempt05ScientificResult(
            record_path=path,
            status="VALID_COMPLETE",
            task_record_sha256="e" * 64,
            gpu_inference_seconds=1.0,
            wall_runtime_seconds=2.0,
            logical_bytes=3,
            allocated_bytes=4,
            peak_memory_mb=5.0,
        )

    finalized: dict[str, Any] = {}
    result = orch.run_attempt05_pipeline(
        authorization_path=tmp_path / "auth.json",
        scientific_schedule=schedule,
        model_independent_states=tuple(object() for _ in range(200)),
        split_assignment=split,  # type: ignore[arg-type]
        calibration_schedule=_calibration_schedule(split),
        attempt05_tooling_commit="1" * 40,
        attempt05_tooling_tree="2" * 40,
        ledger_path=tmp_path / "ledger.jsonl",
        calibration_executor=calibration_executor,
        scientific_executor=scientific_executor,
        finalizer=lambda **kwargs: finalized.setdefault("kwargs", kwargs) or {"status": "V4_MVE_COMPLETED"},
        calibration_fitter=calibration_fitter,
    )

    assert result.status == "V4_MVE_COMPLETED"
    assert len(record_paths) == 400
    assert len(q90_rows) == 2
    assert len(finalized["kwargs"]["record_paths"]) == 400
    assert events[:20] == [("calibration", "VGGT", scene, "L3") for scene in split.calibration]
    assert events[20:40] == [("calibration", "MASt3R", scene, "L3") for scene in split.calibration]
    assert events[40:240] == [
        ("scientific", "VGGT", scene, state)
        for scene in TEST_SCENE_IDS
        for state in SCIENTIFIC_STATES
    ]
    assert events[240:] == [
        ("scientific", "MASt3R", scene, state)
        for scene in TEST_SCENE_IDS
        for state in SCIENTIFIC_STATES
    ]


def test_orchestrator_stops_before_second_model_on_calibration_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    split = FakeSplit((2, 3, 5, 6, 7, 8, 14, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 30, 31))
    seen: list[str] = []
    monkeypatch.setattr(
        orch,
        "create_attempt05_start_receipt",
        lambda **kwargs: {
            "start_receipt_sha256": "a" * 64,
            "budgeted_input_storage": {"logical_bytes": 0, "allocated_bytes": 0},
        },
    )
    monkeypatch.setattr(orch, "append_attempt05_ledger_event", lambda **kwargs: {"event_sha256": "c" * 64})
    monkeypatch.setattr(orch, "validate_v4_split_assignment", lambda value: None)
    monkeypatch.setattr(orch, "validate_scientific_schedule", lambda value: value)

    def calibration_executor(job: orch.Attempt05CalibrationUnit) -> orch.Attempt05CalibrationResult:
        seen.append(job.model_id)
        raise orch.Attempt05OrchestrationError("V4_TEST_CALIBRATION_FAIL")

    with pytest.raises(orch.Attempt05OrchestrationError, match="V4_TEST_CALIBRATION_FAIL"):
        orch.run_attempt05_pipeline(
            authorization_path=tmp_path / "auth.json",
            scientific_schedule=_schedule(),
            model_independent_states=tuple(object() for _ in range(200)),
            split_assignment=split,  # type: ignore[arg-type]
            calibration_schedule=_calibration_schedule(split),
            attempt05_tooling_commit="1" * 40,
            attempt05_tooling_tree="2" * 40,
            ledger_path=tmp_path / "ledger.jsonl",
            calibration_executor=calibration_executor,
            scientific_executor=lambda *_args: (_ for _ in ()).throw(AssertionError("scientific must not run")),
            finalizer=lambda **_kwargs: {},
        )
    assert seen == ["VGGT"]


def test_orchestrator_blocks_at_target_budget_without_consuming_hard_ceiling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    split = FakeSplit((2, 3, 5, 6, 7, 8, 14, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 30, 31))
    monkeypatch.setattr(
        orch,
        "create_attempt05_start_receipt",
        lambda **kwargs: {
            "start_receipt_sha256": "a" * 64,
            "budgeted_input_storage": {"logical_bytes": 0, "allocated_bytes": 0},
        },
    )
    monkeypatch.setattr(orch, "append_attempt05_ledger_event", lambda **kwargs: {"event_sha256": "c" * 64})
    monkeypatch.setattr(orch, "validate_v4_split_assignment", lambda value: None)
    monkeypatch.setattr(orch, "validate_scientific_schedule", lambda value: value)

    def calibration_executor(job: orch.Attempt05CalibrationUnit) -> orch.Attempt05CalibrationResult:
        return orch.Attempt05CalibrationResult(
            sample=_sample(job.model_id, job.scene_id, split, 1.0),
            gpu_inference_seconds=35 * 3600,
            wall_runtime_seconds=1.0,
            logical_bytes=1,
            allocated_bytes=1,
            peak_memory_mb=1.0,
            artifact_sha256="d" * 64,
        )

    with pytest.raises(orch.Attempt05OrchestrationError, match="V4_REAUTHORIZE_GPUH_TARGET_REACHED"):
        orch.run_attempt05_pipeline(
            authorization_path=tmp_path / "auth.json",
            scientific_schedule=_schedule(),
            model_independent_states=tuple(object() for _ in range(200)),
            split_assignment=split,  # type: ignore[arg-type]
            calibration_schedule=_calibration_schedule(split),
            attempt05_tooling_commit="1" * 40,
            attempt05_tooling_tree="2" * 40,
            ledger_path=tmp_path / "ledger.jsonl",
            calibration_executor=calibration_executor,
            scientific_executor=lambda *_args: (_ for _ in ()).throw(AssertionError("scientific must not run")),
            finalizer=lambda **_kwargs: {},
        )

def test_serial_adapter_provider_keeps_one_model_loaded_and_closes_before_switch() -> None:
    events: list[tuple[str, str]] = []

    class Adapter:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def close(self) -> None:
            events.append(("close", self.model_id))

    def factory(model_id: str, _context: Any) -> Adapter:
        events.append(("open", model_id))
        return Adapter(model_id)

    provider = orch._SerialAdapterProvider(
        context=object(),  # type: ignore[arg-type]
        adapter_factory=factory,
    )
    first = provider.adapter_for("VGGT")
    assert provider.adapter_for("VGGT") is first
    second = provider.adapter_for("MASt3R")
    assert second is not first
    provider.close()

    assert events == [
        ("open", "VGGT"),
        ("close", "VGGT"),
        ("open", "MASt3R"),
        ("close", "MASt3R"),
    ]


def _resume_totals(
    *,
    calibration_keys: frozenset[tuple[str, int, str]] = frozenset(),
    scientific_keys: frozenset[tuple[str, int, str]] = frozenset(),
) -> Any:
    return SimpleNamespace(
        gpu_inference_seconds=0.0,
        wall_runtime_seconds=0.0,
        logical_bytes=0,
        allocated_bytes=0,
        calibration_units_completed=len(calibration_keys),
        scientific_units_completed=len(scientific_keys),
        invalid_units=0,
        peak_memory_mb=0.0,
        run_started=True,
        finalized=False,
        calibration_unit_keys=calibration_keys,
        scientific_unit_keys=scientific_keys,
    )


def test_resume_blocks_promoted_calibration_artifact_without_ledger_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    split = FakeSplit(
        (2, 3, 5, 6, 7, 8, 14, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 30, 31)
    )
    monkeypatch.setattr(
        orch,
        "create_attempt05_start_receipt",
        lambda **_kwargs: {
            "start_receipt_sha256": "a" * 64,
            "budgeted_input_storage": {"logical_bytes": 0, "allocated_bytes": 0},
        },
    )
    monkeypatch.setattr(orch, "rehydrate_attempt05_ledger_totals", lambda _path: _resume_totals())
    monkeypatch.setattr(orch, "validate_v4_split_assignment", lambda _value: None)
    monkeypatch.setattr(orch, "validate_scientific_schedule", lambda value: value)

    def resumed(job: orch.Attempt05CalibrationUnit) -> orch.Attempt05CalibrationResult:
        return orch.Attempt05CalibrationResult(
            sample=_sample(job.model_id, job.scene_id, split, 1.0),
            gpu_inference_seconds=0.0,
            wall_runtime_seconds=0.0,
            logical_bytes=0,
            allocated_bytes=0,
            peak_memory_mb=0.0,
            artifact_sha256="d" * 64,
            resumed=True,
        )

    with pytest.raises(
        orch.Attempt05OrchestrationError,
        match="V4_ATTEMPT05_ARTIFACT_LEDGER_MISMATCH",
    ):
        orch.run_attempt05_pipeline(
            authorization_path=tmp_path / "auth.json",
            scientific_schedule=_schedule(),
            model_independent_states=tuple(object() for _ in range(200)),
            split_assignment=split,  # type: ignore[arg-type]
            calibration_schedule=_calibration_schedule(split),
            attempt05_tooling_commit="1" * 40,
            attempt05_tooling_tree="2" * 40,
            ledger_path=tmp_path / "ledger.jsonl",
            calibration_executor=resumed,
            scientific_executor=lambda *_args: (_ for _ in ()).throw(
                AssertionError("scientific must not run")
            ),
            finalizer=lambda **_kwargs: {},
            resume=True,
        )


def test_resume_blocks_promoted_scientific_artifact_without_ledger_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    split = FakeSplit(
        (2, 3, 5, 6, 7, 8, 14, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 30, 31)
    )
    calibration_keys = frozenset(
        (model, scene, "L3")
        for model in SCIENTIFIC_MODELS
        for scene in split.calibration
    )
    monkeypatch.setattr(
        orch,
        "create_attempt05_start_receipt",
        lambda **_kwargs: {
            "start_receipt_sha256": "a" * 64,
            "calibration_schedule_sha256": "b" * 64,
            "budgeted_input_storage": {"logical_bytes": 0, "allocated_bytes": 0},
        },
    )
    monkeypatch.setattr(
        orch,
        "rehydrate_attempt05_ledger_totals",
        lambda _path: _resume_totals(calibration_keys=calibration_keys),
    )
    monkeypatch.setattr(orch, "create_attempt05_q90_freeze_artifact", lambda **_kwargs: {})
    monkeypatch.setattr(orch, "validate_v4_split_assignment", lambda _value: None)
    monkeypatch.setattr(orch, "validate_scientific_schedule", lambda value: value)

    def calibration(job: orch.Attempt05CalibrationUnit) -> orch.Attempt05CalibrationResult:
        return orch.Attempt05CalibrationResult(
            sample=_sample(job.model_id, job.scene_id, split, float(job.scene_id)),
            gpu_inference_seconds=0.0,
            wall_runtime_seconds=0.0,
            logical_bytes=0,
            allocated_bytes=0,
            peak_memory_mb=0.0,
            artifact_sha256="d" * 64,
            resumed=True,
        )

    def scientific(
        unit: ScientificExecutionUnit,
        _calibration: Any,
    ) -> orch.Attempt05ScientificResult:
        return orch.Attempt05ScientificResult(
            record_path=tmp_path / "promoted.json",
            status="RESUMED_VALID_COMPLETE",
            task_record_sha256="e" * 64,
            gpu_inference_seconds=0.0,
            wall_runtime_seconds=0.0,
            logical_bytes=0,
            allocated_bytes=0,
            peak_memory_mb=0.0,
            resumed=True,
        )

    def fit(samples: Any, _split: Any) -> Any:
        rows = tuple(samples)
        return SimpleNamespace(
            model_id=rows[0].model_id,
            alarm_threshold=1.0,
            calibration_identifier="f" * 64,
        )

    with pytest.raises(
        orch.Attempt05OrchestrationError,
        match="V4_ATTEMPT05_ARTIFACT_LEDGER_MISMATCH",
    ):
        orch.run_attempt05_pipeline(
            authorization_path=tmp_path / "auth.json",
            scientific_schedule=_schedule(),
            model_independent_states=tuple(object() for _ in range(200)),
            split_assignment=split,  # type: ignore[arg-type]
            calibration_schedule=_calibration_schedule(split),
            attempt05_tooling_commit="1" * 40,
            attempt05_tooling_tree="2" * 40,
            ledger_path=tmp_path / "ledger.jsonl",
            calibration_executor=calibration,
            scientific_executor=scientific,
            finalizer=lambda **_kwargs: {},
            calibration_fitter=fit,
            resume=True,
        )


def _sha_path(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_binary_ply(path: Path) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    rows = np.zeros(2, dtype=dtype)
    rows["x"] = [0.0, 1.0]
    rows["y"] = [0.0, 1.0]
    rows["z"] = [0.0, 1.0]
    rows["nx"] = 1.0
    rows["ny"] = 0.0
    rows["nz"] = 0.0
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "element vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    path.write_bytes(header + rows.tobytes())


def _write_mask(path: Path) -> None:
    import numpy as np
    import scipy.io

    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(
        path,
        {
            "ObsMask": np.ones((3, 3, 3), dtype=bool),
            "BB": np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
            "Res": np.array([[1.0]]),
        },
    )


def _write_camera(path: Path, view_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"1 0 0 {-view_id}.0 0 1 0 0 0 0 1 0\n", encoding="ascii")


def _binding_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, tuple[Any, ...], Any, FakeSplit, Any]:
    from georeliab_mve.v4_counterfactuals import AssetEvidence, build_scientific_schedule, materialize_dtu_state_identity

    root = tmp_path / "materialized"
    states = []
    bindings = []
    views = tuple(range(1, 9))
    png_by: dict[tuple[int, str, int], Path] = {}
    for scene in TEST_SCENE_IDS:
        ply = root / "Points" / "stl" / f"stl{scene:03d}_total.ply"
        mask = root / "MVS Data" / "ObsMask" / f"ObsMask{scene}_10.mat"
        _write_binary_ply(ply)
        _write_mask(mask)
        cameras = {}
        for view in views:
            cam = root / "MVS Data" / "Calibration" / "cal18" / f"pos_{view:03d}.txt"
            _write_camera(cam, view)
            cameras[view] = AssetEvidence(f"MVS Data/Calibration/cal18/pos_{view:03d}.txt", _sha_path(cam), str(cam))
        for state in SCIENTIFIC_STATES:
            rgb_inputs = {}
            source_inputs = None
            for view in views:
                png = root / "Rectified" / f"scan{scene}" / state / f"rect_{view:03d}.png"
                png.parent.mkdir(parents=True, exist_ok=True)
                png.write_bytes(f"png-{scene}-{state}-{view}".encode("ascii"))
                png_by[(scene, state, view)] = png
                token = {"L1": "1", "L2": "2", "L3": "3", "L4": "4", "L5": "5", "L6": "6", "L7": "0"}.get(state)
                member = (
                    f"Rectified/scan{scene}/rect_{view:03d}_{token}_r5000.png"
                    if token is not None
                    else f"SyntheticFog/scan{scene}/{state}/rect_{view:03d}_3_r5000.png"
                )
                rgb_inputs[view] = AssetEvidence(member, _sha_path(png), str(png))
            if state.startswith("fog"):
                source_inputs = {
                    view: AssetEvidence(f"Rectified/scan{scene}/rect_{view:03d}_3_r5000.png", _sha_path(png_by[(scene, "L3", view)]), str(png_by[(scene, "L3", view)]))
                    for view in views
                }
            state_obj = materialize_dtu_state_identity(
                source_root=None,
                scene_id=scene,
                state_id=state,
                ordered_view_ids=views,
                rgb_inputs=rgb_inputs,
                cameras=cameras,
                gt_point_cloud=AssetEvidence(f"Points/stl/stl{scene:03d}_total.ply", _sha_path(ply), str(ply)),
                observability_mask=AssetEvidence(f"MVS Data/ObsMask/ObsMask{scene}_10.mat", _sha_path(mask), str(mask)),
                clean_source_inputs=source_inputs,
            )
            states.append(state_obj)
            bindings.append(
                {
                    "state_identity_sha256": state_obj.state_identity_sha256,
                    "scene_id": scene,
                    "state_id": state,
                    "ordered_view_ids": list(views),
                    "views": [
                        {
                            "view_id": view,
                            "member": f"rgb-{view}",
                            "path": str(png_by[(scene, state, view)]),
                            "sha256": _sha_path(png_by[(scene, state, view)]),
                            "source_sha256": _sha_path(png_by[(scene, "L3", view)]) if state.startswith("fog") else None,
                            "width": 1600,
                            "height": 1200,
                        }
                        for view in views
                    ],
                    "cameras": [
                        {
                            "view_id": view,
                            "member": f"cam-{view}",
                            "path": str(root / "MVS Data" / "Calibration" / "cal18" / f"pos_{view:03d}.txt"),
                            "sha256": _sha_path(root / "MVS Data" / "Calibration" / "cal18" / f"pos_{view:03d}.txt"),
                        }
                        for view in views
                    ],
                    "gt_point_cloud": {"path": str(ply), "sha256": _sha_path(ply)},
                    "observability_mask": {"path": str(mask), "sha256": _sha_path(mask)},
                }
            )
    split = FakeSplit((2, 3, 5, 6, 7, 8, 14, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 30, 31))
    for scene in split.calibration:
        source = bindings[0]
        bindings.append({**source, "scene_id": scene, "state_id": "L3"})
    schedule = build_scientific_schedule(states)
    runtime_path = tmp_path / "v4-runtime-state-bindings.json"
    runtime_path.write_text(
        __import__("json").dumps(
            {
                "schema_version": "georeliab-v4-attempt-05-runtime-state-bindings-1.0",
                "attempt_id": "attempt-05",
                "binding_count": len(bindings),
                "bindings": bindings,
            }
        ),
        encoding="utf-8",
    )
    overlay = tmp_path / "overlay.toml"
    overlay.write_text(
        "[runtime]\n"
        "vggt_python='3.10.20'\nvggt_torch='2.3.1+cu121'\n"
        "mast3r_python='3.10.20'\nmast3r_torch='2.5.1+cu121'\n"
        "[resources]\n"
        "vggt_checkpoint_sha256='" + "1" * 64 + "'\n"
        "vggt_source_commit='" + "2" * 40 + "'\n"
        "mast3r_checkpoint_sha256='" + "3" * 64 + "'\n"
        "mast3r_source_commit='" + "4" * 40 + "'\n"
        "mast3r_config_sha256='" + "5" * 64 + "'\n"
        "dust3r_source_commit='" + "6" * 40 + "'\n"
        "croco_source_commit='" + "7" * 40 + "'\n",
        encoding="utf-8",
    )
    context = SimpleNamespace(
        run_root=tmp_path / "run",
        tooling_commit="a" * 40,
        tooling_tree="b" * 40,
        selected_gpu={"index": 0},
    )
    monkeypatch.setattr(orch, "validate_v4_split_assignment", lambda value: None)
    return runtime_path, tuple(states), schedule, split, context, overlay


def test_construct_attempt05_runtime_bindings_materializes_truthful_lazy_bindings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_path, states, schedule, split, context, overlay = _binding_fixture(tmp_path, monkeypatch)
    factory_calls: list[str] = []
    parser_counts = {"ply": 0, "mask": 0, "projection": 0}
    original_ply = orch.parse_dtu_binary_ply
    original_mask = orch._load_dtu_obsmask_mat
    original_projection = orch.parse_dtu_projection

    def counted_ply(path: Path) -> Any:
        parser_counts["ply"] += 1
        return original_ply(path)

    def counted_mask(path: Path) -> Any:
        parser_counts["mask"] += 1
        return original_mask(path)

    def counted_projection(path: Path) -> Any:
        parser_counts["projection"] += 1
        return original_projection(path)

    monkeypatch.setattr(orch, "parse_dtu_binary_ply", counted_ply)
    monkeypatch.setattr(orch, "_load_dtu_obsmask_mat", counted_mask)
    monkeypatch.setattr(orch, "parse_dtu_projection", counted_projection)
    context.selected_gpu["index"] = 7


    class FakeAdapter:
        def __init__(self, model: str) -> None:
            self.model = model

        def predict_sample(self, *_args: Any) -> str:
            return self.model

    bindings = orch.construct_attempt05_runtime_bindings(
        runtime_binding_path=runtime_path,
        model_independent_states=states,
        scientific_schedule=schedule,
        calibration_schedule=_calibration_schedule(split),
        split_assignment=split,  # type: ignore[arg-type]
        context=context,  # type: ignore[arg-type]
        attempt05_tooling_commit="8" * 40,
        attempt05_tooling_tree="9" * 40,
        overlay_config_path=overlay,
        adapter_factory=lambda model, ctx: factory_calls.append(f"{model}:{ctx.device}") or FakeAdapter(model),
    )

    assert len(bindings.calibration_bindings) == 40
    assert len(bindings.scientific_bindings) == 400
    sample = bindings.scientific_bindings[("VGGT", 1, "L3")]
    assert sample.manifest.provenance.project_commit == "8" * 40
    assert sample.manifest.provenance.project_tree == "9" * 40
    assert sample.manifest.provenance.project_commit != context.tooling_commit
    assert sample.manifest.provenance.project_tree != context.tooling_tree
    assert sample.manifest.environment["attempt05_tooling_commit"] == "8" * 40
    assert sample.manifest.environment["attempt05_tooling_tree"] == "9" * 40
    assert sample.manifest.provenance.corruption_manifest_sha256 == _sha_path(runtime_path)
    assert sample.gt_points.shape == (2, 3)
    assert sample.gt_camera_c2w.shape == (8, 4, 4)
    assert sample.observability_mask.ndim == 3
    assert sample.output_dir == context.run_root / "stage" / "SCIENTIFIC_MVE" / "records" / "VGGT" / "scan001"
    assert sample.manifest.environment["device"] == "cuda:0"
    assert parser_counts == {"ply": 20, "mask": 20, "projection": 160}
    same_scene_l2 = bindings.scientific_bindings[("VGGT", 1, "L2")]
    same_scene_mast3r = bindings.scientific_bindings[("MASt3R", 1, "fog-s1")]
    assert same_scene_l2.gt_points is sample.gt_points
    assert same_scene_l2.observability_mask is sample.observability_mask
    assert same_scene_l2.gt_camera_c2w is sample.gt_camera_c2w
    assert same_scene_mast3r.gt_points is sample.gt_points

    assert factory_calls == []
    assert sample.adapter.predict_sample(None, None, None) == "VGGT"
    assert factory_calls == ["VGGT:cuda:0"]
    assert bindings.scientific_bindings[("MASt3R", 1, "L3")].adapter.predict_sample(None, None, None) == "MASt3R"
    assert factory_calls == ["VGGT:cuda:0", "MASt3R:cuda:0"]


def test_construct_attempt05_runtime_bindings_rejects_materialized_digest_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_path, states, schedule, split, context, overlay = _binding_fixture(tmp_path, monkeypatch)
    payload = __import__("json").loads(runtime_path.read_text(encoding="utf-8"))
    Path(payload["bindings"][0]["views"][0]["path"]).write_bytes(b"tampered")

    with pytest.raises(orch.Attempt05OrchestrationError, match="DIGEST_MISMATCH"):
        orch.construct_attempt05_runtime_bindings(
            runtime_binding_path=runtime_path,
            model_independent_states=states,
            scientific_schedule=schedule,
            calibration_schedule=_calibration_schedule(split),
            split_assignment=split,  # type: ignore[arg-type]
            context=context,  # type: ignore[arg-type]
            attempt05_tooling_commit="8" * 40,
            attempt05_tooling_tree="9" * 40,
            overlay_config_path=overlay,
            adapter_factory=lambda *_args: object(),
        )






def test_construct_attempt05_runtime_bindings_rejects_state_camera_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_path, states, schedule, split, context, overlay = _binding_fixture(tmp_path, monkeypatch)
    payload = __import__("json").loads(runtime_path.read_text(encoding="utf-8"))
    drift_camera = tmp_path / "drift" / "pos_001.txt"
    _write_camera(drift_camera, 99)
    for row in payload["bindings"]:
        if row["scene_id"] == 1 and row["state_id"] == "L2":
            row["cameras"][0]["path"] = str(drift_camera)
            row["cameras"][0]["sha256"] = _sha_path(drift_camera)
            break
    runtime_path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    with pytest.raises(orch.Attempt05OrchestrationError, match="CAMERA_DIGEST_MISMATCH"):
        orch.construct_attempt05_runtime_bindings(
            runtime_binding_path=runtime_path,
            model_independent_states=states,
            scientific_schedule=schedule,
            calibration_schedule=_calibration_schedule(split),
            split_assignment=split,  # type: ignore[arg-type]
            context=context,  # type: ignore[arg-type]
            attempt05_tooling_commit="8" * 40,
            attempt05_tooling_tree="9" * 40,
            overlay_config_path=overlay,
            adapter_factory=lambda *_args: object(),
        )

