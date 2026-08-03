"""Serial Attempt-05 MVE orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from . import toml_compat as tomllib
from .adapters import RenderedView
from .audit import _load_dtu_obsmask_mat
from .contracts import (
    PredictionArtifact,
    RunManifest,
    RunMode,
    SampleKey,
    ScientificProvenance,
    ScientificValidity,
)
from .prepared_inputs import parse_dtu_binary_ply, parse_dtu_projection
from .runner import RunnerContext, _worker_python_path
from .v4_attempt05_execution import (
    _validate_calibration_schedule,
    append_attempt05_ledger_event,
    Attempt05AuthorizedContext,
    create_attempt05_q90_freeze_artifact,
    create_attempt05_start_receipt,
    evaluate_attempt05_resource_gate,
    finalize_attempt05_scientific_bundle,
    rehydrate_attempt05_ledger_totals,
)
from .v4_counterfactuals import (
    SCIENTIFIC_MODELS,
    ModelIndependentState,
    ScientificExecutionUnit,
    ScientificSchedule,
    validate_scientific_schedule,
    validate_v4_split_assignment,
)
from .v4_execution import (
    V4ExecutionError,
    admit_existing_task_record,
    canonical_record_path,
)
from .v4_attempt05_runtime import (
    decompose_ordered_dtu_projections,
    execute_attempt05_calibration_l3,
    execute_attempt05_unit,
)
from .v4_metrics import (
    CalibrationWarningSample,
    NativeWarningCalibration,
    fit_native_warning_calibration,
)
from .v4_records import (
    Task3ContractError,
    read_task_audit_record,
    write_task_audit_record,
)

class Attempt05OrchestrationError(V4ExecutionError):
    """Raised when Attempt-05 must stop without admitting partial evidence."""


@dataclass(frozen=True, slots=True)
class Attempt05CalibrationUnit:
    model_id: str
    scene_id: int
    state_id: str
    calibration_unit_sha256: str
    sequence_index: int


@dataclass(frozen=True, slots=True)
class Attempt05CalibrationResult:
    sample: CalibrationWarningSample
    gpu_inference_seconds: float
    wall_runtime_seconds: float
    logical_bytes: int
    allocated_bytes: int
    peak_memory_mb: float
    artifact_sha256: str
    resumed: bool = False


@dataclass(frozen=True, slots=True)
class Attempt05ScientificResult:
    record_path: Path
    status: str
    task_record_sha256: str
    gpu_inference_seconds: float
    wall_runtime_seconds: float
    logical_bytes: int
    allocated_bytes: int
    peak_memory_mb: float
    resumed: bool = False
    projection_logical_bytes: int = 0
    projection_allocated_bytes: int = 0
    projection_created: bool = False



@dataclass(frozen=True, slots=True)
class Attempt05CalibrationRuntimeBinding:
    manifest: Any
    sample_key: Any
    rendered_views: Sequence[Any]
    adapter: Any
    output_dir: Path
    split_fingerprint_sha256: str
    inventory_sha256: str
    resume: bool = False


@dataclass(frozen=True, slots=True)
class Attempt05ScientificRuntimeBinding:
    manifest: Any
    sample_key: Any
    rendered_views: Sequence[Any]
    adapter: Any
    output_dir: Path
    gt_points: Any
    gt_camera_c2w: Any
    observability_mask: Any
    gt_dtu_camera_c2w: Any
    observability_bb: Any | None = None
    observability_res: float | None = None
    resume: bool = False
    legacy_output_dir: Path | None = None
    run_root: Path | None = None

@dataclass(frozen=True, slots=True)
class Attempt05RuntimeBindingSet:
    calibration_bindings: Mapping[tuple[str, int], Attempt05CalibrationRuntimeBinding]
    scientific_bindings: Mapping[tuple[str, int, str], Attempt05ScientificRuntimeBinding]
    adapter_provider: Any | None = None


class _LazySerialAdapter:
    def __init__(self, model_id: str, provider: "_SerialAdapterProvider") -> None:
        self._model_id = model_id
        self._provider = provider

    def predict_sample(
        self,
        manifest: RunManifest,
        sample_key: SampleKey,
        rendered_views: Sequence[RenderedView],
    ) -> PredictionArtifact:
        return self._provider.adapter_for(self._model_id).predict_sample(
            manifest, sample_key, rendered_views
        )


_ATTEMPT05_PERSISTENT_WORKER_CODE = r"""
import json
import os
from pathlib import Path
import sys
import traceback

project, root, output_root, config_path, device, model_id, ready_path = sys.argv[1:]
sys.path.insert(0, project)
from georeliab_mve.adapters import RenderedView
from georeliab_mve.contracts import RunManifest, SampleKey
from georeliab_mve.runner import (
    RunnerContext,
    _verify_current_worker_runtime,
    default_adapter_factory,
)

context = RunnerContext(
    root=Path(root),
    output_root=Path(output_root),
    config_path=Path(config_path),
    device=device,
)
_verify_current_worker_runtime(context, model_id.lower())
adapter = default_adapter_factory(model_id, context)

def atomic_json(path, payload):
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    with partial.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)

atomic_json(ready_path, {"status": "READY", "model_id": model_id})
for raw_line in sys.stdin:
    request_name = raw_line.strip()
    if request_name == "__STOP__":
        break
    if not request_name:
        continue
    request_path = Path(request_name)
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    status_path = payload["status_path"]
    try:
        manifest = RunManifest.from_dict(payload["manifest"])
        sample_key = SampleKey.parse(payload["sample_key"])
        views = tuple(
            RenderedView(
                view_id=int(row["view_id"]),
                png_path=Path(row["png_path"]),
                png_sha256=row["png_sha256"],
                width=row.get("width"),
                height=row.get("height"),
                source_sha256=row.get("source_sha256"),
            )
            for row in payload["rendered_views"]
        )
        prediction = adapter.predict_sample(manifest, sample_key, views)
        atomic_json(payload["result_path"], prediction.to_dict())
        atomic_json(status_path, {"status": "PASS"})
    except BaseException:
        atomic_json(
            status_path,
            {"status": "FAIL", "traceback": traceback.format_exc()},
        )
        break
"""


class _IsolatedAttempt05WorkerAdapter:
    def __init__(
        self,
        model_id: str,
        context: RunnerContext,
        *,
        popen_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    ) -> None:
        self._model_id = model_id
        self._context = context
        self._call_root = (
            self._context.output_root / "stage" / "_attempt05_worker_calls"
        )
        self._call_root.mkdir(parents=True, exist_ok=True)
        self._source_root = Path(__file__).resolve().parents[1]
        model_key = model_id.lower()
        worker_python = _worker_python_path(self._context, model_key)
        token = f"{os.getpid()}-{time.time_ns()}-{model_key}"
        self._ready_path = self._call_root / f"{token}.ready.json"
        self._stdout_handle = (self._call_root / f"{token}.stdout.log").open(
            "w", encoding="utf-8"
        )
        self._stderr_handle = (self._call_root / f"{token}.stderr.log").open(
            "w", encoding="utf-8"
        )
        env = dict(os.environ)
        env["PYTHONNOUSERSITE"] = "1"
        self._process = popen_factory(
            [
                str(worker_python),
                "-I",
                "-B",
                "-c",
                _ATTEMPT05_PERSISTENT_WORKER_CODE,
                str(self._source_root),
                str(self._context.root),
                str(self._context.output_root),
                str(self._context.config_path),
                self._context.device,
                self._model_id,
                str(self._ready_path),
            ],
            cwd=str(self._source_root),
            stdin=subprocess.PIPE,
            stdout=self._stdout_handle,
            stderr=self._stderr_handle,
            text=True,
            env=env,
        )
        self._wait_for_path(
            self._ready_path,
            reason_code="V4_ATTEMPT05_MODEL_WORKER_START_FAILED",
            timeout_seconds=1800.0,
        )

    def _wait_for_path(
        self, path: Path, *, reason_code: str, timeout_seconds: float
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while not path.is_file():
            if self._process.poll() is not None or time.monotonic() >= deadline:
                self.close()
                raise Attempt05OrchestrationError(reason_code)
            time.sleep(0.05)

    @staticmethod
    def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
        partial = path.with_name(path.name + ".partial")
        with partial.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(path)

    def predict_sample(
        self,
        manifest: RunManifest,
        sample_key: SampleKey,
        rendered_views: Sequence[RenderedView],
    ) -> PredictionArtifact:
        token = f"{os.getpid()}-{time.time_ns()}-{self._model_id.lower()}"
        request_path = self._call_root / f"{token}.request.json"
        result_path = self._call_root / f"{token}.prediction.json"
        status_path = self._call_root / f"{token}.status.json"
        request = {
            "model_id": self._model_id,
            "context": {
                "root": str(self._context.root),
                "output_root": str(self._context.output_root),
                "config_path": str(self._context.config_path),
                "device": self._context.device,
            },
            "manifest": manifest.to_dict(),
            "sample_key": str(sample_key),
            "rendered_views": [
                {
                    "view_id": view.view_id,
                    "png_path": str(view.png_path),
                    "png_sha256": view.png_sha256,
                    "width": view.width,
                    "height": view.height,
                    "source_sha256": view.source_sha256,
                }
                for view in rendered_views
            ],
            "result_path": str(result_path),
            "status_path": str(status_path),
        }
        self._atomic_json(request_path, request)
        if self._process.stdin is None:
            raise Attempt05OrchestrationError("V4_ATTEMPT05_MODEL_WORKER_FAILED")
        self._process.stdin.write(str(request_path) + "\n")
        self._process.stdin.flush()
        self._wait_for_path(
            status_path,
            reason_code="V4_ATTEMPT05_MODEL_WORKER_FAILED",
            timeout_seconds=3600.0,
        )
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Attempt05OrchestrationError(
                "V4_ATTEMPT05_MODEL_WORKER_STATUS_INVALID"
            ) from exc
        if status.get("status") != "PASS":
            raise Attempt05OrchestrationError("V4_ATTEMPT05_MODEL_WORKER_FAILED")
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            return PredictionArtifact.from_dict(payload)
        except Exception as exc:
            raise Attempt05OrchestrationError("V4_ATTEMPT05_MODEL_WORKER_PREDICTION_INVALID") from exc

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write("__STOP__\n")
                    process.stdin.flush()
                process.wait(timeout=120)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
        for handle_name in ("_stdout_handle", "_stderr_handle"):
            handle = getattr(self, handle_name, None)
            if handle is not None and not handle.closed:
                handle.close()


def attempt05_isolated_adapter_factory(model_id: str, context: RunnerContext) -> _IsolatedAttempt05WorkerAdapter:
    return _IsolatedAttempt05WorkerAdapter(model_id, context)


class _SerialAdapterProvider:
    def __init__(
        self,
        *,
        context: RunnerContext,
        adapter_factory: Callable[[str, RunnerContext], Any],
    ) -> None:
        self._context = context
        self._adapter_factory = adapter_factory
        self._active_model: str | None = None
        self._active_adapter: Any = None

    def proxy(self, model_id: str) -> _LazySerialAdapter:
        return _LazySerialAdapter(model_id, self)

    def adapter_for(self, model_id: str) -> Any:
        if self._active_model != model_id:
            self.close()
            self._active_adapter = self._adapter_factory(model_id, self._context)
            self._active_model = model_id
        return self._active_adapter

    def close(self) -> None:
        adapter = self._active_adapter
        self._active_adapter = None
        self._active_model = None
        close = getattr(adapter, "close", None)
        if callable(close):
            close()

@dataclass(frozen=True, slots=True)
class Attempt05PipelineResult:
    status: str
    calibration_units_completed: int
    scientific_units_completed: int
    invalid_scientific_units: int
    gpu_inference_seconds: float
    wall_runtime_seconds: float
    logical_bytes: int
    allocated_bytes: int
    peak_memory_mb: float
    finalizer_result: Mapping[str, Any]


@dataclass(slots=True)
class _ResourceTotals:
    gpu_inference_seconds: float = 0.0
    wall_runtime_seconds: float = 0.0
    logical_bytes: int = 0
    allocated_bytes: int = 0
    peak_memory_mb: float = 0.0

    def add(
        self,
        *,
        gpu_inference_seconds: float,
        wall_runtime_seconds: float,
        logical_bytes: int,
        allocated_bytes: int,
        peak_memory_mb: float,
    ) -> None:
        self.gpu_inference_seconds += float(gpu_inference_seconds)
        self.wall_runtime_seconds += float(wall_runtime_seconds)
        self.logical_bytes += int(logical_bytes)
        self.allocated_bytes += int(allocated_bytes)
        self.peak_memory_mb = max(self.peak_memory_mb, float(peak_memory_mb))


CalibrationExecutor = Callable[[Attempt05CalibrationUnit], Attempt05CalibrationResult]
ScientificExecutor = Callable[
    [ScientificExecutionUnit, NativeWarningCalibration], Attempt05ScientificResult
]
CalibrationFitter = Callable[[Sequence[CalibrationWarningSample], Any], NativeWarningCalibration]
Finalizer = Callable[..., Mapping[str, Any]]


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Attempt05OrchestrationError("V4_ATTEMPT05_BINDING_JSON_UNREADABLE") from exc
    if not isinstance(payload, dict):
        raise Attempt05OrchestrationError("V4_ATTEMPT05_BINDING_JSON_INVALID")
    return payload


def _binding_path(asset: Mapping[str, Any], *, label: str) -> Path:
    raw = asset.get("path")
    digest = asset.get("sha256")
    if not isinstance(raw, str) or not isinstance(digest, str):
        raise Attempt05OrchestrationError(f"V4_ATTEMPT05_{label}_BINDING_INVALID")
    path = Path(raw)
    if not path.is_file():
        raise Attempt05OrchestrationError(f"V4_ATTEMPT05_{label}_MISSING")
    if _sha256_file(path) != digest:
        raise Attempt05OrchestrationError(f"V4_ATTEMPT05_{label}_DIGEST_MISMATCH")
    return path


def _view_from_binding(row: Mapping[str, Any]) -> RenderedView:
    path = _binding_path(row, label="VIEW")
    return RenderedView(
        view_id=int(row["view_id"]),
        png_path=path,
        png_sha256=str(row["sha256"]),
        width=int(row["width"]) if row.get("width") is not None else None,
        height=int(row["height"]) if row.get("height") is not None else None,
        source_sha256=(
            str(row["source_sha256"])
            if row.get("source_sha256") is not None
            else None
        ),
    )


def _runtime_rows_by_state(runtime_binding_path: Path) -> dict[tuple[int, str], Mapping[str, Any]]:
    payload = _read_json_object(runtime_binding_path)
    rows = payload.get("bindings")
    if (
        payload.get("schema_version") != "georeliab-v4-attempt-05-runtime-state-bindings-1.0"
        or payload.get("attempt_id") != "attempt-05"
        or not isinstance(rows, list)
        or payload.get("binding_count") != len(rows)
    ):
        raise Attempt05OrchestrationError("V4_ATTEMPT05_BINDING_JSON_INVALID")
    result: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise Attempt05OrchestrationError("V4_ATTEMPT05_BINDING_JSON_INVALID")
        key = (int(row["scene_id"]), str(row["state_id"]))
        if key in result:
            raise Attempt05OrchestrationError("V4_ATTEMPT05_BINDING_DUPLICATE")
        result[key] = row
    return result


def _states_by_key(model_independent_states: Sequence[object]) -> dict[tuple[int, str], ModelIndependentState]:
    states = tuple(
        row if isinstance(row, ModelIndependentState) else ModelIndependentState.from_dict(row)
        for row in model_independent_states
    )
    if len(states) != 200:
        raise Attempt05OrchestrationError("V4_ATTEMPT05_MODEL_INDEPENDENT_STATES_NOT_200")
    result: dict[tuple[int, str], ModelIndependentState] = {}
    for state in states:
        key = (state.scene_id, state.state_id)
        if key in result:
            raise Attempt05OrchestrationError("V4_ATTEMPT05_STATE_DUPLICATE")
        result[key] = state
    return result


def _rendered_views(row: Mapping[str, Any], state: ModelIndependentState | None = None) -> tuple[RenderedView, ...]:
    raw_views = row.get("views")
    ordered = tuple(int(item) for item in row.get("ordered_view_ids", ()))
    if not isinstance(raw_views, list) or len(raw_views) != 8 or len(ordered) != 8:
        raise Attempt05OrchestrationError("V4_ATTEMPT05_VIEW_BINDING_INVALID")
    views = tuple(_view_from_binding(view) for view in raw_views if isinstance(view, Mapping))
    if len(views) != 8 or tuple(view.view_id for view in views) != ordered:
        raise Attempt05OrchestrationError("V4_ATTEMPT05_VIEW_BINDING_INVALID")
    if state is not None:
        expected = {item.view_id: item.sha256 for item in state.input_sha256_by_view}
        if any(expected.get(view.view_id) != view.png_sha256 for view in views):
            raise Attempt05OrchestrationError("V4_ATTEMPT05_STATE_VIEW_DIGEST_MISMATCH")
    return views


def _gt_payload(row: Mapping[str, Any], state: ModelIndependentState | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    gt_asset = row.get("gt_point_cloud")
    mask_asset = row.get("observability_mask")
    raw_cameras = row.get("cameras")
    ordered = tuple(int(item) for item in row.get("ordered_view_ids", ()))
    if not isinstance(gt_asset, Mapping) or not isinstance(mask_asset, Mapping) or not isinstance(raw_cameras, list):
        raise Attempt05OrchestrationError("V4_ATTEMPT05_GT_BINDING_INVALID")
    gt_path = _binding_path(gt_asset, label="GT_POINT_CLOUD")
    mask_path = _binding_path(mask_asset, label="OBSERVABILITY_MASK")
    if state is not None and (
        str(gt_asset.get("sha256")) != state.gt_point_cloud_sha256
        or str(mask_asset.get("sha256")) != state.observability_mask_sha256
    ):
        raise Attempt05OrchestrationError("V4_ATTEMPT05_STATE_GT_DIGEST_MISMATCH")
    projections: dict[int, Any] = {}
    camera_sha_by_view = {
        item.view_id: item.sha256 for item in state.camera_sha256_by_view
    } if state is not None else {}
    for camera in raw_cameras:
        if not isinstance(camera, Mapping):
            raise Attempt05OrchestrationError("V4_ATTEMPT05_CAMERA_BINDING_INVALID")
        view_id = int(camera["view_id"])
        camera_path = _binding_path(camera, label="CAMERA")
        if camera_sha_by_view and camera_sha_by_view.get(view_id) != str(camera.get("sha256")):
            raise Attempt05OrchestrationError("V4_ATTEMPT05_STATE_CAMERA_DIGEST_MISMATCH")
        projections[view_id] = parse_dtu_projection(camera_path)
    decomposed = decompose_ordered_dtu_projections(
        projections,
        ordered_view_ids=ordered,
    )
    camera_c2w = np.asarray([item.camera_to_world for item in decomposed], dtype=np.float64)
    obs_mask, bb, res = _load_dtu_obsmask_mat(mask_path)
    return (
        parse_dtu_binary_ply(gt_path),
        camera_c2w,
        obs_mask,
        camera_c2w.copy(),
        bb,
        float(res),
    )


def _scene_gt_signature(row: Mapping[str, Any], state: ModelIndependentState) -> tuple[Any, ...]:
    gt_asset = row.get("gt_point_cloud")
    mask_asset = row.get("observability_mask")
    raw_cameras = row.get("cameras")
    ordered = tuple(int(item) for item in row.get("ordered_view_ids", ()))
    if not isinstance(gt_asset, Mapping) or not isinstance(mask_asset, Mapping) or not isinstance(raw_cameras, list):
        raise Attempt05OrchestrationError("V4_ATTEMPT05_GT_BINDING_INVALID")
    gt_path = _binding_path(gt_asset, label="GT_POINT_CLOUD")
    mask_path = _binding_path(mask_asset, label="OBSERVABILITY_MASK")
    if (
        str(gt_asset.get("sha256")) != state.gt_point_cloud_sha256
        or str(mask_asset.get("sha256")) != state.observability_mask_sha256
    ):
        raise Attempt05OrchestrationError("V4_ATTEMPT05_STATE_GT_DIGEST_MISMATCH")
    camera_sha_by_view = {item.view_id: item.sha256 for item in state.camera_sha256_by_view}
    camera_rows: list[tuple[int, str, str]] = []
    for camera in raw_cameras:
        if not isinstance(camera, Mapping):
            raise Attempt05OrchestrationError("V4_ATTEMPT05_CAMERA_BINDING_INVALID")
        view_id = int(camera["view_id"])
        camera_path = _binding_path(camera, label="CAMERA")
        camera_sha = str(camera.get("sha256"))
        if camera_sha_by_view.get(view_id) != camera_sha:
            raise Attempt05OrchestrationError("V4_ATTEMPT05_STATE_CAMERA_DIGEST_MISMATCH")
        camera_rows.append((view_id, camera_sha, str(camera_path)))
    if tuple(view_id for view_id, _sha, _path in camera_rows) != ordered:
        raise Attempt05OrchestrationError("V4_ATTEMPT05_VIEW_BINDING_INVALID")
    return (
        int(state.scene_id),
        ordered,
        str(gt_asset.get("sha256")),
        str(gt_path),
        str(mask_asset.get("sha256")),
        str(mask_path),
        tuple(camera_rows),
    )

def _overlay_model_specs(overlay_config_path: Path) -> dict[str, dict[str, str | None]]:
    try:
        payload = tomllib.loads(overlay_config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise Attempt05OrchestrationError("V4_ATTEMPT05_OVERLAY_UNREADABLE") from exc
    runtime = payload.get("runtime")
    resources = payload.get("resources")
    if not isinstance(runtime, Mapping) or not isinstance(resources, Mapping):
        raise Attempt05OrchestrationError("V4_ATTEMPT05_OVERLAY_INVALID")
    overlay_sha = _sha256_file(overlay_config_path)
    return {
        "VGGT": {
            "checkpoint_hash": str(resources["vggt_checkpoint_sha256"]),
            "source_commit": str(resources["vggt_source_commit"]),
            "python": str(runtime["vggt_python"]),
            "torch": str(runtime["vggt_torch"]),
            "environment_lock_sha256": overlay_sha,
            "dust3r_source_commit": None,
            "croco_source_commit": None,
            "config_sha256": None,
        },
        "MASt3R": {
            "checkpoint_hash": str(resources["mast3r_checkpoint_sha256"]),
            "source_commit": str(resources["mast3r_source_commit"]),
            "python": str(runtime["mast3r_python"]),
            "torch": str(runtime["mast3r_torch"]),
            "environment_lock_sha256": overlay_sha,
            "dust3r_source_commit": str(resources["dust3r_source_commit"]),
            "croco_source_commit": str(resources["croco_source_commit"]),
            "config_sha256": str(resources["mast3r_config_sha256"]),
        },
    }


def _rgb_digest(views: Sequence[RenderedView]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for view in views:
        digest.update(str(int(view.view_id)).encode("ascii"))
        digest.update(b"\0")
        digest.update(view.png_sha256.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _manifest(
    *,
    model_id: str,
    sample_key: SampleKey,
    views: Sequence[RenderedView],
    model_specs: Mapping[str, Mapping[str, str | None]],
    context: Attempt05AuthorizedContext,
    attempt05_tooling_commit: str,
    attempt05_tooling_tree: str,
    runtime_binding_sha256: str,
    scientific_schedule_sha256: str,
    device: str,
) -> RunManifest:
    spec = model_specs[model_id]
    provenance = ScientificProvenance(
        project_commit=attempt05_tooling_commit,
        project_tree=attempt05_tooling_tree,
        model_source_commit=str(spec["source_commit"]),
        environment_lock_sha256=str(spec["environment_lock_sha256"]),
        corruption_manifest_sha256=runtime_binding_sha256,
        split_view_manifest_sha256=scientific_schedule_sha256,
        dust3r_source_commit=spec.get("dust3r_source_commit"),
        croco_source_commit=spec.get("croco_source_commit"),
    )
    return RunManifest(
        run_id=f"attempt05-{model_id.lower()}-{sample_key.scene}-{sample_key.condition}",
        mode=RunMode.REAL,
        scientific_validity=ScientificValidity.SCIENTIFIC,
        model=model_id,
        checkpoint_hash=str(spec["checkpoint_hash"]),
        dataset=sample_key.dataset,
        split=sample_key.split,
        seed=int(sample_key.seed),
        intervention_version="none",
        corruption_version="georeliab-v4-attempt05",
        environment={
            "device": device,
            "python": str(spec["python"]),
            "torch": str(spec["torch"]),
            "attempt05_tooling_commit": attempt05_tooling_commit,
            "attempt05_tooling_tree": attempt05_tooling_tree,
            **({"config_sha256": str(spec["config_sha256"])} if spec.get("config_sha256") else {}),
        },
        rgb_digest=_rgb_digest(views),
        prompt_digest="fixed-empty-prompt",
        decoder_digest="fixed-native-decoder",
        provenance=provenance,
    )


def construct_attempt05_runtime_bindings(
    *,
    runtime_binding_path: Path,
    model_independent_states: Sequence[object],
    scientific_schedule: ScientificSchedule,
    calibration_schedule: Sequence[Mapping[str, object]],
    split_assignment: Any,
    context: Attempt05AuthorizedContext,
    attempt05_tooling_commit: str,
    attempt05_tooling_tree: str,
    overlay_config_path: Path,
    adapter_factory: Callable[[str, RunnerContext], Any] | None = None,
    resume: bool = False,
) -> Attempt05RuntimeBindingSet:
    """Bind Attempt-05 input closure files to production runtime executors."""

    try:
        validate_v4_split_assignment(split_assignment)
    except Exception as exc:
        raise Attempt05OrchestrationError("V4_ATTEMPT05_SPLIT_INVALID") from exc
    schedule = validate_scientific_schedule(scientific_schedule)
    rows_by_state = _runtime_rows_by_state(runtime_binding_path)
    states_by_key = _states_by_key(model_independent_states)
    model_specs = _overlay_model_specs(overlay_config_path)
    runtime_binding_sha256 = _sha256_file(runtime_binding_path)
    device = "cuda:0"
    runner_context = RunnerContext(
        root=Path(__file__).resolve().parents[1],
        output_root=context.run_root,
        config_path=overlay_config_path,
        device=device,
    )
    adapter_provider = _SerialAdapterProvider(
        context=runner_context,
        adapter_factory=adapter_factory or attempt05_isolated_adapter_factory,
    )
    adapters = {model_id: adapter_provider.proxy(model_id) for model_id in SCIENTIFIC_MODELS}

    calibration: dict[tuple[str, int], Attempt05CalibrationRuntimeBinding] = {}
    for job in _calibration_units_by_model(calibration_schedule).values():
        for unit in job:
            row = rows_by_state.get((unit.scene_id, "L3"))
            if row is None:
                raise Attempt05OrchestrationError("V4_ATTEMPT05_CALIBRATION_BINDING_MISSING")
            views = _rendered_views(row)
            sample_key = SampleKey("dtu", "calibration", f"scan{unit.scene_id:03d}", "views-0001", "L3", "0", "0")
            calibration[(unit.model_id, unit.scene_id)] = Attempt05CalibrationRuntimeBinding(
                manifest=_manifest(
                    model_id=unit.model_id,
                    sample_key=sample_key,
                    views=views,
                    model_specs=model_specs,
                    context=context,
                    attempt05_tooling_commit=attempt05_tooling_commit,
                    attempt05_tooling_tree=attempt05_tooling_tree,
                    runtime_binding_sha256=runtime_binding_sha256,
                    scientific_schedule_sha256=schedule.schedule_sha256,
                    device=device,
                ),
                sample_key=sample_key,
                rendered_views=views,
                adapter=adapters[unit.model_id],
                output_dir=context.run_root / "stage" / "scientific-mve" / "calibration" / unit.model_id / f"scan{unit.scene_id:03d}" / "L3",
                split_fingerprint_sha256=split_assignment.fingerprint_sha256,
                inventory_sha256=split_assignment.inventory_sha256,
                resume=resume,
            )

    scientific: dict[tuple[str, int, str], Attempt05ScientificRuntimeBinding] = {}
    gt_cache: dict[int, tuple[tuple[Any, ...], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]] = {}
    for unit in schedule.units:
        key = (unit.scene_id, unit.state_id)
        row = rows_by_state.get(key)
        state = states_by_key.get(key)
        if row is None or state is None:
            raise Attempt05OrchestrationError("V4_ATTEMPT05_SCIENTIFIC_BINDING_MISSING")
        if unit.state_identity_sha256 != state.state_identity_sha256:
            raise Attempt05OrchestrationError("V4_ATTEMPT05_UNIT_STATE_MISMATCH")
        views = _rendered_views(row, state)
        signature = _scene_gt_signature(row, state)
        cached = gt_cache.get(unit.scene_id)
        if cached is None:
            gt = _gt_payload(row, state)
            gt_cache[unit.scene_id] = (signature, gt)
        else:
            cached_signature, gt = cached
            if cached_signature != signature:
                raise Attempt05OrchestrationError("V4_ATTEMPT05_SCENE_GT_BINDING_DRIFT")
        sample_key = SampleKey("dtu", "test", f"scan{unit.scene_id:03d}", "views-0001", unit.state_id, "0", "0")
        output_dir = (
            context.run_root
            / "stage"
            / "SCIENTIFIC_MVE"
            / "bundles"
            / unit.model_id
            / f"scan{unit.scene_id:03d}"
            / unit.state_id
        )
        scientific[(unit.model_id, unit.scene_id, unit.state_id)] = Attempt05ScientificRuntimeBinding(
            manifest=_manifest(
                model_id=unit.model_id,
                sample_key=sample_key,
                views=views,
                model_specs=model_specs,
                context=context,
                attempt05_tooling_commit=attempt05_tooling_commit,
                attempt05_tooling_tree=attempt05_tooling_tree,
                runtime_binding_sha256=runtime_binding_sha256,
                scientific_schedule_sha256=schedule.schedule_sha256,
                device=device,
            ),
            sample_key=sample_key,
            rendered_views=views,
            adapter=adapters[unit.model_id],
            output_dir=output_dir,
            gt_points=gt[0],
            gt_camera_c2w=gt[1],
            observability_mask=gt[2],
            gt_dtu_camera_c2w=gt[3],
            observability_bb=gt[4],
            observability_res=gt[5],
            resume=resume,
            legacy_output_dir=canonical_record_path(context.run_root, unit).parent,
            run_root=context.run_root,
        )
    if len(calibration) != 40 or len(scientific) != 400:
        raise Attempt05OrchestrationError("V4_ATTEMPT05_BINDING_COUNT_MISMATCH")
    return Attempt05RuntimeBindingSet(
        calibration_bindings=calibration,
        scientific_bindings=scientific,
        adapter_provider=adapter_provider,
    )

def _calibration_units_by_model(
    calibration_schedule: Sequence[Mapping[str, object]],
) -> dict[str, tuple[Attempt05CalibrationUnit, ...]]:
    rows = _validate_calibration_schedule(calibration_schedule)
    by_model: dict[str, list[Attempt05CalibrationUnit]] = {
        model: [] for model in SCIENTIFIC_MODELS
    }
    for index, row in enumerate(rows):
        by_model[str(row["model_id"])].append(
            Attempt05CalibrationUnit(
                model_id=str(row["model_id"]),
                scene_id=int(row["scene_id"]),
                state_id="L3",
                calibration_unit_sha256=str(row["calibration_unit_sha256"]),
                sequence_index=index,
            )
        )
    return {model: tuple(units) for model, units in by_model.items()}


def _scientific_units_by_model(
    schedule: ScientificSchedule,
) -> dict[str, tuple[ScientificExecutionUnit, ...]]:
    validated = validate_scientific_schedule(schedule)
    if len(validated.units) != 400:
        raise Attempt05OrchestrationError("V4_SCHEDULE_NOT_EXACT_400")
    by_model: dict[str, list[ScientificExecutionUnit]] = {
        model: [] for model in SCIENTIFIC_MODELS
    }
    for unit in validated.units:
        if unit.model_id not in by_model:
            raise Attempt05OrchestrationError("V4_ATTEMPT05_MODEL_SCOPE_MISMATCH")
        by_model[unit.model_id].append(unit)
    if any(len(units) != 200 for units in by_model.values()):
        raise Attempt05OrchestrationError("V4_ATTEMPT05_MODEL_UNIT_COUNT_MISMATCH")
    return {model: tuple(units) for model, units in by_model.items()}


def _q90_rows(calibrations: Mapping[str, NativeWarningCalibration]) -> list[dict[str, object]]:
    return [
        {
            "model_id": model,
            "state_id": "L3",
            "scene_count": 20,
            "quantile_probability": 0.90,
            "quantile_method": "linear",
            "alarm_threshold": calibrations[model].alarm_threshold,
            "calibration_identifier": calibrations[model].calibration_identifier,
        }
        for model in SCIENTIFIC_MODELS
    ]


def _default_fitter(
    samples: Sequence[CalibrationWarningSample],
    split_assignment: Any,
) -> NativeWarningCalibration:
    return fit_native_warning_calibration(samples, split_assignment=split_assignment)




def _tree_logical_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _tree_allocated_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        stat = path.stat()
        return int(getattr(stat, "st_blocks", 0) * 512) if getattr(stat, "st_blocks", 0) else stat.st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            stat = child.stat()
            total += int(getattr(stat, "st_blocks", 0) * 512) if getattr(stat, "st_blocks", 0) else stat.st_size
    return total


def _prediction_runtime(prediction: PredictionArtifact | None) -> tuple[float, float]:
    if prediction is None:
        return 0.0, 0.0
    return float(prediction.runtime_seconds), float(prediction.peak_memory_mb)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_matches_unit(path: Path, unit: ScientificExecutionUnit) -> bool:
    if not path.is_file():
        return False
    try:
        record = read_task_audit_record(path)
    except Task3ContractError as exc:
        raise Attempt05OrchestrationError(
            "V4_ATTEMPT05_LEGACY_TASK_RECORD_INVALID"
        ) from exc
    return (
        record.model_id == unit.model_id
        and record.scene_id == unit.scene_id
        and record.state_id == unit.state_id
        and record.execution_unit_sha256 == unit.execution_unit_sha256
        and record.state_identity_sha256 == unit.state_identity_sha256
        and record.pair_identity_sha256 == unit.pair_identity_sha256
    )


def _publish_canonical_record(
    *,
    source_record: Any,
    unit: ScientificExecutionUnit,
    run_root: Path,
) -> tuple[Path, bool, int, int]:
    path = canonical_record_path(run_root, unit)
    existed = path.exists()
    try:
        write_task_audit_record(path, source_record)
        admitted = admit_existing_task_record(path, unit, root=run_root)
    except (Task3ContractError, V4ExecutionError) as exc:
        raise Attempt05OrchestrationError(
            "V4_ATTEMPT05_CANONICAL_RECORD_PROJECTION_INVALID"
        ) from exc
    if admitted.record_sha256 != source_record.record_sha256:
        raise Attempt05OrchestrationError(
            "V4_ATTEMPT05_CANONICAL_RECORD_PROJECTION_INVALID"
        )
    stat = path.stat()
    allocated = (
        int(getattr(stat, "st_blocks", 0) * 512)
        if getattr(stat, "st_blocks", 0)
        else stat.st_size
    )
    return path, not existed, stat.st_size, allocated


def make_attempt05_runtime_executors(
    *,
    calibration_bindings: Mapping[tuple[str, int], Attempt05CalibrationRuntimeBinding],
    scientific_bindings: Mapping[tuple[str, int, str], Attempt05ScientificRuntimeBinding],
) -> tuple[CalibrationExecutor, ScientificExecutor]:
    """Create production executors that call the Attempt-05 runtime bridge.

    The wrappers are deliberately thin: they do not call the legacy runner, do
    not retry, do not delete partials, and convert runtime artifacts into the
    orchestration-level accounting dataclasses.
    """

    def calibration_executor(job: Attempt05CalibrationUnit) -> Attempt05CalibrationResult:
        key = (job.model_id, job.scene_id)
        binding = calibration_bindings.get(key)
        if binding is None:
            raise Attempt05OrchestrationError("V4_ATTEMPT05_CALIBRATION_BINDING_MISSING")
        before = time.perf_counter()
        result = execute_attempt05_calibration_l3(
            manifest=binding.manifest,
            sample_key=binding.sample_key,
            model_id=job.model_id,
            scene_id=job.scene_id,
            rendered_views=binding.rendered_views,
            adapter=binding.adapter,
            output_dir=binding.output_dir,
            resume=binding.resume,
        )
        wall = time.perf_counter() - before
        sample = CalibrationWarningSample(
            model_id=job.model_id,
            scene_id=job.scene_id,
            state_id="L3",
            warning_score=float(result.warning_score),
            split_fingerprint_sha256=binding.split_fingerprint_sha256,
            inventory_sha256=binding.inventory_sha256,
        )
        gpu_seconds, peak_mb = _prediction_runtime(result.prediction)
        resumed = result.prediction is None
        return Attempt05CalibrationResult(
            sample=sample,
            gpu_inference_seconds=gpu_seconds,
            wall_runtime_seconds=0.0 if resumed else wall,
            logical_bytes=0 if resumed else _tree_logical_bytes(binding.output_dir),
            allocated_bytes=0 if resumed else _tree_allocated_bytes(binding.output_dir),
            peak_memory_mb=peak_mb,
            artifact_sha256=result.evidence_sha256,
            resumed=resumed,
        )

    def scientific_executor(unit: ScientificExecutionUnit, calibration: NativeWarningCalibration) -> Attempt05ScientificResult:
        key = (unit.model_id, unit.scene_id, unit.state_id)
        binding = scientific_bindings.get(key)
        if binding is None:
            raise Attempt05OrchestrationError("V4_ATTEMPT05_SCIENTIFIC_BINDING_MISSING")
        before = time.perf_counter()
        execution_output_dir = binding.output_dir
        if (
            binding.resume
            and not execution_output_dir.exists()
            and binding.legacy_output_dir is not None
            and _record_matches_unit(
                binding.legacy_output_dir / "task_audit_record.json", unit
            )
        ):
            execution_output_dir = binding.legacy_output_dir
        result = execute_attempt05_unit(
            unit=unit,
            manifest=binding.manifest,
            sample_key=binding.sample_key,
            rendered_views=binding.rendered_views,
            adapter=binding.adapter,
            calibration=calibration,
            output_dir=execution_output_dir,
            gt_points=binding.gt_points,
            gt_camera_c2w=binding.gt_camera_c2w,
            observability_mask=binding.observability_mask,
            gt_dtu_camera_c2w=binding.gt_dtu_camera_c2w,
            observability_bb=binding.observability_bb,
            observability_res=binding.observability_res,
            resume=binding.resume,
        )
        wall = time.perf_counter() - before
        gpu_seconds, peak_mb = _prediction_runtime(result.prediction)
        resumed = result.prediction is None
        canonical_path, projection_created, projection_logical, projection_allocated = (
            _publish_canonical_record(
                source_record=result.record,
                unit=unit,
                run_root=binding.run_root or binding.output_dir.parents[5],
            )
        )
        return Attempt05ScientificResult(
            record_path=canonical_path,
            status=result.status,
            task_record_sha256=result.record.record_sha256,
            gpu_inference_seconds=gpu_seconds,
            wall_runtime_seconds=0.0 if resumed else wall,
            logical_bytes=0 if resumed else _tree_logical_bytes(execution_output_dir),
            allocated_bytes=0 if resumed else _tree_allocated_bytes(execution_output_dir),
            peak_memory_mb=peak_mb,
            resumed=resumed,
            projection_logical_bytes=projection_logical,
            projection_allocated_bytes=projection_allocated,
            projection_created=projection_created,
        )

    return calibration_executor, scientific_executor


def _assert_pre_dispatch_budget(totals: _ResourceTotals) -> None:
    decision = evaluate_attempt05_resource_gate(
        gpu_inference_seconds=totals.gpu_inference_seconds,
        wall_runtime_seconds=totals.wall_runtime_seconds,
        new_logical_bytes=totals.logical_bytes,
        new_allocated_bytes=totals.allocated_bytes,
    )
    if decision.status != "PASS":
        raise Attempt05OrchestrationError(decision.reason_code)

def run_attempt05_pipeline(
    *,
    authorization_path: Path,
    scientific_schedule: ScientificSchedule,
    model_independent_states: Sequence[object],
    split_assignment: Any,
    calibration_schedule: Sequence[Mapping[str, object]],
    attempt05_tooling_commit: str,
    attempt05_tooling_tree: str,
    ledger_path: Path,
    calibration_executor: CalibrationExecutor,
    scientific_executor: ScientificExecutor,
    finalizer: Finalizer = finalize_attempt05_scientific_bundle,
    calibration_fitter: CalibrationFitter = _default_fitter,
    resume: bool = False,
    input_storage: Mapping[str, object] | None = None,
) -> Attempt05PipelineResult:
    """Run calibration, freeze Q90, run scientific units, then finalize."""

    try:
        validate_v4_split_assignment(split_assignment)
    except Exception as exc:
        raise Attempt05OrchestrationError("V4_ATTEMPT05_SPLIT_INVALID") from exc
    calibration_by_model = _calibration_units_by_model(calibration_schedule)
    scientific_by_model = _scientific_units_by_model(scientific_schedule)
    start = create_attempt05_start_receipt(
        authorization_path=authorization_path,
        schedule=scientific_schedule,
        model_independent_states=model_independent_states,
        split_assignment=split_assignment,
        calibration_schedule=calibration_schedule,
        attempt05_tooling_commit=attempt05_tooling_commit,
        attempt05_tooling_tree=attempt05_tooling_tree,
        input_storage=input_storage,
        resume=resume,
    )
    ledger_resume = rehydrate_attempt05_ledger_totals(ledger_path) if resume else None
    if ledger_resume and ledger_resume.finalized:
        raise Attempt05OrchestrationError("V4_ATTEMPT05_ALREADY_FINALIZED")
    start_storage = start.get("budgeted_input_storage")
    if not isinstance(start_storage, Mapping):
        raise Attempt05OrchestrationError(
            "V4_ATTEMPT05_INPUT_STORAGE_BINDING_INVALID"
        )
    baseline_logical = int(start_storage.get("logical_bytes", 0))
    baseline_allocated = int(start_storage.get("allocated_bytes", 0))
    if not ledger_resume or not ledger_resume.run_started:
        append_attempt05_ledger_event(
            ledger_path=ledger_path,
            event_type="MVE_RUN_STARTED",
            payload={
                "attempt_id": "attempt-05",
                "start_receipt_sha256": str(start["start_receipt_sha256"]),
                "scientific_result": "NO_SCIENTIFIC_RESULT",
                "inventory_sha256": str(split_assignment.inventory_sha256),
                "retry_count": 0,
                "gpu_inference_seconds_total": 0.0,
                "wall_runtime_seconds_total": 0.0,
                "logical_bytes_total": baseline_logical,
                "allocated_bytes_total": baseline_allocated,
                "peak_memory_mb_total": 0.0,
            },
        )

    totals = _ResourceTotals(
        gpu_inference_seconds=ledger_resume.gpu_inference_seconds if ledger_resume else 0.0,
        wall_runtime_seconds=ledger_resume.wall_runtime_seconds if ledger_resume else 0.0,
        logical_bytes=(
            ledger_resume.logical_bytes if ledger_resume else baseline_logical
        ),
        allocated_bytes=(
            ledger_resume.allocated_bytes if ledger_resume else baseline_allocated
        ),
        peak_memory_mb=ledger_resume.peak_memory_mb if ledger_resume else 0.0,
    )
    calibration_samples: dict[str, list[CalibrationWarningSample]] = {
        model: [] for model in SCIENTIFIC_MODELS
    }
    calibration_units_completed = ledger_resume.calibration_units_completed if ledger_resume else 0
    for model in SCIENTIFIC_MODELS:
        for job in calibration_by_model[model]:
            _assert_pre_dispatch_budget(totals)
            result = calibration_executor(job)
            if result.sample.model_id != model or result.sample.scene_id != job.scene_id:
                raise Attempt05OrchestrationError(
                    "V4_ATTEMPT05_CALIBRATION_SAMPLE_MISMATCH"
                )
            if not result.resumed:
                totals.add(
                    gpu_inference_seconds=result.gpu_inference_seconds,
                    wall_runtime_seconds=result.wall_runtime_seconds,
                    logical_bytes=result.logical_bytes,
                    allocated_bytes=result.allocated_bytes,
                    peak_memory_mb=result.peak_memory_mb,
                )
                calibration_units_completed += 1
                append_attempt05_ledger_event(
                    ledger_path=ledger_path,
                    event_type="CALIBRATION_UNIT_COMPLETE",
                    payload={
                    "model_id": model,
                    "scene_id": job.scene_id,
                    "state_id": "L3",
                    "artifact_sha256": result.artifact_sha256,
                    "inventory_sha256": result.sample.inventory_sha256,
                    "retry_count": 0,
                    "gpu_inference_seconds_total": totals.gpu_inference_seconds,
                    "wall_runtime_seconds_total": totals.wall_runtime_seconds,
                    "logical_bytes_total": totals.logical_bytes,
                    "allocated_bytes_total": totals.allocated_bytes,
                    "peak_memory_mb_total": totals.peak_memory_mb,
                    "storage_bytes_total": totals.logical_bytes,
                    },
                )
                _assert_pre_dispatch_budget(totals)
            elif (
                ledger_resume is None
                or (model, job.scene_id, "L3")
                not in ledger_resume.calibration_unit_keys
            ):
                raise Attempt05OrchestrationError(
                    "V4_ATTEMPT05_ARTIFACT_LEDGER_MISMATCH"
                )
            calibration_samples[model].append(result.sample)

    calibrations: dict[str, NativeWarningCalibration] = {}
    for model in SCIENTIFIC_MODELS:
        calibrations[model] = calibration_fitter(
            tuple(calibration_samples[model]),
            split_assignment,
        )
    create_attempt05_q90_freeze_artifact(
        authorization_path=authorization_path,
        calibration_schedule_sha256=str(start["calibration_schedule_sha256"]),
        q90_calibrations=_q90_rows(calibrations),
        attempt05_tooling_commit=attempt05_tooling_commit,
        attempt05_tooling_tree=attempt05_tooling_tree,
    )

    record_paths: list[Path] = []
    scientific_units_completed = ledger_resume.scientific_units_completed if ledger_resume else 0
    invalid_scientific_units = ledger_resume.invalid_units if ledger_resume else 0
    for model in SCIENTIFIC_MODELS:
        for unit in scientific_by_model[model]:
            _assert_pre_dispatch_budget(totals)
            result = scientific_executor(unit, calibrations[model])
            projection_key = (unit.model_id, unit.scene_id, unit.state_id)
            projection_ledgered = (
                ledger_resume is not None
                and projection_key in ledger_resume.projection_unit_keys
            )
            if projection_ledgered and result.projection_created:
                raise Attempt05OrchestrationError(
                    "V4_ATTEMPT05_CANONICAL_PROJECTION_LEDGER_MISMATCH"
                )
            if not projection_ledgered:
                if (
                    result.projection_logical_bytes <= 0
                    or result.projection_allocated_bytes <= 0
                ):
                    raise Attempt05OrchestrationError(
                        "V4_ATTEMPT05_CANONICAL_PROJECTION_ACCOUNTING_INVALID"
                    )
                totals.add(
                    gpu_inference_seconds=0.0,
                    wall_runtime_seconds=0.0,
                    logical_bytes=result.projection_logical_bytes,
                    allocated_bytes=result.projection_allocated_bytes,
                    peak_memory_mb=0.0,
                )
                append_attempt05_ledger_event(
                    ledger_path=ledger_path,
                    event_type="CANONICAL_RECORD_PROJECTION",
                    payload={
                        "model_id": unit.model_id,
                        "scene_id": unit.scene_id,
                        "state_id": unit.state_id,
                        "task_record_sha256": result.task_record_sha256,
                        "projection_path": str(result.record_path),
                        "retry_count": 0,
                        "gpu_inference_seconds_total": totals.gpu_inference_seconds,
                        "wall_runtime_seconds_total": totals.wall_runtime_seconds,
                        "logical_bytes_total": totals.logical_bytes,
                        "allocated_bytes_total": totals.allocated_bytes,
                        "peak_memory_mb_total": totals.peak_memory_mb,
                    },
                )
                _assert_pre_dispatch_budget(totals)
            if not result.resumed:
                totals.add(
                    gpu_inference_seconds=result.gpu_inference_seconds,
                    wall_runtime_seconds=result.wall_runtime_seconds,
                    logical_bytes=result.logical_bytes,
                    allocated_bytes=result.allocated_bytes,
                    peak_memory_mb=result.peak_memory_mb,
                )
            elif (
                ledger_resume is None
                or (unit.model_id, unit.scene_id, unit.state_id)
                not in ledger_resume.scientific_unit_keys
            ):
                raise Attempt05OrchestrationError(
                    "V4_ATTEMPT05_ARTIFACT_LEDGER_MISMATCH"
                )
            if result.status in {"INVALID_FAILURE_RECORDED", "RESUMED_INVALID_FAILURE_RECORDED"}:
                if not result.resumed:
                    invalid_scientific_units += 1
            elif result.status not in {"VALID_COMPLETE", "RESUMED_VALID_COMPLETE"}:
                raise Attempt05OrchestrationError("V4_ATTEMPT05_UNIT_STATUS_INVALID")
            record_paths.append(result.record_path)
            if not result.resumed:
                scientific_units_completed += 1
                append_attempt05_ledger_event(
                    ledger_path=ledger_path,
                    event_type="SCIENTIFIC_UNIT_COMPLETE",
                    payload={
                    "model_id": unit.model_id,
                    "scene_id": unit.scene_id,
                    "state_id": unit.state_id,
                    "task_record_sha256": result.task_record_sha256,
                    "status": result.status,
                    "inventory_sha256": str(split_assignment.inventory_sha256),
                    "retry_count": 0,
                    "gpu_inference_seconds_total": totals.gpu_inference_seconds,
                    "wall_runtime_seconds_total": totals.wall_runtime_seconds,
                    "logical_bytes_total": totals.logical_bytes,
                    "allocated_bytes_total": totals.allocated_bytes,
                    "peak_memory_mb_total": totals.peak_memory_mb,
                    "storage_bytes_total": totals.logical_bytes,
                    },
                )
                _assert_pre_dispatch_budget(totals)
    if len(record_paths) != 400:
        raise Attempt05OrchestrationError("V4_RECORD_COUNT_NOT_400")
    finalizer_result = finalizer(
        authorization_path=authorization_path,
        record_paths=tuple(record_paths),
        scientific_schedule=scientific_schedule,
        model_independent_states=model_independent_states,
        split_assignment=split_assignment,
        native_warning_calibrations=tuple(calibrations[model] for model in SCIENTIFIC_MODELS),
    )
    status = str(finalizer_result.get("status", "V4_MVE_COMPLETED"))
    append_attempt05_ledger_event(
        ledger_path=ledger_path,
        event_type="MVE_FINALIZED",
        payload={
            "status": status,
            "scientific_result": "SCIENTIFIC_RESULT_AVAILABLE",
            "record_count": 400,
            "invalid_scientific_units": invalid_scientific_units,
            "inventory_sha256": str(split_assignment.inventory_sha256),
            "retry_count": 0,
            "gpu_inference_seconds_total": totals.gpu_inference_seconds,
            "wall_runtime_seconds_total": totals.wall_runtime_seconds,
            "logical_bytes_total": totals.logical_bytes,
            "allocated_bytes_total": totals.allocated_bytes,
            "peak_memory_mb_total": totals.peak_memory_mb,
        },
    )
    return Attempt05PipelineResult(
        status=status,
        calibration_units_completed=calibration_units_completed,
        scientific_units_completed=scientific_units_completed,
        invalid_scientific_units=invalid_scientific_units,
        gpu_inference_seconds=totals.gpu_inference_seconds,
        wall_runtime_seconds=totals.wall_runtime_seconds,
        logical_bytes=totals.logical_bytes,
        allocated_bytes=totals.allocated_bytes,
        peak_memory_mb=totals.peak_memory_mb,
        finalizer_result=finalizer_result,
    )











