"""Opt-in Gate 2 GPU kill/resume smoke harness.

This file is deliberately test-only.  It executes twelve non-scientific L3
model calls, injects three real SIGKILL boundaries, resumes from the durable
transaction state, and writes an auditable qualification bundle.  It is never
collected as an ordinary pytest test and never advances to a Pilot.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from georeliab_mve import toml_compat as tomllib  # noqa: E402
from georeliab_mve.adapters import RenderedView  # noqa: E402
from georeliab_mve.contracts import (  # noqa: E402
    PredictionArtifact,
    RunManifest,
    RunMode,
    SampleKey,
    ScientificProvenance,
    ScientificValidity,
)
from georeliab_mve.v4_attempt05_recovery import (  # noqa: E402
    AUTHORIZED_GPU_UUID,
    AUTHORIZED_PHYSICAL_GPU_INDEX,
    NO_SCIENTIFIC_RESULT,
    JournaledLedger,
    RecoverySmokeManifest,
    UnitTransactionStore,
    _load_latest_receipt,
    atomic_write_bytes,
    atomic_write_json,
    build_recovery_smoke_manifest,
    evaluate_recovery_smoke,
    read_hash_chain,
    sha256_file,
)
from georeliab_mve.v4_counterfactuals import (  # noqa: E402
    ModelIndependentState,
    parse_scientific_schedule,
)
from georeliab_mve.v4_attempt05_inputs import (  # noqa: E402
    validate_attempt05_input_closure,
)
from tests.local_gate2_prepare import (  # noqa: E402
    validate_formal_gate2_authorization,
    validate_formal_home_closure_payload,
    validate_local_closure,
)


SCHEMA_VERSION = "georeliab-v4-gate2-gpu-smoke-harness-1.0"
LOCAL_SCHEMA_VERSION = "georeliab-v4-local-gate2-gpu-smoke-harness-1.0"
LOCAL_VALIDATION_CLASS = "LOCAL_GATE2_DEVELOPMENT_VALIDATION"
CANONICAL_COMMIT = "6de08f7a89f88de1de79cef09de74b4e909f27b0"
CANONICAL_TREE = "111078e2f3031061ea8aec8cf7cbf9bea77ebbb7"
FROZEN_PARENT_COMMIT = "a7875354a7bd900a0b418092da0e6a69ffba6745"
EXCLUDED_PRODUCTION_PATHS = (
    # The harness never imports the scientific Attempt-05 orchestrator.  Its
    # pre-existing worktree repair is explicitly excluded and recorded rather
    # than silently treating an unrelated dirty file as qualified Gate 2 code.
    "georeliab_mve/v4_attempt05_orchestrator.py",
)
INTERRUPTION_PHASES = frozenset(
    {
        "canonical_promotion_before_projection",
        "projection_before_completion",
        "inference_before_prepared_receipt",
    }
)
FORBIDDEN_SCIENTIFIC_TOKENS = (
    b'"scientific_validity":"SCIENTIFIC"',
    b'"event_type":"MVE_FINALIZED"',
    b'"auroc"',
    b'"auc"',
    b'"boundary_lag"',
    b'"claim"',
)


class Gate2HarnessError(RuntimeError):
    """Fail-closed Gate 2 harness error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Gate2HarnessError(f"GATE2_JSON_UNREADABLE:{path}") from exc


def _write_json_no_clobber(path: Path, value: object) -> None:
    if path.exists():
        raise Gate2HarnessError(f"GATE2_EVIDENCE_COLLISION:{path}")
    atomic_write_json(path, value)


def _safe_key(unit_key: str) -> str:
    prefix = unit_key.replace("|", "-").replace("/", "-").lower()
    return f"{prefix}-{_sha_bytes(unit_key.encode('utf-8'))[:12]}"


def _refuse_home_output(path: Path) -> None:
    resolved = path.resolve()
    if resolved == Path("/home") or Path("/home") in resolved.parents:
        raise Gate2HarnessError("GATE2_OUTPUT_UNDER_HOME_FORBIDDEN")


def _require_home_output(path: Path) -> None:
    resolved = path.expanduser().resolve()
    home = Path.home().resolve()
    if resolved == home or home not in resolved.parents:
        raise Gate2HarnessError("LOCAL_GATE2_OUTPUT_OUTSIDE_HOME_FORBIDDEN")


def _require_formal_home_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    formal_root = (Path.home() / "georeliab-gate2-formal").resolve()
    if resolved == formal_root or formal_root not in resolved.parents:
        raise Gate2HarnessError("FORMAL_GATE2_HOME_ROOT_OUTPUT_INVALID")
    return resolved


def _require_formal_home_output(path: Path) -> None:
    """Require a fresh output descendant of the dedicated formal home root."""

    resolved = _require_formal_home_path(path)
    if resolved.exists():
        raise Gate2HarnessError(f"FORMAL_GATE2_OUTPUT_NOT_FRESH_EXISTS:{resolved}")


def _fsync_append(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


class EventLog:
    """Write matching machine-readable and human-readable append-only logs."""

    def __init__(self, output_root: Path) -> None:
        self.events_path = output_root / "events.jsonl"
        self.text_path = output_root / "gate2.log"

    def write(self, event: str, **fields: object) -> None:
        row = {
            "schema_version": SCHEMA_VERSION,
            "recorded_at": _utc_now(),
            "event": event,
            "scientific_result": NO_SCIENTIFIC_RESULT,
            **fields,
        }
        _fsync_append(self.events_path, _canonical_bytes(row))
        summary = " ".join(
            f"{key}={json.dumps(value, sort_keys=True)}"
            for key, value in sorted(fields.items())
        )
        _fsync_append(
            self.text_path,
            f"[{row['recorded_at']}] {event} {summary}\n".encode("utf-8"),
        )
        print(f"[{row['recorded_at']}] {event} {summary}", flush=True)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _source_manifest(repo: Path) -> dict[str, object]:
    commit = _git(repo, "rev-parse", CANONICAL_COMMIT)
    tree = _git(repo, "rev-parse", f"{CANONICAL_COMMIT}^{{tree}}")
    if commit != CANONICAL_COMMIT or tree != CANONICAL_TREE:
        raise Gate2HarnessError("GATE2_CANONICAL_IDENTITY_MISMATCH")
    canonical_paths = tuple(
        path
        for path in _git(
            repo,
            "ls-tree",
            "-r",
            "--name-only",
            CANONICAL_COMMIT,
            "--",
            "georeliab_mve",
        ).splitlines()
        if path.endswith(".py") and path not in EXCLUDED_PRODUCTION_PATHS
    )
    if not canonical_paths:
        raise Gate2HarnessError("GATE2_CANONICAL_SOURCE_LIST_EMPTY")
    files: list[dict[str, str]] = []
    for relative in canonical_paths:
        worktree_bytes = (repo / relative).read_bytes()
        canonical_bytes = subprocess.run(
            ["git", "show", f"{CANONICAL_COMMIT}:{relative}"],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
        if worktree_bytes != canonical_bytes:
            raise Gate2HarnessError(f"GATE2_PRODUCTION_SOURCE_DRIFT:{relative}")
        files.append(
            {
                "path": relative,
                "sha256": _sha_bytes(worktree_bytes),
                "canonical_sha256": _sha_bytes(canonical_bytes),
            }
        )
    harness_path = Path(__file__).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "canonical_commit": CANONICAL_COMMIT,
        "canonical_tree": CANONICAL_TREE,
        "frozen_parent_commit": FROZEN_PARENT_COMMIT,
        "production_source_zero_drift": True,
        "excluded_production_paths": list(EXCLUDED_PRODUCTION_PATHS),
        "production_files": files,
        "test_harness_path": str(harness_path),
        "test_harness_sha256": sha256_file(harness_path),
        "worktree_head": _git(repo, "rev-parse", "HEAD"),
        "worktree_status": _git(repo, "status", "--porcelain"),
        "scientific_result": NO_SCIENTIFIC_RESULT,
    }


def _overlay_payload(path: Path) -> dict[str, Any]:
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise Gate2HarnessError("GATE2_OVERLAY_UNREADABLE") from exc
    if not isinstance(payload, dict):
        raise Gate2HarnessError("GATE2_OVERLAY_INVALID")
    runtime = payload.get("runtime")
    resources = payload.get("resources")
    if not isinstance(runtime, Mapping) or not isinstance(resources, Mapping):
        raise Gate2HarnessError("GATE2_OVERLAY_INVALID")
    return payload


def _model_spec(
    overlay: Mapping[str, Any], model: str, overlay_sha: str
) -> dict[str, str | None]:
    runtime = overlay["runtime"]
    resources = overlay["resources"]
    if model == "VGGT":
        return {
            "checkpoint": str(resources["vggt_checkpoint_sha256"]),
            "source_commit": str(resources["vggt_source_commit"]),
            "python": str(runtime["vggt_python"]),
            "torch": str(runtime["vggt_torch"]),
            "environment_lock": overlay_sha,
            "dust3r_commit": None,
            "croco_commit": None,
        }
    if model == "MASt3R":
        return {
            "checkpoint": str(resources["mast3r_checkpoint_sha256"]),
            "source_commit": str(resources["mast3r_source_commit"]),
            "python": str(runtime["mast3r_python"]),
            "torch": str(runtime["mast3r_torch"]),
            "environment_lock": overlay_sha,
            "dust3r_commit": str(resources["dust3r_source_commit"]),
            "croco_commit": str(resources["croco_source_commit"]),
        }
    raise Gate2HarnessError(f"GATE2_MODEL_INVALID:{model}")


def _runtime_rows(path: Path) -> dict[tuple[int, str], Mapping[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise Gate2HarnessError("GATE2_RUNTIME_BINDINGS_INVALID")
    rows = payload.get("bindings")
    if (
        payload.get("schema_version")
        != "georeliab-v4-attempt-05-runtime-state-bindings-1.0"
        or payload.get("attempt_id") != "attempt-05"
        or not isinstance(rows, list)
        or payload.get("binding_count") != len(rows)
    ):
        raise Gate2HarnessError("GATE2_RUNTIME_BINDINGS_INVALID")
    result: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise Gate2HarnessError("GATE2_RUNTIME_BINDINGS_INVALID")
        key = (int(row["scene_id"]), str(row["state_id"]))
        if key in result:
            raise Gate2HarnessError("GATE2_RUNTIME_BINDINGS_DUPLICATE")
        result[key] = row
    return result


def _state_rows(path: Path) -> dict[tuple[int, str], ModelIndependentState]:
    payload = _read_json(path)
    if isinstance(payload, Mapping):
        raw = payload.get("states", payload.get("model_independent_states"))
    else:
        raw = payload
    if not isinstance(raw, list):
        raise Gate2HarnessError("GATE2_STATE_INVENTORY_INVALID")
    states = tuple(ModelIndependentState.from_dict(row) for row in raw)
    if len(states) != 200:
        raise Gate2HarnessError("GATE2_STATE_INVENTORY_NOT_200")
    return {(state.scene_id, state.state_id): state for state in states}


def _closure_input_rows(
    payload: Mapping[str, object], *, boundary: str
) -> tuple[
    dict[tuple[int, str], Mapping[str, Any]],
    dict[tuple[int, str], Any],
]:
    """Load the same six frozen L3 bit bindings behind an explicit boundary."""

    raw_bindings = payload.get("bindings")
    if not isinstance(raw_bindings, list):
        raise Gate2HarnessError(f"{boundary}_GATE2_BINDINGS_INVALID")
    bindings: dict[tuple[int, str], Mapping[str, Any]] = {}
    states: dict[tuple[int, str], Any] = {}
    for row in raw_bindings:
        if not isinstance(row, Mapping):
            raise Gate2HarnessError(f"{boundary}_GATE2_BINDINGS_INVALID")
        key = (int(row.get("scene_id", -1)), str(row.get("state_id", "")))
        raw_views = row.get("views")
        if key in bindings or key[1] != "L3" or not isinstance(raw_views, list):
            raise Gate2HarnessError(f"{boundary}_GATE2_BINDINGS_INVALID")
        hashes = tuple(
            SimpleNamespace(view_id=int(view["view_id"]), sha256=str(view["sha256"]))
            for view in raw_views
            if isinstance(view, Mapping)
        )
        if len(hashes) != 8:
            raise Gate2HarnessError(f"{boundary}_GATE2_BINDINGS_INVALID")
        bindings[key] = row
        states[key] = SimpleNamespace(input_sha256_by_view=hashes)
    if len(bindings) != 6:
        raise Gate2HarnessError(f"{boundary}_GATE2_BINDING_COUNT_INVALID")
    return bindings, states


def _local_input_rows(
    closure_path: Path,
) -> tuple[
    dict[tuple[int, str], Mapping[str, Any]],
    dict[tuple[int, str], Any],
]:
    """Load only the six L3 bindings from the test-only local closure."""

    return _closure_input_rows(
        validate_local_closure(closure_path), boundary="LOCAL"
    )


def _formal_home_input_rows(
    closure_path: Path,
) -> tuple[
    dict[tuple[int, str], Mapping[str, Any]],
    dict[tuple[int, str], Any],
]:
    """Load revalidated bits without reading any prior prediction output."""

    payload = _read_json(closure_path)
    if not isinstance(payload, Mapping):
        raise Gate2HarnessError("FORMAL_GATE2_CLOSURE_INVALID")
    validated = validate_formal_home_closure_payload(
        payload, expected_formal_root=closure_path.resolve().parents[1]
    )
    return _closure_input_rows(validated, boundary="FORMAL")


def _bound_views(
    row: Mapping[str, Any], state: ModelIndependentState
) -> tuple[RenderedView, ...]:
    raw_views = row.get("views")
    ordered = tuple(int(value) for value in row.get("ordered_view_ids", ()))
    if not isinstance(raw_views, list) or len(raw_views) != 8 or len(ordered) != 8:
        raise Gate2HarnessError("GATE2_VIEW_BINDING_INVALID")
    expected = {item.view_id: item.sha256 for item in state.input_sha256_by_view}
    views: list[RenderedView] = []
    for raw in raw_views:
        if not isinstance(raw, Mapping):
            raise Gate2HarnessError("GATE2_VIEW_BINDING_INVALID")
        path = Path(str(raw.get("path", "")))
        digest = str(raw.get("sha256", ""))
        view_id = int(raw["view_id"])
        if not path.is_file() or sha256_file(path) != digest:
            raise Gate2HarnessError(f"GATE2_VIEW_DIGEST_MISMATCH:{path}")
        if expected.get(view_id) != digest:
            raise Gate2HarnessError("GATE2_STATE_VIEW_DIGEST_MISMATCH")
        views.append(
            RenderedView(
                view_id=view_id,
                png_path=path,
                png_sha256=digest,
                width=int(raw["width"]) if raw.get("width") is not None else None,
                height=int(raw["height"]) if raw.get("height") is not None else None,
                source_sha256=(
                    str(raw["source_sha256"])
                    if raw.get("source_sha256") is not None
                    else None
                ),
            )
        )
    if tuple(view.view_id for view in views) != ordered:
        raise Gate2HarnessError("GATE2_VIEW_ORDER_MISMATCH")
    return tuple(views)


def _rgb_digest(views: Sequence[RenderedView]) -> str:
    digest = hashlib.sha256()
    for view in views:
        digest.update(str(view.view_id).encode("ascii"))
        digest.update(b"\0")
        digest.update(view.png_sha256.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _smoke_run_manifest(
    *,
    model: str,
    scene_id: int,
    views: Sequence[RenderedView],
    model_spec: Mapping[str, str | None],
    runtime_binding_sha: str,
    schedule_identity_sha: str,
    local_development: bool = False,
) -> tuple[RunManifest, SampleKey]:
    sample_key = SampleKey(
        "dtu",
        "local-gate2" if local_development else "gate2",
        f"scan{scene_id:03d}",
        "views-0001",
        "L3",
        "0",
        "0",
    )
    provenance = ScientificProvenance(
        project_commit=CANONICAL_COMMIT,
        project_tree=CANONICAL_TREE,
        model_source_commit=str(model_spec["source_commit"]),
        environment_lock_sha256=str(model_spec["environment_lock"]),
        corruption_manifest_sha256=runtime_binding_sha,
        split_view_manifest_sha256=schedule_identity_sha,
        dust3r_source_commit=model_spec.get("dust3r_commit"),
        croco_source_commit=model_spec.get("croco_commit"),
    )
    return (
        RunManifest(
            run_id=(
                f"local-gate2-{model.lower()}-scan{scene_id:03d}-l3"
                if local_development
                else f"gate2-{model.lower()}-scan{scene_id:03d}-l3"
            ),
            mode=RunMode.SMOKE,
            scientific_validity=ScientificValidity.NON_SCIENTIFIC_SMOKE,
            model=model,
            checkpoint_hash=str(model_spec["checkpoint"]),
            dataset="dtu",
            split="local-gate2" if local_development else "gate2",
            seed=0,
            intervention_version="none",
            corruption_version=(
                "georeliab-v4-local-gate2-development-l3"
                if local_development
                else "georeliab-v4-gate2-l3"
            ),
            environment={
                "device": "cuda:0",
                "physical_gpu_index": str(AUTHORIZED_PHYSICAL_GPU_INDEX),
                "gpu_uuid": AUTHORIZED_GPU_UUID,
                "python": str(model_spec["python"]),
                "torch": str(model_spec["torch"]),
            },
            rgb_digest=_rgb_digest(views),
            prompt_digest="fixed-empty-prompt",
            decoder_digest="fixed-native-decoder",
            provenance=provenance,
        ),
        sample_key,
    )


_MODEL_WORKER_CODE = r"""
import json
from pathlib import Path
import sys

project, request_name, result_name = sys.argv[1:]
sys.path.insert(0, project)
from georeliab_mve.adapters import RenderedView
from georeliab_mve.contracts import RunManifest, SampleKey
from georeliab_mve.runner import (
    RunnerContext,
    _verify_current_worker_runtime,
    default_adapter_factory,
)

request = json.loads(Path(request_name).read_text(encoding="utf-8"))
context = RunnerContext(
    root=Path(request["runtime_root"]),
    output_root=Path(request["output_root"]),
    config_path=Path(request["overlay_config"]),
    device="cuda:0",
)
model = request["model"]
_verify_current_worker_runtime(context, model.lower())
adapter = default_adapter_factory(model, context)
manifest = RunManifest.from_dict(request["manifest"])
sample_key = SampleKey.parse(request["sample_key"])
views = tuple(
    RenderedView(
        view_id=int(row["view_id"]),
        png_path=Path(row["png_path"]),
        png_sha256=row["png_sha256"],
        width=row.get("width"),
        height=row.get("height"),
        source_sha256=row.get("source_sha256"),
    )
    for row in request["views"]
)
prediction = adapter.predict_sample(manifest, sample_key, views)
Path(result_name).write_text(
    json.dumps(prediction.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
"""


_LOCAL_MODEL_WORKER_CODE = r"""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

project, request_name, result_name = sys.argv[1:]
sys.path.insert(0, project)
from georeliab_mve.adapters import RenderedView
from georeliab_mve.contracts import RunManifest, SampleKey
from georeliab_mve.runner import _verify_current_worker_runtime, default_adapter_factory

request = json.loads(Path(request_name).read_text(encoding="utf-8"))
# RunnerContext intentionally forbids every /home write for formal execution.
# This test-only namespace preserves that production guard while allowing a
# clearly labelled home-owned development smoke to call the real adapters.
context = SimpleNamespace(
    root=Path(request["runtime_root"]),
    output_root=Path(request["output_root"]),
    config_path=Path(request["overlay_config"]),
    device="cuda:0",
)
model = request["model"]
_verify_current_worker_runtime(context, model.lower())
adapter = default_adapter_factory(model, context)
manifest = RunManifest.from_dict(request["manifest"])
sample_key = SampleKey.parse(request["sample_key"])
views = tuple(
    RenderedView(
        view_id=int(row["view_id"]),
        png_path=Path(row["png_path"]),
        png_sha256=row["png_sha256"],
        width=row.get("width"),
        height=row.get("height"),
        source_sha256=row.get("source_sha256"),
    )
    for row in request["views"]
)
prediction = adapter.predict_sample(manifest, sample_key, views)
Path(result_name).write_text(
    json.dumps(prediction.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
"""


def _model_python(overlay: Mapping[str, Any], model: str) -> Path:
    runtime = overlay["runtime"]
    env_root = Path(str(runtime["vggt_env" if model == "VGGT" else "mast3r_env"]))
    candidates = (env_root / "bin" / "python", env_root / "Scripts" / "python.exe")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise Gate2HarnessError(f"GATE2_MODEL_PYTHON_MISSING:{model}:{env_root}")


def _run_model_inference(
    *,
    repo: Path,
    overlay_path: Path,
    overlay: Mapping[str, Any],
    model: str,
    manifest: RunManifest,
    sample_key: SampleKey,
    views: Sequence[RenderedView],
    inference_root: Path,
    timeout_seconds: float,
    local_development: bool = False,
    home_owned_formal: bool = False,
) -> PredictionArtifact:
    inference_root.mkdir(parents=True, exist_ok=False)
    request_path = inference_root / "request.json"
    result_path = inference_root / "prediction.json"
    request = {
        "runtime_root": str(overlay["runtime"]["root"]),
        "output_root": str(inference_root / "payload"),
        "overlay_config": str(overlay_path),
        "model": model,
        "manifest": manifest.to_dict(),
        "sample_key": str(sample_key),
        "views": [
            {
                "view_id": view.view_id,
                "png_path": str(view.png_path),
                "png_sha256": view.png_sha256,
                "width": view.width,
                "height": view.height,
                "source_sha256": view.source_sha256,
            }
            for view in views
        ],
    }
    _write_json_no_clobber(request_path, request)
    env = dict(os.environ)
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0",
            "GEORELIAB_PHYSICAL_GPU_DEVICE": "cuda:0",
            "GEORELIAB_PHYSICAL_GPU_UUID": AUTHORIZED_GPU_UUID,
        }
    )
    with (
        (inference_root / "model.stdout.log").open("xb") as stdout,
        (inference_root / "model.stderr.log").open("xb") as stderr,
    ):
        result = subprocess.run(
            [
                str(_model_python(overlay, model)),
                "-I",
                "-B",
                "-c",
                (
                    _LOCAL_MODEL_WORKER_CODE
                    if local_development or home_owned_formal
                    else _MODEL_WORKER_CODE
                ),
                str(repo),
                str(request_path),
                str(result_path),
            ],
            cwd=repo,
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=False,
            timeout=timeout_seconds,
        )
    if result.returncode != 0 or not result_path.is_file():
        raise Gate2HarnessError(
            f"GATE2_MODEL_INFERENCE_FAILED:{model}:exit={result.returncode}"
        )
    payload = _read_json(result_path)
    if not isinstance(payload, Mapping):
        raise Gate2HarnessError("GATE2_PREDICTION_INVALID")
    prediction = PredictionArtifact.from_dict(payload)
    if prediction.invalid_prediction:
        raise Gate2HarnessError(f"GATE2_INVALID_PREDICTION:{model}")
    return prediction


def _uri_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or (parsed.netloc not in ("", "localhost")):
        raise Gate2HarnessError("GATE2_PREDICTION_URI_INVALID")
    return Path(unquote(parsed.path))


def _transaction_files(
    manifest: RunManifest, prediction: PredictionArtifact
) -> dict[str, bytes]:
    files = {
        "geometry.npz": _uri_path(prediction.geometry_prediction_uri).read_bytes(),
        "confidence.npz": _uri_path(prediction.native_confidence_uri).read_bytes(),
        "valid-mask.npz": _uri_path(prediction.valid_mask_uri).read_bytes(),
        "run-manifest.json": _canonical_bytes(manifest.to_dict()),
        "prediction-artifact.json": _canonical_bytes(prediction.to_dict()),
    }
    for label, payload in files.items():
        if not payload:
            raise Gate2HarnessError(f"GATE2_EMPTY_TRANSACTION_FILE:{label}")
    return files


def _idempotency_key(schedule_identity: str, unit_key: str) -> str:
    return _sha_bytes(
        f"v4-recovery-smoke-01\0{schedule_identity}\0{unit_key}".encode("utf-8")
    )


def _pause_for_supervisor(boundary_path: Path, payload: Mapping[str, object]) -> None:
    _write_json_no_clobber(boundary_path, payload)
    while True:
        signal.pause()


def _inference_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len([line for line in path.read_bytes().splitlines() if line])


def _observation(
    *,
    smoke: RecoverySmokeManifest,
    unit_key: str,
    canonical_dir: Path,
    ledger_path: Path,
    inference_count_path: Path,
    inference_receipt_root: Path,
) -> dict[str, object]:
    receipt = _load_latest_receipt(canonical_dir)
    rows = read_hash_chain(ledger_path, allow_empty=False)
    projections = 0
    completions = 0
    for row in rows:
        payload = row.get("payload")
        if not isinstance(payload, Mapping) or payload.get("unit_key") != unit_key:
            continue
        if row.get("event_type") == "CANONICAL_RECORD_PROJECTION":
            projections += 1
        elif row.get("event_type") == "SMOKE_UNIT_COMPLETE":
            completions += 1
    canonical_present = receipt.state == "LEDGER_COMMITTED" and all(
        (canonical_dir / name).is_file() and sha256_file(canonical_dir / name) == digest
        for name, digest in receipt.files.items()
    )
    inference_receipts = []
    for path in sorted(inference_receipt_root.glob("attempt-*.json")):
        payload = _read_json(path)
        if not isinstance(payload, Mapping) or payload.get("unit_key") != unit_key:
            raise Gate2HarnessError("GATE2_INFERENCE_RECEIPT_INVALID")
        inference_receipts.append(payload)
    gpu_inference_seconds = sum(
        float(row.get("runtime_seconds", 0.0)) for row in inference_receipts
    )
    peak_memory_mb = max(
        (float(row.get("peak_memory_mb", 0.0)) for row in inference_receipts),
        default=0.0,
    )
    return {
        "unit_key": unit_key,
        "inference_start_count": _inference_count(inference_count_path),
        "completion_count": completions,
        "projection_count": projections,
        "overwrite_count": 0,
        "interruption_phase": smoke.interruption_plan.get(unit_key),
        "gpu_uuid": smoke.gpu_uuid,
        "physical_gpu_index": smoke.physical_gpu_index,
        "canonical_present": canonical_present,
        "ledger_committed": receipt.state == "LEDGER_COMMITTED",
        "gpu_inference_seconds": gpu_inference_seconds,
        "peak_memory_mb": peak_memory_mb,
        "scientific_marker": NO_SCIENTIFIC_RESULT,
    }


def _run_unit_worker(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[1]
    output_root = args.output_root
    smoke = RecoverySmokeManifest.from_mapping(_read_json(args.smoke_manifest))
    unit_key = args.unit_key
    if unit_key not in smoke.unit_keys:
        raise Gate2HarnessError("GATE2_UNIT_NOT_IN_MANIFEST")
    model, scene_text, state_id = unit_key.split("|")
    scene_id = int(scene_text)
    if state_id != "L3":
        raise Gate2HarnessError("GATE2_NON_L3_UNIT_FORBIDDEN")
    overlay = _overlay_payload(args.overlay_config)
    overlay_sha = sha256_file(args.overlay_config)
    home_owned_formal = bool(getattr(args, "home_owned_formal", False))
    if args.local_development:
        runtime_binding_path = args.input_closure_dir / "local-gate2-input-closure.json"
        runtime_rows, states = _local_input_rows(runtime_binding_path)
    elif home_owned_formal:
        runtime_binding_path = args.input_closure_dir / "formal-gate2-input-closure.json"
        runtime_rows, states = _formal_home_input_rows(runtime_binding_path)
    else:
        runtime_binding_path = args.input_closure_dir / "v4-runtime-state-bindings.json"
        states_path = args.input_closure_dir / "v4-model-independent-states.json"
        runtime_rows = _runtime_rows(runtime_binding_path)
        states = _state_rows(states_path)
    row = runtime_rows.get((scene_id, "L3"))
    state = states.get((scene_id, "L3"))
    if row is None or state is None:
        raise Gate2HarnessError("GATE2_SELECTED_INPUT_MISSING")
    views = _bound_views(row, state)
    run_manifest, sample_key = _smoke_run_manifest(
        model=model,
        scene_id=scene_id,
        views=views,
        model_spec=_model_spec(overlay, model, overlay_sha),
        runtime_binding_sha=sha256_file(runtime_binding_path),
        schedule_identity_sha=smoke.schedule_identity_sha256,
        local_development=args.local_development,
    )
    safe = _safe_key(unit_key)
    canonical_dir = output_root / "canonical" / safe
    ledger_path = output_root / "ledger.jsonl"
    count_path = output_root / "control" / f"{safe}.inference-starts.jsonl"
    inference_receipt_root = output_root / "control" / "inference-receipts" / safe
    boundary_path = output_root / "control" / f"{safe}.boundary.json"
    idempotency = _idempotency_key(smoke.schedule_identity_sha256, unit_key)
    projection_payload = {
        "unit_key": unit_key,
        "canonical_path": str(canonical_dir),
        "gpu_uuid": smoke.gpu_uuid,
        "physical_gpu_index": smoke.physical_gpu_index,
        "scientific_result": NO_SCIENTIFIC_RESULT,
    }
    completion_payload = dict(projection_payload)
    ledger_events = (
        {"event_type": "CANONICAL_RECORD_PROJECTION", "payload": projection_payload},
        {"event_type": "SMOKE_UNIT_COMPLETE", "payload": completion_payload},
    )
    ledger = JournaledLedger(ledger_path)
    store = UnitTransactionStore(
        output_root / "transactions",
        attempt_id=smoke.attempt_id,
        ledger=ledger,
        schedule_identity_sha256=smoke.schedule_identity_sha256,
    )
    if canonical_dir.exists():
        receipt = _load_latest_receipt(canonical_dir)
    else:
        partial = canonical_dir.with_name(canonical_dir.name + ".partial")
        if partial.exists():
            raise Gate2HarnessError("GATE2_UNEXPECTED_PARTIAL_TREE")
        attempt_index = _inference_count(count_path) + 1
        _fsync_append(
            count_path,
            _canonical_bytes(
                {
                    "unit_key": unit_key,
                    "attempt_index": attempt_index,
                    "started_at": _utc_now(),
                    "scientific_result": NO_SCIENTIFIC_RESULT,
                }
            ),
        )
        inference_root = (
            output_root / "inference" / safe / f"attempt-{attempt_index:02d}"
        )
        prediction = _run_model_inference(
            repo=repo,
            overlay_path=args.overlay_config,
            overlay=overlay,
            model=model,
            manifest=run_manifest,
            sample_key=sample_key,
            views=views,
            inference_root=inference_root,
            timeout_seconds=args.model_timeout_seconds,
            local_development=args.local_development,
            home_owned_formal=home_owned_formal,
        )
        _write_json_no_clobber(
            inference_receipt_root / f"attempt-{attempt_index:02d}.json",
            {
                "schema_version": SCHEMA_VERSION,
                "unit_key": unit_key,
                "attempt_index": attempt_index,
                "runtime_seconds": prediction.runtime_seconds,
                "peak_memory_mb": prediction.peak_memory_mb,
                "payload_digests": dict(prediction.payload_digests),
                "invalid_prediction": prediction.invalid_prediction,
                "completed_at": _utc_now(),
                "scientific_result": NO_SCIENTIFIC_RESULT,
            },
        )
        if args.fault_phase == "inference_before_prepared_receipt":
            _pause_for_supervisor(
                boundary_path,
                {
                    "unit_key": unit_key,
                    "phase": args.fault_phase,
                    "pid": os.getpid(),
                    "scientific_result": NO_SCIENTIFIC_RESULT,
                },
            )
        receipt = store.prepare_unit(
            idempotency_key=idempotency,
            unit_key=unit_key,
            stage="GATE2_GPU_SMOKE",
            canonical_dir=canonical_dir,
            files=_transaction_files(run_manifest, prediction),
            ledger_events=ledger_events,
        )
        receipt = store.promote_unit(receipt)
        if args.fault_phase == "canonical_promotion_before_projection":
            _pause_for_supervisor(
                boundary_path,
                {
                    "unit_key": unit_key,
                    "phase": args.fault_phase,
                    "pid": os.getpid(),
                    "scientific_result": NO_SCIENTIFIC_RESULT,
                },
            )
        if args.fault_phase == "projection_before_completion":
            durable_projection = dict(projection_payload)
            durable_projection["_transaction_idempotency_key"] = f"{idempotency}:0"
            ledger.append("CANONICAL_RECORD_PROJECTION", durable_projection)
            _pause_for_supervisor(
                boundary_path,
                {
                    "unit_key": unit_key,
                    "phase": args.fault_phase,
                    "pid": os.getpid(),
                    "scientific_result": NO_SCIENTIFIC_RESULT,
                },
            )
    committed = store.commit_unit(receipt)
    if committed.state != "LEDGER_COMMITTED":
        raise Gate2HarnessError("GATE2_UNIT_NOT_LEDGER_COMMITTED")
    observation = _observation(
        smoke=smoke,
        unit_key=unit_key,
        canonical_dir=canonical_dir,
        ledger_path=ledger_path,
        inference_count_path=count_path,
        inference_receipt_root=inference_receipt_root,
    )
    _write_json_no_clobber(output_root / "observations" / f"{safe}.json", observation)
    return 0


def _gpu_inventory() -> dict[str, object]:
    inventory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    rows = []
    for line in inventory:
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            raise Gate2HarnessError("GATE2_GPU_INVENTORY_INVALID")
        rows.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "pci_bus_id": parts[2],
                "name": parts[3],
            }
        )
    expected = [
        row
        for row in rows
        if row["index"] == AUTHORIZED_PHYSICAL_GPU_INDEX
        and row["uuid"] == AUTHORIZED_GPU_UUID
    ]
    if len(expected) != 1:
        raise Gate2HarnessError("GATE2_AUTHORIZED_GPU_IDENTITY_MISMATCH")
    processes_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    processes = []
    if processes_result.returncode == 0:
        for line in processes_result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",", 2)]
            if len(parts) == 3:
                processes.append(
                    {"gpu_uuid": parts[0], "pid": parts[1], "process_name": parts[2]}
                )
    gpu1_owners = [
        row for row in processes if row.get("gpu_uuid") != AUTHORIZED_GPU_UUID
    ]
    return {
        "gpus": rows,
        "compute_processes": processes,
        "gpu1_owners": gpu1_owners,
        "authorized_gpu_uuid": AUTHORIZED_GPU_UUID,
        "authorized_physical_gpu_index": AUTHORIZED_PHYSICAL_GPU_INDEX,
        "scientific_result": NO_SCIENTIFIC_RESULT,
    }


def _non_target_gpu_processes_forbidden(*, local_development: bool) -> bool:
    """Require machine-wide exclusivity only for the formal Gate 2 mode."""

    return not local_development


def _wait_for_boundary_or_exit(
    process: subprocess.Popen[bytes], boundary_path: Path, timeout_seconds: float
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if boundary_path.is_file():
            return
        if process.poll() is not None:
            raise Gate2HarnessError(
                f"GATE2_WORKER_EXITED_BEFORE_BOUNDARY:{process.returncode}"
            )
        time.sleep(0.2)
    raise Gate2HarnessError("GATE2_BOUNDARY_TIMEOUT")


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=30)


def _worker_command(
    args: argparse.Namespace,
    smoke_manifest_path: Path,
    unit_key: str,
    fault_phase: str | None,
) -> list[str]:
    if bool(getattr(args, "home_owned_formal", False)):
        worker_command = "_formal-home-unit-worker"
    elif args.local_development:
        worker_command = "_local-unit-worker"
    else:
        worker_command = "_unit-worker"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        worker_command,
        "--output-root",
        str(args.output_root),
        "--input-closure-dir",
        str(args.input_closure_dir),
        "--overlay-config",
        str(args.overlay_config),
        "--smoke-manifest",
        str(smoke_manifest_path),
        "--unit-key",
        unit_key,
        "--model-timeout-seconds",
        str(args.model_timeout_seconds),
    ]
    if fault_phase is not None:
        command.extend(["--fault-phase", fault_phase])
    return command


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _recorded_gpu_seconds(output_root: Path) -> float:
    receipt_root = output_root / "control" / "inference-receipts"
    total = 0.0
    for path in receipt_root.glob("*/attempt-*.json"):
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            raise Gate2HarnessError("GATE2_INFERENCE_RECEIPT_INVALID")
        total += float(payload.get("runtime_seconds", 0.0))
    return total


def _load_observations(
    output_root: Path, smoke: RecoverySmokeManifest
) -> list[dict[str, object]]:
    rows = []
    for unit_key in smoke.unit_keys:
        path = output_root / "observations" / f"{_safe_key(unit_key)}.json"
        payload = _read_json(path)
        if not isinstance(payload, dict):
            raise Gate2HarnessError("GATE2_OBSERVATION_INVALID")
        rows.append(payload)
    return rows


def _scan_for_scientific_markers(output_root: Path) -> list[str]:
    violations: list[str] = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".log"}:
            continue
        payload = path.read_bytes().lower()
        for token in FORBIDDEN_SCIENTIFIC_TOKENS:
            if token.lower() in payload:
                violations.append(
                    f"{path.relative_to(output_root)}:{token.decode('ascii')}"
                )
    return violations


def _write_manifest(output_root: Path) -> None:
    rows = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            rows.append(
                f"{sha256_file(path)}  {path.relative_to(output_root).as_posix()}"
            )
    target = output_root / "MANIFEST.sha256"
    if target.exists():
        raise Gate2HarnessError("GATE2_MANIFEST_COLLISION")
    atomic_write_bytes(target, ("\n".join(rows) + "\n").encode("utf-8"))


def _formal_home_preflight(args: argparse.Namespace) -> dict[str, object]:
    """Validate closure and authorization before output creation or GPU probes."""

    input_dir = _require_formal_home_path(args.input_closure_dir)
    authorization_path = _require_formal_home_path(args.authorization_manifest)
    if input_dir.name != "manifests" or authorization_path.parent != input_dir:
        raise Gate2HarnessError("FORMAL_GATE2_MANIFEST_ROOT_IDENTITY_INVALID")
    if authorization_path.name != "formal-gate2-authorization.json":
        raise Gate2HarnessError("FORMAL_GATE2_AUTHORIZATION_PATH_INVALID")
    closure_path = input_dir / "formal-gate2-input-closure.json"
    closure_payload = _read_json(closure_path)
    if not isinstance(closure_payload, Mapping):
        raise Gate2HarnessError("FORMAL_GATE2_CLOSURE_INVALID")
    formal_root = input_dir.parent
    closure = validate_formal_home_closure_payload(
        closure_payload, expected_formal_root=formal_root
    )
    if (
        closure.get("production_source_commit") != CANONICAL_COMMIT
        or closure.get("production_source_tree") != CANONICAL_TREE
    ):
        raise Gate2HarnessError("FORMAL_GATE2_PRODUCTION_SOURCE_IDENTITY_MISMATCH")
    overlay_path = args.overlay_config.expanduser().resolve()
    if (
        Path(str(closure.get("overlay_path", ""))).expanduser().resolve()
        != overlay_path
        or closure.get("overlay_sha256") != sha256_file(overlay_path)
    ):
        raise Gate2HarnessError("FORMAL_GATE2_OVERLAY_DIGEST_MISMATCH")
    authorization_payload = _read_json(authorization_path)
    if not isinstance(authorization_payload, Mapping):
        raise Gate2HarnessError("FORMAL_GATE2_AUTHORIZATION_INVALID")
    authorization = validate_formal_gate2_authorization(
        authorization_payload,
        expected_closure_sha256=sha256_file(closure_path),
        expected_output_root=args.output_root,
    )
    if (
        authorization.get("production_source_commit")
        != closure.get("production_source_commit")
        or authorization.get("test_only_source_commit")
        != closure.get("test_only_source_commit")
        or authorization.get("schedule_identity_sha256")
        != closure.get("schedule_identity_sha256")
    ):
        raise Gate2HarnessError("FORMAL_GATE2_AUTHORIZATION_PROVENANCE_MISMATCH")
    for field in ("max_gpu_seconds", "max_wall_seconds", "max_storage_bytes"):
        if getattr(args, field) != authorization.get(field):
            raise Gate2HarnessError(f"FORMAL_GATE2_AUTHORIZATION_BUDGET_MISMATCH:{field}")
    return {
        "closure_path": closure_path,
        "closure": closure,
        "authorization_path": authorization_path,
        "authorization": authorization,
    }


def _run_supervisor(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[1]
    home_owned_formal = bool(getattr(args, "home_owned_formal", False))
    formal_preflight: dict[str, object] | None = None
    if home_owned_formal:
        _require_formal_home_output(args.output_root)
        formal_preflight = _formal_home_preflight(args)
    elif args.local_development:
        _require_home_output(args.output_root)
        _require_home_output(args.input_closure_dir)
    else:
        _refuse_home_output(args.output_root)
    if args.output_root.exists():
        raise Gate2HarnessError("GATE2_OUTPUT_ROOT_EXISTS")
    args.output_root.mkdir(parents=True, exist_ok=False)
    log = EventLog(args.output_root)
    started = time.monotonic()
    authorization_note = (
        str(formal_preflight["authorization"]["authorization_note"])
        if formal_preflight is not None
        else args.authorization_note
    )
    try:
        log.write(
            "LOCAL_GATE2_START" if args.local_development else "GATE2_START",
            authorization_note=authorization_note,
            validation_class=(
                LOCAL_VALIDATION_CLASS if args.local_development else "FORMAL_GATE2"
            ),
        )
        source = _source_manifest(repo)
        _write_json_no_clobber(args.output_root / "source-manifest.json", source)
        gpu_before = _gpu_inventory()
        _write_json_no_clobber(
            args.output_root / "gpu-inventory-before.json", gpu_before
        )
        if gpu_before["gpu1_owners"] and _non_target_gpu_processes_forbidden(
            local_development=args.local_development
        ):
            raise Gate2HarnessError("GATE2_GPU1_PROJECT_PROCESS_PRESENT")
        if args.local_development and gpu_before["gpu1_owners"]:
            log.write(
                "LOCAL_NON_TARGET_GPU_PROCESSES_OBSERVED",
                process_count=len(gpu_before["gpu1_owners"]),
                processes=gpu_before["gpu1_owners"],
                target_gpu_uuid=AUTHORIZED_GPU_UUID,
            )
        if args.local_development:
            input_closure_path = (
                args.input_closure_dir / "local-gate2-input-closure.json"
            )
            input_closure = validate_local_closure(input_closure_path)
            embedded_smoke = input_closure.get("smoke_manifest")
            if not isinstance(embedded_smoke, Mapping):
                raise Gate2HarnessError("LOCAL_GATE2_SMOKE_MANIFEST_MISSING")
            smoke = RecoverySmokeManifest.from_mapping(embedded_smoke)
            schedule_path = None
        elif home_owned_formal:
            assert formal_preflight is not None
            input_closure_path = formal_preflight["closure_path"]
            input_closure = formal_preflight["closure"]
            if not isinstance(input_closure_path, Path) or not isinstance(
                input_closure, Mapping
            ):
                raise Gate2HarnessError("FORMAL_GATE2_PREFLIGHT_STATE_INVALID")
            embedded_smoke = input_closure.get("smoke_manifest")
            if not isinstance(embedded_smoke, Mapping):
                raise Gate2HarnessError("FORMAL_GATE2_SMOKE_MANIFEST_MISSING")
            smoke = RecoverySmokeManifest.from_mapping(embedded_smoke)
            schedule_path = None
        else:
            input_closure_path = (
                args.input_closure_dir / "v4-attempt05-input-closure.json"
            )
            input_closure = validate_attempt05_input_closure(input_closure_path)
            schedule_path = args.input_closure_dir / "v4-scientific-schedule-400.json"
            schedule = parse_scientific_schedule(
                schedule_path.read_text(encoding="utf-8")
            )
            smoke = build_recovery_smoke_manifest(
                schedule_identity_sha256=schedule.schedule_sha256,
                support_scene_ids=schedule.scene_ids,
            )
        smoke_path = args.output_root / "smoke-manifest.json"
        _write_json_no_clobber(smoke_path, smoke.to_dict())
        input_manifest = {
            "schema_version": (
                LOCAL_SCHEMA_VERSION if args.local_development else SCHEMA_VERSION
            ),
            "validation_class": (
                LOCAL_VALIDATION_CLASS if args.local_development else "FORMAL_GATE2"
            ),
            "formal_gate2_equivalent": not args.local_development,
            "input_closure_dir": str(args.input_closure_dir),
            "input_closure_path": str(input_closure_path),
            "input_closure_sha256": sha256_file(input_closure_path),
            "input_closure_status": input_closure["status"],
            "schedule_identity_sha256": smoke.schedule_identity_sha256,
            "runtime_binding_path": str(input_closure_path)
            if args.local_development or home_owned_formal
            else str(args.input_closure_dir / "v4-runtime-state-bindings.json"),
            "runtime_binding_sha256": sha256_file(input_closure_path)
            if args.local_development or home_owned_formal
            else sha256_file(args.input_closure_dir / "v4-runtime-state-bindings.json"),
            "overlay_config": str(args.overlay_config),
            "overlay_sha256": sha256_file(args.overlay_config),
            "attempt05_predictions_read": False,
            "scientific_result": NO_SCIENTIFIC_RESULT,
        }
        if formal_preflight is not None:
            authorization_path = formal_preflight["authorization_path"]
            if not isinstance(authorization_path, Path):
                raise Gate2HarnessError("FORMAL_GATE2_PREFLIGHT_STATE_INVALID")
            input_manifest.update(
                {
                    "authorization_manifest": str(authorization_path),
                    "authorization_sha256": sha256_file(authorization_path),
                    "prediction_outputs_reused": False,
                }
            )
        if schedule_path is not None:
            input_manifest.update(
                {
                    "schedule_path": str(schedule_path),
                    "schedule_sha256": sha256_file(schedule_path),
                }
            )
        _write_json_no_clobber(args.output_root / "input-manifest.json", input_manifest)
        for ordinal, unit_key in enumerate(smoke.unit_keys, 1):
            elapsed_before_unit = time.monotonic() - started
            if elapsed_before_unit > args.max_wall_seconds:
                raise Gate2HarnessError("GATE2_WALL_BUDGET_EXCEEDED")
            if _tree_bytes(args.output_root) > args.max_storage_bytes:
                raise Gate2HarnessError("GATE2_STORAGE_BUDGET_EXCEEDED")
            if _recorded_gpu_seconds(args.output_root) > args.max_gpu_seconds:
                raise Gate2HarnessError("GATE2_GPU_BUDGET_EXCEEDED")
            remaining_wall = max(1.0, args.max_wall_seconds - elapsed_before_unit)
            unit_timeout = min(args.unit_timeout_seconds, remaining_wall)
            fault_phase = smoke.interruption_plan.get(unit_key)
            log.write(
                "UNIT_DISPATCH",
                ordinal=ordinal,
                unit_key=unit_key,
                fault_phase=fault_phase,
            )
            safe = _safe_key(unit_key)
            logs = args.output_root / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            with (
                (logs / f"{ordinal:02d}-{safe}-first.stdout.log").open("xb") as stdout,
                (logs / f"{ordinal:02d}-{safe}-first.stderr.log").open("xb") as stderr,
            ):
                process = subprocess.Popen(
                    _worker_command(args, smoke_path, unit_key, fault_phase),
                    cwd=repo,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                try:
                    if fault_phase is None:
                        returncode = process.wait(timeout=unit_timeout)
                        if returncode != 0:
                            raise Gate2HarnessError(
                                f"GATE2_UNIT_WORKER_FAILED:{unit_key}:exit={returncode}"
                            )
                    else:
                        boundary_path = (
                            args.output_root / "control" / f"{safe}.boundary.json"
                        )
                        _wait_for_boundary_or_exit(process, boundary_path, unit_timeout)
                        boundary = _read_json(boundary_path)
                        if (
                            not isinstance(boundary, Mapping)
                            or boundary.get("phase") != fault_phase
                        ):
                            raise Gate2HarnessError("GATE2_BOUNDARY_EVIDENCE_INVALID")
                        log.write(
                            "CONTROLLED_SIGKILL",
                            ordinal=ordinal,
                            unit_key=unit_key,
                            phase=fault_phase,
                            worker_pid=process.pid,
                        )
                        os.killpg(process.pid, signal.SIGKILL)
                        returncode = process.wait(timeout=30)
                        if returncode != -signal.SIGKILL:
                            raise Gate2HarnessError(
                                f"GATE2_SIGKILL_NOT_OBSERVED:{unit_key}:exit={returncode}"
                            )
                finally:
                    _kill_process_group(process)
                if fault_phase is not None:
                    with (
                        (logs / f"{ordinal:02d}-{safe}-resume.stdout.log").open(
                            "xb"
                        ) as resume_stdout,
                        (logs / f"{ordinal:02d}-{safe}-resume.stderr.log").open(
                            "xb"
                        ) as resume_stderr,
                    ):
                        resumed = subprocess.Popen(
                            _worker_command(args, smoke_path, unit_key, None),
                            cwd=repo,
                            stdout=resume_stdout,
                            stderr=resume_stderr,
                            start_new_session=True,
                        )
                        try:
                            resumed_returncode = resumed.wait(
                                timeout=min(
                                    args.unit_timeout_seconds,
                                    max(
                                        1.0,
                                        args.max_wall_seconds
                                        - (time.monotonic() - started),
                                    ),
                                )
                            )
                            if resumed_returncode != 0:
                                raise Gate2HarnessError(
                                    f"GATE2_RESUME_FAILED:{unit_key}:exit={resumed_returncode}"
                                )
                        finally:
                            _kill_process_group(resumed)
            observation = _read_json(args.output_root / "observations" / f"{safe}.json")
            log.write(
                "UNIT_COMPLETE",
                ordinal=ordinal,
                unit_key=unit_key,
                inference_start_count=observation.get("inference_start_count"),
                projection_count=observation.get("projection_count"),
                completion_count=observation.get("completion_count"),
            )
        observations = _load_observations(args.output_root, smoke)
        result = evaluate_recovery_smoke(smoke, observations=observations)
        projection_violations = [
            str(row["unit_key"])
            for row in observations
            if row.get("projection_count") != 1
        ]
        marker_violations = _scan_for_scientific_markers(args.output_root)
        gpu_after = _gpu_inventory()
        _write_json_no_clobber(args.output_root / "gpu-inventory-after.json", gpu_after)
        gpu_runtime_seconds = sum(
            float(row.get("gpu_inference_seconds", 0.0)) for row in observations
        )
        elapsed = time.monotonic() - started
        storage_bytes = _tree_bytes(args.output_root)
        budget_violations = []
        if gpu_runtime_seconds > args.max_gpu_seconds:
            budget_violations.append("GATE2_GPU_BUDGET_EXCEEDED")
        if elapsed > args.max_wall_seconds:
            budget_violations.append("GATE2_WALL_BUDGET_EXCEEDED")
        if storage_bytes > args.max_storage_bytes:
            budget_violations.append("GATE2_STORAGE_BUDGET_EXCEEDED")
        if gpu_after["gpu1_owners"] and _non_target_gpu_processes_forbidden(
            local_development=args.local_development
        ):
            budget_violations.append("GATE2_GPU1_PROJECT_PROCESS_PRESENT")
        passed = (
            result["status"] == "V4_RECOVERY_RUNTIME_QUALIFIED"
            and not projection_violations
            and not marker_violations
            and not budget_violations
        )
        qualification = {
            **result,
            "schema_version": (
                LOCAL_SCHEMA_VERSION if args.local_development else SCHEMA_VERSION
            ),
            "status": (
                (
                    "LOCAL_GATE2_DEVELOPMENT_PASS"
                    if passed
                    else "LOCAL_GATE2_DEVELOPMENT_FAILED"
                )
                if args.local_development
                else (
                    "V4_GATE2_GPU_SMOKE_PASS" if passed else "V4_GATE2_GPU_SMOKE_FAILED"
                )
            ),
            "validation_class": (
                LOCAL_VALIDATION_CLASS if args.local_development else "FORMAL_GATE2"
            ),
            "formal_gate2_equivalent": not args.local_development,
            "recovery_runtime_status": result["status"],
            "projection_violations": projection_violations,
            "scientific_marker_violations": marker_violations,
            "budget_violations": budget_violations,
            "gpu_inference_seconds": gpu_runtime_seconds,
            "wall_seconds": elapsed,
            "storage_bytes": storage_bytes,
            "non_target_gpu_processes_before": gpu_before["gpu1_owners"],
            "non_target_gpu_processes_after": gpu_after["gpu1_owners"],
            "gate2_started": True,
            "pilot_started": False,
            "attempt06_started": False,
            "scientific_result": NO_SCIENTIFIC_RESULT,
        }
        _write_json_no_clobber(args.output_root / "qualification.json", qualification)
        log.write("GATE2_TERMINAL", status=qualification["status"])
        _write_manifest(args.output_root)
        return 0 if passed else 2
    except BaseException as exc:
        failure_status = (
            "LOCAL_GATE2_DEVELOPMENT_FAILED"
            if args.local_development
            else "V4_GATE2_GPU_SMOKE_FAILED"
        )
        failure = {
            "schema_version": (
                LOCAL_SCHEMA_VERSION if args.local_development else SCHEMA_VERSION
            ),
            "status": failure_status,
            "validation_class": (
                LOCAL_VALIDATION_CLASS if args.local_development else "FORMAL_GATE2"
            ),
            "formal_gate2_equivalent": not args.local_development,
            "reason_code": str(exc),
            "exception_type": type(exc).__name__,
            "gate2_started": True,
            "pilot_started": False,
            "attempt06_started": False,
            "scientific_result": NO_SCIENTIFIC_RESULT,
        }
        qualification_path = args.output_root / "qualification.json"
        if not qualification_path.exists():
            _write_json_no_clobber(qualification_path, failure)
        log.write(
            "GATE2_TERMINAL",
            status=failure_status,
            reason_code=str(exc),
        )
        if not (args.output_root / "MANIFEST.sha256").exists():
            _write_manifest(args.output_root)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--input-closure-dir", type=Path, required=True)
    run.add_argument("--overlay-config", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--authorization-note", required=True)
    run.add_argument("--model-timeout-seconds", type=float, default=7200.0)
    run.add_argument("--unit-timeout-seconds", type=float, default=7500.0)
    run.add_argument("--max-gpu-seconds", type=float, default=21600.0)
    run.add_argument("--max-wall-seconds", type=float, default=43200.0)
    run.add_argument("--max-storage-bytes", type=int, default=50 * 1024**3)
    run.set_defaults(local_development=False, home_owned_formal=False)

    local_run = subparsers.add_parser("local-run")
    local_run.add_argument("--input-closure-dir", type=Path, required=True)
    local_run.add_argument("--overlay-config", type=Path, required=True)
    local_run.add_argument("--output-root", type=Path, required=True)
    local_run.add_argument("--authorization-note", required=True)
    local_run.add_argument("--model-timeout-seconds", type=float, default=7200.0)
    local_run.add_argument("--unit-timeout-seconds", type=float, default=7500.0)
    local_run.add_argument("--max-gpu-seconds", type=float, default=21600.0)
    local_run.add_argument("--max-wall-seconds", type=float, default=43200.0)
    local_run.add_argument("--max-storage-bytes", type=int, default=50 * 1024**3)
    local_run.set_defaults(local_development=True, home_owned_formal=False)

    formal_home_run = subparsers.add_parser("formal-home-run")
    formal_home_run.add_argument("--input-closure-dir", type=Path, required=True)
    formal_home_run.add_argument("--overlay-config", type=Path, required=True)
    formal_home_run.add_argument("--output-root", type=Path, required=True)
    formal_home_run.add_argument(
        "--authorization-manifest", type=Path, required=True
    )
    formal_home_run.add_argument("--model-timeout-seconds", type=float, default=7200.0)
    formal_home_run.add_argument("--unit-timeout-seconds", type=float, default=7500.0)
    formal_home_run.add_argument("--max-gpu-seconds", type=float, default=21600.0)
    formal_home_run.add_argument("--max-wall-seconds", type=float, default=43200.0)
    formal_home_run.add_argument(
        "--max-storage-bytes", type=int, default=25 * 1024**3
    )
    formal_home_run.set_defaults(local_development=False, home_owned_formal=True)

    worker = subparsers.add_parser("_unit-worker")
    worker.add_argument("--input-closure-dir", type=Path, required=True)
    worker.add_argument("--overlay-config", type=Path, required=True)
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--smoke-manifest", type=Path, required=True)
    worker.add_argument("--unit-key", required=True)
    worker.add_argument("--fault-phase", choices=sorted(INTERRUPTION_PHASES))
    worker.add_argument("--model-timeout-seconds", type=float, default=7200.0)
    worker.set_defaults(local_development=False, home_owned_formal=False)

    local_worker = subparsers.add_parser("_local-unit-worker")
    local_worker.add_argument("--input-closure-dir", type=Path, required=True)
    local_worker.add_argument("--overlay-config", type=Path, required=True)
    local_worker.add_argument("--output-root", type=Path, required=True)
    local_worker.add_argument("--smoke-manifest", type=Path, required=True)
    local_worker.add_argument("--unit-key", required=True)
    local_worker.add_argument("--fault-phase", choices=sorted(INTERRUPTION_PHASES))
    local_worker.add_argument("--model-timeout-seconds", type=float, default=7200.0)
    local_worker.set_defaults(local_development=True, home_owned_formal=False)

    formal_home_worker = subparsers.add_parser("_formal-home-unit-worker")
    formal_home_worker.add_argument("--input-closure-dir", type=Path, required=True)
    formal_home_worker.add_argument("--overlay-config", type=Path, required=True)
    formal_home_worker.add_argument("--output-root", type=Path, required=True)
    formal_home_worker.add_argument("--smoke-manifest", type=Path, required=True)
    formal_home_worker.add_argument("--unit-key", required=True)
    formal_home_worker.add_argument(
        "--fault-phase", choices=sorted(INTERRUPTION_PHASES)
    )
    formal_home_worker.add_argument(
        "--model-timeout-seconds", type=float, default=7200.0
    )
    formal_home_worker.set_defaults(local_development=False, home_owned_formal=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {
        "_unit-worker",
        "_local-unit-worker",
        "_formal-home-unit-worker",
    }:
        return _run_unit_worker(args)
    return _run_supervisor(args)


if __name__ == "__main__":
    raise SystemExit(main())
