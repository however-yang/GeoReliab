"""Resumable GeoReliab real-MVE runner core.

This module owns the execution-governance layer: frozen schedules, sharding,
stage freezes, portable claims, atomic bundle commits, budget accounting, and
append-only ledgers.  It deliberately keeps heavy model imports behind
``default_adapter_factory`` so schedule and dry-run paths remain lightweight.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence
import argparse
import sys

import numpy as np

from . import toml_compat as tomllib
from .adapters import RenderedView
from .materialization import (
    FROZEN_TYPING_EXTENSIONS_DIST_INFO,
    FROZEN_TYPING_EXTENSIONS_SHA256,
    FROZEN_TYPING_EXTENSIONS_SITE,
    FROZEN_TYPING_EXTENSIONS_VERSION,
)
from .contracts import (
    AuditRecord,
    ContractError,
    PredictionArtifact,
    RunManifest,
    RunMode,
    SampleKey,
    ScientificProvenance,
    ScientificValidity,
    read_json_artifact,
    validate_artifact_bundle,
    write_json_artifact,
)
from .preparation import TEST_SCENES


MODELS = ("VGGT", "MASt3R")
CONDITIONS = (
    ("clean", 0),
    ("fog", 1),
    ("fog", 2),
    ("fog", 3),
    ("low-light-noise", 1),
    ("low-light-noise", 2),
    ("low-light-noise", 3),
    ("defocus", 1),
    ("defocus", 2),
    ("defocus", 3),
)
PREFLIGHT_CONDITIONS = (
    ("clean", 0),
    ("fog", 1),
    ("low-light-noise", 1),
    ("defocus", 1),
)
ZERO_UPDATE_SUBSETS = ((0, 4), (1, 5), (2, 6), (3, 7))
GPU_HOUR_LIMIT = 50.0
STORAGE_BYTE_LIMIT = 1_000_000_000_000
CLAIM_STALE_SECONDS = 6 * 60 * 60
LEDGER_LOCK_STALE_SECONDS = 60.0


class RunnerError(RuntimeError):
    """Raised when runner governance must fail closed."""


class ZeroUpdateTerminalFailure(RunnerError):
    """Raised after persisting an immutable terminal P5 invalid-output record."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__("P5_INVALID_SUBSET_PREDICTION")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    checkpoint_sha256: str
    source_commit: str
    python: str
    torch: str
    environment_lock_sha256: str
    typing_extensions_site: str
    typing_extensions_sha256: str
    typing_extensions_version: str
    config_sha256: str | None = None
    dust3r_source_commit: str | None = None
    croco_source_commit: str | None = None


@dataclass(frozen=True, slots=True)
class ScheduleItem:
    stage: str
    model: str
    sample_key: SampleKey
    scene_id: int
    condition: str
    severity: int
    rendered_views: tuple[RenderedView, ...]
    identity: str
    subset: tuple[int, int] | None = None
    parent_identity: str | None = None


@dataclass(frozen=True, slots=True)
class RunnerContext:
    root: Path
    output_root: Path
    config_path: Path | None
    device: str

    def __post_init__(self) -> None:
        _refuse_home_write(self.root)
        _refuse_home_write(self.output_root)


@dataclass(frozen=True, slots=True)
class ClaimResult:
    path: Path
    acquired: bool
    stale_reclaimed: bool = False
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RunResult:
    item_identity: str
    state: str
    bundle_dir: Path
    invalid_prediction: bool
    reason_code: str
    runtime_seconds: float
    peak_memory_mb: float


AdapterFactory = Callable[[str, RunnerContext], Any]
AuditFactory = Callable[[RunManifest, PredictionArtifact, Path], AuditRecord]
_UPSTREAM_CACHE: dict[tuple[Any, ...], Any] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _canonical_ordered_png_bundle_digest(
    views: Sequence[RenderedView], *, expected_count: int
) -> str:
    if len(views) != expected_count:
        raise RunnerError(
            f"rendered sample must bind exactly {expected_count} PNG views"
        )
    digest = hashlib.sha256()
    seen: set[int] = set()
    for view in views:
        view_id = int(view.view_id)
        if view_id in seen:
            raise RunnerError("rendered sample has duplicate view_id")
        seen.add(view_id)
        digest.update(str(view_id).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(view.png_sha256).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _refuse_home_write(path: Path) -> None:
    text = path.as_posix()
    if text == "/home" or text.startswith("/home/"):
        raise RunnerError("runner refuses writes below /home/smli or any /home path")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        handle.write(raw)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    partial.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(value, encoding="utf-8")
    with partial.open("r+b") as handle:
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    partial.replace(path)


def append_ledger_row(output_root: Path, stage: str, row: Mapping[str, Any]) -> None:
    _append_ledger(output_root, stage, row)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot read required runner JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RunnerError(f"runner JSON must be an object: {path}")
    return payload


def _load_npz_uri(uri: str) -> Mapping[str, np.ndarray]:
    if not uri.startswith("file://"):
        raise RunnerError(f"runner only supports local file NPZ URIs: {uri}")
    from urllib.parse import unquote, urlparse
    from urllib.request import url2pathname

    parsed = urlparse(uri)
    raw_path = url2pathname(unquote(parsed.path))
    if raw_path.startswith("/") and len(raw_path) > 2 and raw_path[2] == ":":
        raw_path = raw_path[1:]
    path = Path(raw_path)
    try:
        return dict(np.load(path, allow_pickle=False))
    except Exception as exc:
        raise RunnerError(f"cannot read prediction NPZ payload {path}: {exc}") from exc


def _source_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_object(args: Sequence[str], *, allow_empty: bool = False) -> str:
    try:
        result = subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True, timeout=30, cwd=str(_source_root())
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RunnerError(f"cannot resolve source git identity with git {' '.join(args)}") from exc
    value = result.stdout.strip().lower()
    if not value and not allow_empty:
        raise RunnerError(f"empty source git identity for git {' '.join(args)}")
    return value


def _project_identity() -> tuple[str, str]:
    return (
        _git_object(["rev-parse", "HEAD"]),
        _git_object(["rev-parse", "HEAD^{tree}"]),
    )


def _verify_clean_source_tree() -> None:
    dirty = _git_object(["status", "--porcelain"], allow_empty=True)
    if dirty:
        raise RunnerError("source worktree is dirty; refusing REAL/SMOKE execution with unverifiable project identity")


def _stage_to_prepared(stage: str) -> tuple[str, str, Path]:
    if stage == "preflight":
        return "smoke", "dev", Path("prepared/render_inputs_smoke.json")
    if stage == "smoke":
        return "smoke", "dev", Path("prepared/render_inputs_smoke.json")
    if stage == "test":
        return "test", "test", Path("prepared/render_inputs_test.json")
    raise RunnerError(f"unsupported runner stage: {stage}")


def _selected_models(model: str) -> tuple[str, ...]:
    normalized = model.lower()
    if normalized == "all":
        return MODELS
    if normalized == "vggt":
        return ("VGGT",)
    if normalized == "mast3r":
        return ("MASt3R",)
    raise RunnerError("--model must be vggt, mast3r, or all")


def _record_scene(record: Mapping[str, Any]) -> int:
    if "scene_id" in record:
        return int(record["scene_id"])
    key = str(record.get("sample_key", ""))
    for part in key.split("/"):
        if part.startswith("scan") and part[4:].isdigit():
            return int(part[4:])
    raise RunnerError("prepared record does not expose scene_id or scan sample_key")


def _record_view(record: Mapping[str, Any]) -> int:
    if "view_id" not in record:
        raise RunnerError("prepared record does not expose frozen view_id")
    return int(record["view_id"])


def _split_views(root: Path) -> dict[str, list[int]]:
    payload = _load_json(root / "manifests" / "split_view_manifest.json")
    views = payload.get("views")
    if not isinstance(views, dict):
        raise RunnerError("split/view manifest is missing views")
    return {str(key): [int(item) for item in value] for key, value in views.items()}


def _rendered_view(root: Path, stage_dir: str, scene: int, view: int, condition: str, severity: int) -> RenderedView:
    path = root / "rendered" / stage_dir / f"scan{scene:03d}_view{view:03d}_{condition}_s{severity}.png"
    metadata_path = path.with_suffix(".json")
    metadata = _load_json(metadata_path)
    digest = sha256_file(path)
    if metadata.get("rendered_png_sha256") != digest:
        raise RunnerError(f"rendered PNG digest mismatch: {path}")
    return RenderedView(
        view_id=view,
        png_path=path,
        png_sha256=digest,
        source_sha256=metadata.get("raw_source_sha256"),
    )


def _identity_payload(
    *, stage: str, model: str, sample_key: SampleKey, scene: int, condition: str,
    severity: int, rendered_views: Sequence[RenderedView], subset: tuple[int, int] | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "model": model,
        "sample_key": str(sample_key),
        "scene_id": scene,
        "condition": condition,
        "severity": severity,
        "subset": list(subset) if subset is not None else None,
        "rendered": [(view.view_id, view.png_sha256) for view in rendered_views],
    }


def build_schedule(root: Path, stage: str, *, model: str = "all") -> tuple[ScheduleItem, ...]:
    stage_dir, split, prepared_rel = _stage_to_prepared(stage)
    prepared = _load_json(root / prepared_rel)
    records = prepared.get("records")
    if not isinstance(records, list):
        raise RunnerError("prepared input is missing records")
    views_by_scene = _split_views(root)
    conditions = PREFLIGHT_CONDITIONS if stage == "preflight" else CONDITIONS
    scenes = sorted({_record_scene(record) for record in records})
    if stage == "preflight":
        scenes = scenes[:1]
    elif stage == "smoke":
        scenes = scenes[:10]
    elif stage == "test":
        scenes = list(TEST_SCENES)
    items: list[ScheduleItem] = []
    for selected_model in _selected_models(model):
        for scene in scenes:
            view_ids = views_by_scene.get(str(scene))
            if view_ids is None or len(view_ids) != 8:
                raise RunnerError(f"scene scan{scene:03d} does not have eight frozen views")
            for condition, severity in conditions:
                sample = SampleKey("dtu", split, f"scan{scene}", "views-0001", condition, str(severity), "0")
                rendered = tuple(_rendered_view(root, stage_dir, scene, view, condition, severity) for view in view_ids)
                payload = _identity_payload(
                    stage=stage, model=selected_model, sample_key=sample, scene=scene,
                    condition=condition, severity=severity, rendered_views=rendered,
                )
                items.append(ScheduleItem(stage, selected_model, sample, scene, condition, severity, rendered, _sha_json(payload)))
    return tuple(sorted(items, key=lambda item: item.identity))


def full_schedule(root: Path, stage: str) -> tuple[ScheduleItem, ...]:
    return build_schedule(root, stage, model="all")


def shard_schedule(items: Sequence[ScheduleItem], *, index: int, total: int) -> tuple[ScheduleItem, ...]:
    if total <= 0 or index < 0 or index >= total:
        raise RunnerError("--shard must be INDEX/TOTAL with 0 <= INDEX < TOTAL")
    return tuple(item for item in items if int(item.identity, 16) % total == index)


def zero_update_schedule_allowed(native_gate: Mapping[str, Any]) -> dict[str, Any]:
    if native_gate.get("schema_version") != "native-phenomenon-gate-v1":
        return {"status": "BLOCKED_PENDING_EVIDENCE", "reason_code": "NATIVE_CONFIDENCE_GATE_UNBOUND"}
    if native_gate.get("status") == "BLOCKED_PENDING_EVIDENCE":
        return {"status": "BLOCKED_PENDING_EVIDENCE", "reason_code": str(native_gate.get("reason_code", "NATIVE_CONFIDENCE_GATE_BLOCKED"))}
    if native_gate.get("status") == "FAIL":
        return {"status": "SHORT_CIRCUIT_P5", "reason_code": "NATIVE_CONFIDENCE_GATE_NOT_PASS"}
    if native_gate.get("status") != "PASS":
        return {"status": "BLOCKED_PENDING_EVIDENCE", "reason_code": "NATIVE_CONFIDENCE_GATE_INVALID"}
    return {"status": "PASS"}


def _derive_native_gate_from_p3(evidence_path: Path) -> tuple[str, tuple[str, ...]]:
    from .audit import AuditError, load_stage_evidence_manifest
    from .gates import GateStatus, evaluate_georeliab_gate

    try:
        evidence = load_stage_evidence_manifest(evidence_path)
    except AuditError as exc:
        raise RunnerError(f"native phenomenon gate cannot load P3 evidence: {exc}") from exc
    counts = dict(evidence.schedule_counts)
    if (
        counts.get("scheduled") != 400
        or counts.get("completed") != 400
        or counts.get("missing") != 0
    ):
        raise RunnerError("native phenomenon gate requires complete P3 evidence")
    evaluated = evaluate_georeliab_gate(evidence.to_gate_input())
    reasons = tuple(str(reason) for reason in evaluated.reason_codes)
    if evaluated.status is GateStatus.BLOCKED:
        if reasons == ("P5_DOWNSTREAM_SCHEDULE_COUNTS_INVALID",):
            return "PASS", reasons
        raise RunnerError("native phenomenon gate P3 decision remains non-terminal")
    if evaluated.status is not GateStatus.FAIL:
        raise RunnerError("native phenomenon gate expects a pre-P5 blocked or failed decision")
    if not reasons:
        raise RunnerError("native phenomenon gate P3 failure has no reason code")
    return "FAIL", reasons


def load_native_phenomenon_gate(context: RunnerContext) -> dict[str, Any]:
    path = context.output_root / "stage" / "test" / "native_phenomenon_gate.json"
    if not path.exists():
        return {"schema_version": "native-phenomenon-gate-v1", "status": "BLOCKED_PENDING_EVIDENCE", "reason_code": "NATIVE_CONFIDENCE_GATE_MISSING"}
    payload = _load_json(path)
    if payload.get("schema_version") != "native-phenomenon-gate-v1" or payload.get("status") not in {"PASS", "FAIL"}:
        return {"schema_version": "native-phenomenon-gate-v1", "status": "BLOCKED_PENDING_EVIDENCE", "reason_code": "NATIVE_CONFIDENCE_GATE_INVALID"}
    freeze_path = context.output_root / "stage" / "test" / "stage_freeze.json"
    if not freeze_path.exists():
        return {"schema_version": "native-phenomenon-gate-v1", "status": "BLOCKED_PENDING_EVIDENCE", "reason_code": "TEST_STAGE_FREEZE_MISSING"}
    freeze = _load_json(freeze_path)
    if payload.get("stage_freeze_sha256") != sha256_file(freeze_path) or payload.get("test_stage_fingerprint") != freeze.get("stage_fingerprint"):
        return {"schema_version": "native-phenomenon-gate-v1", "status": "BLOCKED_PENDING_EVIDENCE", "reason_code": "NATIVE_CONFIDENCE_GATE_FREEZE_MISMATCH"}
    evidence_path = context.output_root / "stage" / "test" / "stage_evidence_p3.json"
    if not payload.get("p3_stage_evidence_sha256") or not evidence_path.exists() or payload.get("p3_stage_evidence_sha256") != sha256_file(evidence_path):
        return {"schema_version": "native-phenomenon-gate-v1", "status": "BLOCKED_PENDING_EVIDENCE", "reason_code": "NATIVE_CONFIDENCE_GATE_EVIDENCE_MISMATCH"}
    try:
        expected_status, expected_reasons = _derive_native_gate_from_p3(evidence_path)
    except RunnerError:
        return {"schema_version": "native-phenomenon-gate-v1", "status": "BLOCKED_PENDING_EVIDENCE", "reason_code": "NATIVE_CONFIDENCE_GATE_RECOMPUTE_FAILED"}
    if payload.get("status") != expected_status or tuple(payload.get("reason_codes", ())) != expected_reasons:
        return {"schema_version": "native-phenomenon-gate-v1", "status": "BLOCKED_PENDING_EVIDENCE", "reason_code": "NATIVE_CONFIDENCE_GATE_DECISION_MISMATCH"}
    return payload


def write_native_phenomenon_gate_from_audit_output(context: RunnerContext, audit_output: Mapping[str, Any]) -> Path:
    freeze_path = context.output_root / "stage" / "test" / "stage_freeze.json"
    evidence_path = context.output_root / "stage" / "test" / "stage_evidence_p3.json"
    if not freeze_path.exists() or not evidence_path.exists():
        raise RunnerError("native phenomenon gate requires frozen P3 stage evidence")
    expected_evidence_sha = sha256_file(evidence_path)
    expected_evidence_path = str(evidence_path)
    supplied_evidence_path = str(audit_output.get("stage_evidence_path", ""))
    supplied_evidence_sha = str(audit_output.get("stage_evidence_sha256", ""))
    if supplied_evidence_path != expected_evidence_path or supplied_evidence_sha != expected_evidence_sha:
        raise RunnerError("native phenomenon gate audit output is not bound to the current P3 stage evidence")
    status, reason_codes = _derive_native_gate_from_p3(evidence_path)
    supplied_gate = audit_output.get("georeliab_gate")
    if not isinstance(supplied_gate, Mapping):
        raise RunnerError("native phenomenon gate requires georeliab_gate from audit-georeliab")
    supplied_reasons = tuple(str(item) for item in supplied_gate.get("reason_codes", ()))
    if supplied_reasons != reason_codes:
        raise RunnerError("native phenomenon gate reason_codes do not match recomputed P3 evidence")
    skip_reason = audit_output.get("p5_skip_reason")
    if status == "PASS" and skip_reason is not None:
        raise RunnerError("native phenomenon PASS cannot carry a P5 skip reason")
    if "CONFIDENCE_FAILURE_GATE_NOT_MET" in reason_codes and skip_reason != "P4_CONFIDENCE_PHENOMENON_GATE_FAILED":
        raise RunnerError("native phenomenon confidence failure requires the frozen P5 skip reason")
    freeze = _load_json(freeze_path)
    payload = {
        "schema_version": "native-phenomenon-gate-v1",
        "status": status,
        "reason_codes": list(reason_codes),
        "test_stage_fingerprint": freeze.get("stage_fingerprint"),
        "stage_freeze_sha256": sha256_file(freeze_path),
        "p3_stage_evidence_path": expected_evidence_path,
        "p3_stage_evidence_sha256": expected_evidence_sha,
    }
    output = context.output_root / "stage" / "test" / "native_phenomenon_gate.json"
    if output.exists():
        existing = _load_json(output)
        if existing != payload:
            raise RunnerError("immutable native_phenomenon_gate.json would change")
        return output
    _atomic_json(output, payload)
    return output


def build_zero_update_schedule(root: Path, native_gate: Mapping[str, Any], *, model: str = "all") -> tuple[ScheduleItem, ...]:
    allowed = zero_update_schedule_allowed(native_gate)
    if allowed["status"] != "PASS":
        return ()
    base = [item for item in build_schedule(root, "test", model=model) if item.condition in {"fog", "low-light-noise", "defocus"} and item.severity == 2]
    items: list[ScheduleItem] = []
    for item in base:
        for subset in ZERO_UPDATE_SUBSETS:
            payload = _identity_payload(
                stage="zero-update", model=item.model, sample_key=item.sample_key,
                scene=item.scene_id, condition=item.condition, severity=item.severity,
                rendered_views=item.rendered_views, subset=subset,
            )
            items.append(ScheduleItem("zero-update", item.model, item.sample_key, item.scene_id, item.condition, item.severity, item.rendered_views, _sha_json(payload), subset, item.identity))
    return tuple(sorted(items, key=lambda entry: entry.identity))


def frozen_model_spec(model: str, config_path: Path | None) -> ModelSpec:
    defaults: dict[str, dict[str, str | None]] = {
        "VGGT": {
            "checkpoint_sha256": "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0",
            "source_commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
            "python": "3.10.20",
            "torch": "2.3.1+cu121",
            "typing_extensions_site": str(FROZEN_TYPING_EXTENSIONS_SITE),
            "typing_extensions_sha256": FROZEN_TYPING_EXTENSIONS_SHA256,
            "typing_extensions_version": FROZEN_TYPING_EXTENSIONS_VERSION,
        },
        "MASt3R": {
            "checkpoint_sha256": "0a615eb05fa9db654050aa655945ee5696e7c6c1b7f93f1ee8c37249010f6feb",
            "config_sha256": "718eb93dc4f9e4332b60cc0041af962d712cbd346d7770ce35c5b22cff68eae4",
            "source_commit": "f5209afc300cec36239a7ac992263f36847bbba0",
            "python": "3.10.20",
            "torch": "2.5.1+cu121",
            "dust3r_source_commit": "3cc8c88c413bb9e34c41db0e0eef99c2ee010b12",
            "croco_source_commit": "d7de0705845239092414480bd829228723bf20de",
            "typing_extensions_site": str(FROZEN_TYPING_EXTENSIONS_SITE),
            "typing_extensions_sha256": FROZEN_TYPING_EXTENSIONS_SHA256,
            "typing_extensions_version": FROZEN_TYPING_EXTENSIONS_VERSION,
        },
    }
    if model not in defaults:
        raise RunnerError(f"unsupported model: {model}")
    values = dict(defaults[model])
    overlay_digest = "0" * 64
    if config_path is not None and config_path.exists():
        overlay_digest = sha256_file(config_path)
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
        resources = payload.get("resources", {}) if isinstance(payload, dict) else {}
        runtime = payload.get("runtime", {}) if isinstance(payload, dict) else {}
        values["typing_extensions_site"] = str(runtime.get("typing_extensions_site", values["typing_extensions_site"]))
        values["typing_extensions_sha256"] = str(resources.get("typing_extensions_sha256", values["typing_extensions_sha256"]))
        values["typing_extensions_version"] = str(resources.get("typing_extensions_version", values["typing_extensions_version"]))
        if model == "VGGT":
            values["checkpoint_sha256"] = str(resources.get("vggt_checkpoint_sha256", values["checkpoint_sha256"]))
            values["source_commit"] = str(resources.get("vggt_source_commit", values["source_commit"]))
            values["python"] = str(runtime.get("vggt_python", values["python"]))
            values["torch"] = str(runtime.get("vggt_torch", values["torch"]))
        else:
            values["checkpoint_sha256"] = str(resources.get("mast3r_checkpoint_sha256", values["checkpoint_sha256"]))
            values["config_sha256"] = str(resources.get("mast3r_config_sha256", values["config_sha256"]))
            values["source_commit"] = str(resources.get("mast3r_source_commit", values["source_commit"]))
            values["python"] = str(runtime.get("mast3r_python", values["python"]))
            values["torch"] = str(runtime.get("mast3r_torch", values["torch"]))
            values["dust3r_source_commit"] = str(resources.get("dust3r_source_commit", values["dust3r_source_commit"]))
            values["croco_source_commit"] = str(resources.get("croco_source_commit", values["croco_source_commit"]))
    return ModelSpec(
        name=model,
        checkpoint_sha256=str(values["checkpoint_sha256"]),
        source_commit=str(values["source_commit"]),
        python=str(values["python"]),
        torch=str(values["torch"]),
        environment_lock_sha256=overlay_digest,
        typing_extensions_site=str(values["typing_extensions_site"]),
        typing_extensions_sha256=str(values["typing_extensions_sha256"]),
        typing_extensions_version=str(values["typing_extensions_version"]),
        config_sha256=values.get("config_sha256"),
        dust3r_source_commit=values.get("dust3r_source_commit"),
        croco_source_commit=values.get("croco_source_commit"),
    )


def _model_specs(config_path: Path | None) -> dict[str, ModelSpec]:
    return {model: frozen_model_spec(model, config_path) for model in MODELS}


def _stage_fingerprint(context: RunnerContext, stage: str, items: Sequence[ScheduleItem] | None = None) -> dict[str, Any]:
    if items is None:
        items = full_schedule(context.root, stage)
    project_commit, project_tree = _project_identity()
    stage_dir, _split, prepared_rel = _stage_to_prepared("test" if stage == "test" else "smoke") if stage in {"test", "smoke", "preflight"} else ("test", "test", Path("prepared/render_inputs_test.json"))
    prepared_path = context.root / prepared_rel
    prepared_payload = _load_json(prepared_path)
    split_path = context.root / "manifests" / "split_view_manifest.json"
    materialization_path = context.root / "manifests" / "frozen_materialization.json"
    corruption_path = context.root / "manifests" / "corruption_calibration.json"
    optional = {
        "corruption_calibration_qa_sha256": context.root / "manifests" / "corruption_calibration_qa.json",
        "tartanair_native_fog_sanity_sha256": context.root / "evidence" / "tartanair_native_fog_sanity.json",
        "test_render_lock_sha256": context.root / "manifests" / "test_render_lock.json",
        "test_render_index_sha256": context.root / "manifests" / "test_render_index.json",
        "tartanair_prepared_input_sha256": context.root / "prepared" / "tartanair_p000_pairs.json",
    }
    optional_hashes = {key: sha256_file(path) for key, path in optional.items() if path.exists()}
    model_specs = {key: asdict(value) for key, value in _model_specs(context.config_path).items()}
    render_fingerprint = _sha_json([
        (item.identity, [(view.view_id, view.png_sha256, view.source_sha256) for view in item.rendered_views])
        for item in items
    ])
    payload = {
        "schema_version": "georeliab-stage-fingerprint-v1",
        "stage": stage,
        "project_commit": project_commit,
        "project_tree": project_tree,
        "model_specs": model_specs,
        "environment_lock_sha256": _sha_json(model_specs),
        "split_view_manifest_sha256": sha256_file(split_path),
        "frozen_materialization_sha256": sha256_file(materialization_path),
        "prepared_materialization_sha256": str(prepared_payload.get("materialization_sha256", "")),
        "corruption_manifest_sha256": sha256_file(corruption_path),
        "prepared_input_sha256": sha256_file(prepared_path),
        "schedule_count": len(items),
        "schedule_fingerprint": _sha_json([item.identity for item in items]),
        "render_digest_fingerprint": render_fingerprint,
        **optional_hashes,
    }
    payload["stage_fingerprint"] = _sha_json(payload)
    return payload


def make_manifest(item: ScheduleItem, root: Path, model_specs: Mapping[str, ModelSpec], *, device: str) -> RunManifest:
    spec = model_specs[item.model]
    split_sha = sha256_file(root / "manifests" / "split_view_manifest.json")
    corruption_sha = sha256_file(root / "manifests" / "corruption_calibration.json")
    project_commit, project_tree = _project_identity()
    mode = RunMode.SMOKE if item.stage == "smoke" else RunMode.REAL
    validity = ScientificValidity.NON_SCIENTIFIC_SMOKE if mode is RunMode.SMOKE else ScientificValidity.SCIENTIFIC
    provenance = ScientificProvenance(
        project_commit=project_commit,
        project_tree=project_tree,
        model_source_commit=spec.source_commit,
        environment_lock_sha256=spec.environment_lock_sha256,
        corruption_manifest_sha256=corruption_sha,
        split_view_manifest_sha256=split_sha,
        dust3r_source_commit=spec.dust3r_source_commit,
        croco_source_commit=spec.croco_source_commit,
    )
    return RunManifest(
        run_id=f"{item.stage}-{item.model.lower()}-{item.identity[:16]}",
        mode=mode,
        scientific_validity=validity,
        model=item.model,
        checkpoint_hash=spec.checkpoint_sha256,
        dataset=item.sample_key.dataset,
        split=item.sample_key.split,
        seed=int(item.sample_key.seed),
        intervention_version="none",
        corruption_version="georeliab-c-v1",
        environment={
            "device": device,
            "python": spec.python,
            "torch": spec.torch,
            "typing_extensions_site": spec.typing_extensions_site,
            "typing_extensions_sha256": spec.typing_extensions_sha256,
            "typing_extensions_version": spec.typing_extensions_version,
            **({"config_sha256": spec.config_sha256} if spec.config_sha256 else {}),
        },
        rgb_digest=_canonical_ordered_png_bundle_digest(
            item.rendered_views,
            expected_count=6 if item.stage == "zero-update" else 8,
        ),
        prompt_digest="fixed-empty-prompt",
        decoder_digest="fixed-native-decoder",
        provenance=provenance,
    )


def _empty_invalid_prediction(manifest: RunManifest, sample_key: SampleKey, output_dir: Path, reason: str) -> PredictionArtifact:
    output_dir.mkdir(parents=True, exist_ok=True)
    geo = output_dir / "geometry_prediction.npz"
    conf = output_dir / "native_confidence.npz"
    mask = output_dir / "valid_mask.npz"
    np.savez(
        geo,
        points_world=np.empty((0, 3)),
        camera_c2w=np.empty((0, 4, 4)),
        intrinsics=np.empty((0, 3, 3)),
        pixel_xy=np.empty((0, 2)),
        view_id=np.empty((0,), dtype=np.int64),
        metadata=json.dumps({"invalid_reason": reason, "reason_code": "ADAPTER_EXCEPTION"}),
    )
    np.savez(conf, raw_confidence=np.empty((0,)))
    np.savez(mask, valid_mask=np.empty((0,), dtype=bool))
    return PredictionArtifact(
        manifest.run_id,
        str(sample_key),
        geo.as_uri(),
        conf.as_uri(),
        mask.as_uri(),
        None,
        0.0,
        0.0,
        True,
        {"geometry_prediction_uri": sha256_file(geo), "native_confidence_uri": sha256_file(conf), "valid_mask_uri": sha256_file(mask)},
    )


def default_adapter_factory(model: str, context: RunnerContext) -> Any:
    from .adapters import MASt3RAdapter, RealMASt3RUpstream, RealVGGTUpstream, VGGTAdapter

    frozen = _frozen_runtime_from_config(model, context)
    key = (
        model,
        context.device,
        str(frozen.source),
        frozen.source_commit,
        str(frozen.environment),
        frozen.python_version,
        frozen.torch_version,
        str(frozen.checkpoint),
        frozen.checkpoint_sha256,
        str(getattr(frozen, "config", None)),
        getattr(frozen, "config_sha256", None),
        str(getattr(frozen, "dust3r_source", None)),
        getattr(frozen, "dust3r_source_commit", None),
        str(getattr(frozen, "croco_source", None)),
        getattr(frozen, "croco_source_commit", None),
        str(frozen.typing_extensions_site),
        frozen.typing_extensions_sha256,
        frozen.typing_extensions_version,
    )
    if model == "VGGT":
        upstream = _UPSTREAM_CACHE.get(key)
        if upstream is None:
            upstream = RealVGGTUpstream(frozen, device=context.device)
            _UPSTREAM_CACHE[key] = upstream
        return VGGTAdapter(frozen, output_root=context.output_root, device=context.device, upstream=upstream)
    if model == "MASt3R":
        cache_dir = context.output_root / "mast3r_cache"
        upstream = _UPSTREAM_CACHE.get(key)
        if upstream is None:
            upstream = RealMASt3RUpstream(frozen, device=context.device, cache_dir=cache_dir)
            _UPSTREAM_CACHE[key] = upstream
        if hasattr(upstream, "cache_dir"):
            upstream.cache_dir = cache_dir
        return MASt3RAdapter(frozen, output_root=context.output_root, device=context.device, cache_dir=cache_dir, upstream=upstream)
    raise RunnerError(f"unsupported model: {model}")


def _overlay_payload(context: RunnerContext) -> dict[str, Any]:
    if context.config_path is None or not context.config_path.exists():
        raise RunnerError("real execution requires frozen A100 overlay config")
    try:
        return tomllib.loads(context.config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RunnerError(f"cannot read frozen A100 overlay config: {context.config_path}") from exc


def _verify_output_root_policy(context: RunnerContext) -> None:
    payload = _overlay_payload(context)
    runtime = payload.get("runtime", {})
    configured = str(runtime.get("root", "")).strip()
    normalized = configured.replace("\\", "/").rstrip("/")
    if not configured:
        raise RunnerError("frozen A100 overlay is missing runtime.root")
    if normalized == "/home" or normalized.startswith("/home/"):
        raise RunnerError("real execution refuses every output below /home")
    try:
        configured_root = Path(configured).resolve()
        output_root = context.output_root.resolve()
        if output_root == configured_root:
            return
        try:
            relative = output_root.relative_to(configured_root)
        except ValueError:
            relative = None
        if relative is not None and relative.parts in {
            ("preflight-real", "repeat-a"),
            ("preflight-real", "repeat-b"),
        }:
            return
        if output_root != configured_root:
            raise RunnerError("real output root must equal the frozen A100 runtime.root")
    except OSError as exc:
        raise RunnerError("cannot resolve the frozen real output root") from exc


def _frozen_runtime_from_config(model: str, context: RunnerContext) -> Any:
    from .adapters import FrozenRuntime

    payload = _overlay_payload(context)
    runtime = payload.get("runtime", {})
    resources = payload.get("resources", {})
    if model == "VGGT":
        return FrozenRuntime(
            source=Path(runtime["vggt_source"]), source_commit=resources["vggt_source_commit"],
            environment=Path(runtime["vggt_env"]), python_version=runtime["vggt_python"], torch_version=runtime["vggt_torch"],
            checkpoint=Path(resources["vggt_checkpoint"]), checkpoint_sha256=resources["vggt_checkpoint_sha256"],
            typing_extensions_site=Path(runtime.get("typing_extensions_site", FROZEN_TYPING_EXTENSIONS_SITE)),
            typing_extensions_sha256=resources.get("typing_extensions_sha256", FROZEN_TYPING_EXTENSIONS_SHA256),
            typing_extensions_version=resources.get("typing_extensions_version", FROZEN_TYPING_EXTENSIONS_VERSION),
        )
    if model == "MASt3R":
        return FrozenRuntime(
            source=Path(runtime["mast3r_source"]), source_commit=resources["mast3r_source_commit"],
            environment=Path(runtime["mast3r_env"]), python_version=runtime["mast3r_python"], torch_version=runtime["mast3r_torch"],
            checkpoint=Path(resources["mast3r_checkpoint"]), checkpoint_sha256=resources["mast3r_checkpoint_sha256"],
            config=Path(resources["mast3r_config"]), config_sha256=resources["mast3r_config_sha256"],
            dust3r_source=Path(runtime["dust3r_source"]), dust3r_source_commit=resources["dust3r_source_commit"],
            croco_source=Path(runtime["croco_source"]), croco_source_commit=resources["croco_source_commit"],
            typing_extensions_site=Path(runtime.get("typing_extensions_site", FROZEN_TYPING_EXTENSIONS_SITE)),
            typing_extensions_sha256=resources.get("typing_extensions_sha256", FROZEN_TYPING_EXTENSIONS_SHA256),
            typing_extensions_version=resources.get("typing_extensions_version", FROZEN_TYPING_EXTENSIONS_VERSION),
        )
    raise RunnerError(f"unsupported model: {model}")


def _verify_frozen_runtimes(context: RunnerContext, model: str) -> None:
    from .adapters import AdapterError, verify_frozen_runtime

    try:
        for selected_model in _selected_models(model):
            verify_frozen_runtime(selected_model, _frozen_runtime_from_config(selected_model, context))
    except (AdapterError, KeyError, TypeError, OSError) as exc:
        raise RunnerError(f"frozen runtime verification failed before real dispatch: {exc}") from exc


def production_audit_factory(_manifest: RunManifest, _prediction: PredictionArtifact, _output_dir: Path) -> AuditRecord:
    raise RunnerError(
        "production audit factory requires explicit Task3 DTU GT audit binding; "
        "tests may pass a fixture audit factory explicitly"
    )


def make_production_audit_factory(context: RunnerContext) -> AuditFactory:
    """Bind the real frozen-DTU audit implementation without importing it on dry-run paths."""
    try:
        from .runner_audit import audit_prediction_with_frozen_dtu
    except ImportError as exc:  # pragma: no cover - exercised only before Task3/Task5 audit module lands
        raise RunnerError("production DTU audit implementation is unavailable") from exc

    def audit(manifest: RunManifest, prediction: PredictionArtifact, output_dir: Path) -> AuditRecord:
        return audit_prediction_with_frozen_dtu(
            root=context.root,
            manifest=manifest,
            prediction=prediction,
            output_dir=output_dir,
        )

    return audit


def _bundle_dir(output_root: Path, item: ScheduleItem) -> Path:
    return output_root / "stage" / item.stage / "bundles" / item.model.lower() / item.identity


def _zero_update_bundle_dir(output_root: Path, item: ScheduleItem) -> Path:
    return output_root / "stage" / "zero-update" / "bundles" / item.model.lower() / item.identity


def load_completed_bundle(bundle_dir: Path) -> tuple[RunManifest, PredictionArtifact, AuditRecord]:
    manifest = read_json_artifact(bundle_dir / "run_manifest.json", RunManifest)
    prediction = read_json_artifact(bundle_dir / "prediction_artifact.json", PredictionArtifact)
    audit = read_json_artifact(bundle_dir / "audit_record.json", AuditRecord)
    validate_artifact_bundle(manifest, prediction, audit)
    return manifest, prediction, audit


def _subset_views(item: ScheduleItem) -> tuple[RenderedView, ...]:
    if item.subset is None:
        return item.rendered_views
    omitted = set(item.subset)
    return tuple(view for index, view in enumerate(item.rendered_views) if index not in omitted)


def _parent_bundle_for_zero_update(context: RunnerContext, item: ScheduleItem) -> tuple[Path, RunManifest, PredictionArtifact, AuditRecord]:
    if item.parent_identity is None:
        raise RunnerError("zero-update item is missing parent P3 identity")
    parent_dir = context.output_root / "stage" / "test" / "bundles" / item.model.lower() / item.parent_identity
    manifest, prediction, audit = load_completed_bundle(parent_dir)
    if manifest.model != item.model or prediction.sample_key != str(item.sample_key):
        raise RunnerError("zero-update parent linkage does not match model/sample_key")
    key = SampleKey.parse(prediction.sample_key)
    if key.split != "test" or key.severity != "2" or key.condition not in {"fog", "low-light-noise", "defocus"}:
        raise RunnerError("zero-update parent must be a severity-2 P3 corruption bundle")
    freeze_path = context.output_root / "stage" / "test" / "stage_freeze.json"
    stage_item_path = parent_dir / "stage_item.json"
    if not freeze_path.exists() or not stage_item_path.exists():
        raise RunnerError("zero-update parent is not bound to a frozen P3 stage")
    freeze = _load_json(freeze_path)
    stage_item = _load_json(stage_item_path)
    if stage_item.get("stage_fingerprint") != freeze.get("stage_fingerprint"):
        raise RunnerError("zero-update parent stage fingerprint does not match current P3 freeze")
    return parent_dir, manifest, prediction, audit


def _validate_zero_subset_npz(path: Path, item: ScheduleItem, *, parent_manifest: RunManifest | None = None, parent_prediction: PredictionArtifact | None = None) -> None:
    payload = dict(np.load(path, allow_pickle=False))
    required = {"points", "camera_centers", "view_ids", "parent_model", "parent_sample_key", "parent_project_commit", "parent_run_id"}
    if set(payload) < required:
        raise RunnerError("zero-update subset artifact is missing required fields")
    points = np.asarray(payload["points"], dtype=np.float64)
    centers = np.asarray(payload["camera_centers"], dtype=np.float64)
    view_ids = np.asarray(payload["view_ids"], dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise RunnerError("zero-update subset points must be finite Nx3")
    if points.shape[0] == 0:
        raise RunnerError("zero-update subset points must be non-empty for evidence")
    if centers.shape != (6, 3) or not np.isfinite(centers).all():
        raise RunnerError("zero-update subset camera_centers must be finite 6x3")
    expected = np.asarray([index for index in range(8) if item.subset is None or index not in set(item.subset)], dtype=np.int64)
    if not np.array_equal(view_ids, expected):
        raise RunnerError("zero-update subset view_ids must be full-view ordinal indexes")
    if str(payload["parent_model"]) != item.model or str(payload["parent_sample_key"]) != str(item.sample_key):
        raise RunnerError("zero-update subset parent linkage mismatch")
    if parent_manifest is not None:
        parent_commit = parent_manifest.provenance.project_commit if parent_manifest.provenance else ""
        if str(payload["parent_project_commit"]) != str(parent_commit):
            raise RunnerError("zero-update subset parent project commit mismatch")
        if str(payload["parent_run_id"]) != str(parent_manifest.run_id):
            raise RunnerError("zero-update subset parent run id mismatch")
    if parent_prediction is not None and str(payload["parent_sample_key"]) != str(parent_prediction.sample_key):
        raise RunnerError("zero-update subset parent prediction sample mismatch")


def _rewrite_bundle_uri_payloads(bundle_dir: Path, partial_dir: Path, *, work_dir: Path | None = None) -> None:
    target_dir = work_dir or bundle_dir
    partial_uri = partial_dir.as_uri()
    final_uri = bundle_dir.as_uri()
    for path in sorted(target_dir.rglob("*.json")):
        payload = _load_json(path)

        def rewrite(value: Any) -> Any:
            if isinstance(value, str):
                return value.replace(partial_uri, final_uri)
            if isinstance(value, dict):
                return {key: rewrite(child) for key, child in value.items()}
            if isinstance(value, list):
                return [rewrite(child) for child in value]
            return value

        rewritten = rewrite(payload)
        if rewritten != payload:
            _atomic_json(path, rewritten)


def claim_item(
    claim_root: Path,
    item: ScheduleItem,
    *,
    stale_seconds: int = CLAIM_STALE_SECONDS,
) -> ClaimResult:
    claim_root.mkdir(parents=True, exist_ok=True)
    path = claim_root / f"{item.identity}.lock"
    payload = {"identity": item.identity, "created_at": _utc_now(), "pid": os.getpid()}
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(path), flags)
    except FileExistsError:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            created = datetime.fromisoformat(str(existing.get("created_at", "")).replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - created).total_seconds()
        except Exception:
            age = stale_seconds + 1
        if age <= stale_seconds:
            return ClaimResult(path, False, False, "already-claimed")
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        reclaimed = claim_item(claim_root, item, stale_seconds=stale_seconds)
        return ClaimResult(reclaimed.path, reclaimed.acquired, True, reclaimed.reason)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    return ClaimResult(path, True)


def _release_claim(claim: ClaimResult) -> None:
    if claim.acquired:
        try:
            claim.path.unlink()
        except FileNotFoundError:
            pass


def _stage_item_payload_for_context(item: ScheduleItem, fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "georeliab-schedule-item-v1",
        "identity": item.identity,
        "stage": item.stage,
        "model": item.model,
        "sample_key": str(item.sample_key),
        "scene_id": item.scene_id,
        "condition": item.condition,
        "severity": item.severity,
        "subset": list(item.subset) if item.subset is not None else None,
        "parent_identity": item.parent_identity,
        "stage_fingerprint": fingerprint["stage_fingerprint"],
        "freeze": dict(fingerprint),
    }


def _existing_bundle_state(bundle_dir: Path, item: ScheduleItem, fingerprint: Mapping[str, Any]) -> str | None:
    if not bundle_dir.exists():
        return None
    try:
        payload = _load_json(bundle_dir / "stage_item.json")
        if payload.get("identity") != item.identity:
            raise RunnerError("provenance-conflict: existing valid artifact has different schedule identity")
        if payload.get("stage_fingerprint") != fingerprint.get("stage_fingerprint"):
            raise RunnerError("provenance-conflict: existing valid artifact has different stage fingerprint")
        load_completed_bundle(bundle_dir)
        summary = _load_json(bundle_dir / "scene_summary.json")
        if summary.get("schema_version") != "validated-scene-summary-v1" or summary.get("producer") != "audit-georeliab":
            raise RunnerError("provenance-conflict: existing artifact is missing validated scene summary")
        if summary.get("audit_sha256") != sha256_file(bundle_dir / "audit_record.json"):
            raise RunnerError("provenance-conflict: existing scene summary is not bound to audit digest")
    except (RunnerError, ContractError, OSError, ValueError) as exc:
        raise RunnerError(f"provenance-conflict: existing artifact is not a valid matching bundle: {exc}") from exc
    return "complete"


def _write_scene_summary(bundle_dir: Path, manifest: RunManifest, prediction: PredictionArtifact, audit: AuditRecord, audit_path: Path) -> None:
    dense_summary: dict[str, Any] = {"voxel_count": 0, "failure_count_2mm": int(audit.failure_label)}
    dense_uri = audit.metadata.get("dense_audit_uri")
    if dense_uri:
        try:
            dense = _load_npz_uri(dense_uri)
        except RunnerError:
            dense = dict(np.load(bundle_dir / Path(dense_uri).name, allow_pickle=False))
        try:
            gt_error = np.asarray(dense["gt_error"], dtype=np.float64)
            failure = np.asarray(dense["failure_label"], dtype=bool)
            dense_summary = {
                "voxel_count": int(len(np.asarray(dense["voxel_points"]))),
                "failure_count_2mm": int(np.count_nonzero(failure)),
                "mean_error_mm": float(np.mean(gt_error[np.isfinite(gt_error)])) if np.any(np.isfinite(gt_error)) else None,
            }
        except Exception as exc:
            dense_summary = {"voxel_count": 0, "failure_count_2mm": int(audit.failure_label), "summary_error": str(exc)}
    payload = {
        "schema_version": "validated-scene-summary-v1",
        "producer": "audit-georeliab",
        "run_id": manifest.run_id,
        "sample_key": prediction.sample_key,
        "audit_sha256": sha256_file(audit_path),
        "dense_audit_uri": audit.metadata.get("dense_audit_uri", ""),
        "invalid_counts": {"invalid_prediction": int(prediction.invalid_prediction)},
        "schedule_counts": {"bundle": 1},
        "summary": dense_summary,
        "corruption_severity_monotonic": False,
        "cross_view_consistent": False,
        "gt_geometry_invariant": False,
    }
    _atomic_json(bundle_dir / "scene_summary.json", payload)


def _append_ledger(output_root: Path, stage: str, row: Mapping[str, Any]) -> None:
    path = output_root / "stage" / stage / "ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    acquired = False
    for _ in range(200):
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, PermissionError):
            try:
                lock_age = time.time() - lock.stat().st_mtime
            except FileNotFoundError:
                continue
            except OSError:
                lock_age = 0.0
            if lock_age > LEDGER_LOCK_STALE_SECONDS:
                try:
                    lock.unlink()
                except (FileNotFoundError, PermissionError):
                    pass
                continue
            time.sleep(0.005)
            continue
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"created_at": _utc_now(), "pid": os.getpid()}, handle, sort_keys=True)
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass
            acquired = True
            break
    if not acquired:
        raise RunnerError("ledger lock acquisition timed out")
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def _artifact_digest_map(bundle_dir: Path, names: Sequence[str]) -> dict[str, str]:
    return {name: sha256_file(bundle_dir / name) for name in names if (bundle_dir / name).exists()}


def _next_ledger_attempt(output_root: Path, stage: str, identity: str) -> int:
    path = output_root / "stage" / stage / "ledger.jsonl"
    if not path.exists():
        return 1
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("item_identity", row.get("identity", ""))) == identity:
            count += 1
    return count + 1


def read_stage_ledger(output_root: Path, stage: str) -> dict[str, Any]:
    path = output_root / "stage" / stage / "ledger.jsonl"
    rows: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    counts = {"scheduled": 0, "completed": 0, "invalid": 0, "skipped": 0, "retried": 0, "missing": 0}
    latest: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        latest[str(row.get("item_identity", row.get("identity", index)))] = row
    for row in latest.values():
        state = row.get("state")
        if state in {"completed", "skipped"}:
            counts["completed"] += 1
        elif state in counts:
            counts[state] += 1
        if state == "skipped":
            counts["skipped"] += 1
        if row.get("invalid_prediction") is True:
            counts["invalid"] += 1
        if row.get("retried") is True:
            counts["retried"] += 1
    return {"schema_version": "georeliab-stage-ledger-v1", "stage": stage, "counts": counts, "rows": rows}


def _latest_stage_rows(output_root: Path, stage: str) -> dict[str, dict[str, Any]]:
    path = output_root / "stage" / stage / "ledger.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        latest[str(row.get("item_identity", row.get("identity", index)))] = row
    return latest


def stage_progress_counts(output_root: Path, stage: str, items: Sequence[ScheduleItem]) -> dict[str, int]:
    latest = _latest_stage_rows(output_root, stage)
    counts = {"scheduled": len(items), "completed": 0, "invalid": 0, "missing": 0, "skipped": 0, "retried": 0}
    for item in items:
        row = latest.get(item.identity)
        if row is None:
            counts["missing"] += 1
            continue
        state = row.get("state")
        if state in {"completed", "skipped"}:
            counts["completed"] += 1
        else:
            counts["missing"] += 1
        if state == "skipped":
            counts["skipped"] += 1
        if row.get("invalid_prediction") is True:
            counts["invalid"] += 1
        if row.get("retried") is True:
            counts["retried"] += 1
    return counts


def execute_item(
    context: RunnerContext,
    item: ScheduleItem,
    *,
    adapter_factory: AdapterFactory = default_adapter_factory,
    audit_factory: AuditFactory = production_audit_factory,
    stage_fingerprint: Mapping[str, Any] | None = None,
) -> RunResult:
    if audit_factory is production_audit_factory:
        raise RunnerError("production audit factory must be explicitly configured before execution")
    fingerprint = dict(stage_fingerprint or _stage_fingerprint(context, item.stage, full_schedule(context.root, item.stage) if item.stage in {"smoke", "test"} else build_schedule(context.root, item.stage, model="all")))
    bundle_dir = _bundle_dir(context.output_root, item)
    if _existing_bundle_state(bundle_dir, item, fingerprint) == "complete":
        existing_prediction = read_json_artifact(bundle_dir / "prediction_artifact.json", PredictionArtifact)
        invalid_existing = bool(existing_prediction.invalid_prediction)
        reason_code = "INVALID_PREDICTION" if invalid_existing else "EXISTING_VALID_ARTIFACT"
        result = RunResult(item.identity, "skipped", bundle_dir, invalid_existing, reason_code, 0.0, existing_prediction.peak_memory_mb)
        _append_ledger(context.output_root, item.stage, {**asdict(result), "item_identity": item.identity, "timestamp": _utc_now(), "attempt": _next_ledger_attempt(context.output_root, item.stage, item.identity), "artifact_digests": _artifact_digest_map(bundle_dir, ("run_manifest.json", "prediction_artifact.json", "audit_record.json", "stage_item.json", "scene_summary.json")), "artifact_bytes": _dir_size(bundle_dir)})
        return result
    claim = claim_item(context.output_root / "stage" / item.stage / "claims", item)
    if not claim.acquired:
        raise RunnerError(f"schedule item is already claimed: {item.identity}")
    partial = bundle_dir.with_name(bundle_dir.name + ".partial")
    retried = partial.exists()
    start = time.perf_counter()
    try:
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True, exist_ok=True)
        specs = _model_specs(context.config_path)
        manifest = make_manifest(item, context.root, specs, device=context.device)
        reason_code = "OK"
        try:
            item_context = RunnerContext(root=context.root, output_root=partial / "adapter", config_path=context.config_path, device=context.device)
            adapter = adapter_factory(item.model, item_context)
            prediction = adapter.predict_sample(manifest, item.sample_key, item.rendered_views)
            if prediction.invalid_prediction:
                reason_code = "INVALID_PREDICTION"
        except Exception as exc:
            reason_code = "ADAPTER_EXCEPTION"
            exception_payload = {
                "schema_version": "georeliab-adapter-exception-v1",
                "reason_code": reason_code,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
            _atomic_json(partial / "adapter_exception.json", exception_payload)
            prediction = _empty_invalid_prediction(manifest, item.sample_key, partial, f"{type(exc).__name__}: {exc}")
        audit = audit_factory(manifest, prediction, partial)
        write_json_artifact(partial / "run_manifest.json", manifest)
        write_json_artifact(partial / "prediction_artifact.json", prediction)
        write_json_artifact(partial / "audit_record.json", audit)
        _atomic_json(partial / "stage_item.json", _stage_item_payload_for_context(item, fingerprint))
        validate_artifact_bundle(manifest, prediction, audit)
        _rewrite_bundle_uri_payloads(bundle_dir, partial, work_dir=partial)
        final_manifest = read_json_artifact(partial / "run_manifest.json", RunManifest)
        final_prediction = read_json_artifact(partial / "prediction_artifact.json", PredictionArtifact)
        final_audit = read_json_artifact(partial / "audit_record.json", AuditRecord)
        _write_scene_summary(partial, final_manifest, final_prediction, final_audit, partial / "audit_record.json")
        if bundle_dir.exists():
            raise RunnerError("provenance-conflict: destination appeared before atomic commit")
        partial.replace(bundle_dir)
        load_completed_bundle(bundle_dir)
        runtime = time.perf_counter() - start
        result = RunResult(item.identity, "completed", bundle_dir, prediction.invalid_prediction, reason_code, runtime, prediction.peak_memory_mb)
        _append_ledger(context.output_root, item.stage, {**asdict(result), "item_identity": item.identity, "timestamp": _utc_now(), "attempt": _next_ledger_attempt(context.output_root, item.stage, item.identity), "retried": retried, "artifact_digests": _artifact_digest_map(bundle_dir, ("run_manifest.json", "prediction_artifact.json", "audit_record.json", "stage_item.json", "scene_summary.json")), "artifact_bytes": _dir_size(bundle_dir)})
        return result
    finally:
        _release_claim(claim)


def execute_zero_update_item(
    context: RunnerContext,
    item: ScheduleItem,
    *,
    adapter_factory: AdapterFactory = default_adapter_factory,
    stage_fingerprint: Mapping[str, Any],
) -> RunResult:
    fingerprint = dict(stage_fingerprint)
    bundle_dir = _zero_update_bundle_dir(context.output_root, item)
    if bundle_dir.exists():
        payload = _load_json(bundle_dir / "stage_item.json")
        artifact = bundle_dir / "subset_prediction.npz"
        if payload.get("identity") == item.identity and payload.get("stage_fingerprint") == fingerprint.get("stage_fingerprint") and artifact.exists():
            _parent_dir, parent_manifest, parent_prediction, _parent_audit = _parent_bundle_for_zero_update(context, item)
            zero_result_path = bundle_dir / "zero_update_result.json"
            zero_result = _load_json(zero_result_path)
            if (
                payload.get("subset_prediction_sha256") != sha256_file(artifact)
                or payload.get("zero_update_result_sha256") != sha256_file(zero_result_path)
            ):
                raise RunnerError("provenance-conflict: zero-update bundle digest mismatch")
            invalid_existing = bool(zero_result.get("invalid_prediction"))
            if not invalid_existing:
                _validate_zero_subset_npz(artifact, item, parent_manifest=parent_manifest, parent_prediction=parent_prediction)
            if zero_result.get("subset_prediction_sha256") != sha256_file(artifact):
                raise RunnerError("provenance-conflict: zero-update artifact digest mismatch")
            reason_code = "INVALID_PREDICTION" if invalid_existing else "EXISTING_VALID_ARTIFACT"
            result = RunResult(item.identity, "skipped", bundle_dir, invalid_existing, reason_code, 0.0, 0.0)
            _append_ledger(context.output_root, item.stage, {**asdict(result), "item_identity": item.identity, "timestamp": _utc_now(), "attempt": _next_ledger_attempt(context.output_root, item.stage, item.identity), "artifact_digests": _artifact_digest_map(bundle_dir, ("subset_prediction.npz", "zero_update_result.json", "stage_item.json")), "artifact_bytes": _dir_size(bundle_dir)})
            return result
        raise RunnerError("provenance-conflict: existing zero-update artifact does not match stage fingerprint")
    _parent_bundle_for_zero_update(context, item)
    claim = claim_item(context.output_root / "stage" / item.stage / "claims", item)
    if not claim.acquired:
        raise RunnerError(f"zero-update item is already claimed: {item.identity}")
    partial = bundle_dir.with_name(bundle_dir.name + ".partial")
    start = time.perf_counter()
    try:
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True, exist_ok=True)
        subset_views = _subset_views(item)
        subset_item = ScheduleItem(item.stage, item.model, item.sample_key, item.scene_id, item.condition, item.severity, subset_views, item.identity, item.subset, item.parent_identity)
        manifest = make_manifest(subset_item, context.root, _model_specs(context.config_path), device=context.device)
        reason_code = "OK"
        invalid = False
        peak_memory = 0.0
        ordinal_ids = np.asarray([index for index in range(8) if item.subset is None or index not in set(item.subset)], dtype=np.int64)
        try:
            item_context = RunnerContext(root=context.root, output_root=partial / "adapter", config_path=context.config_path, device=context.device)
            adapter = adapter_factory(item.model, item_context)
            prediction = adapter.predict_sample(manifest, item.sample_key, subset_views)
            geometry = _load_npz_uri(prediction.geometry_prediction_uri)
            valid_payload = _load_npz_uri(prediction.valid_mask_uri)
            points = np.asarray(geometry["points_world"], dtype=np.float64)
            valid = np.asarray(valid_payload["valid_mask"], dtype=bool)
            if valid.shape != (len(points),):
                raise RunnerError("zero-update valid_mask shape does not match points")
            keep = valid & np.isfinite(points).all(axis=1)
            points = points[keep]
            cameras = np.asarray(geometry["camera_c2w"], dtype=np.float64)
            if cameras.ndim == 3 and cameras.shape[1:] == (4, 4) and len(cameras) >= len(ordinal_ids):
                camera_centers = cameras[:len(ordinal_ids), :3, 3]
            else:
                camera_centers = np.empty((0, 3), dtype=np.float64)
            peak_memory = float(prediction.peak_memory_mb)
            if prediction.invalid_prediction:
                invalid = True
                reason_code = "INVALID_PREDICTION"
                points = np.empty((0, 3), dtype=np.float64)
            elif len(points) == 0:
                invalid = True
                reason_code = "EMPTY_SUBSET_PREDICTION"
        except Exception as exc:
            reason_code = "ADAPTER_EXCEPTION"
            invalid = True
            _atomic_json(partial / "adapter_exception.json", {"schema_version": "georeliab-adapter-exception-v1", "reason_code": reason_code, "exception_type": type(exc).__name__, "message": str(exc)})
            points = np.empty((0, 3), dtype=np.float64)
            camera_centers = np.empty((0, 3), dtype=np.float64)
        _parent_dir, parent_manifest, parent_prediction, _parent_audit = _parent_bundle_for_zero_update(context, item)
        subset_path = partial / "subset_prediction.npz"
        np.savez(
            subset_path,
            points=points,
            camera_centers=camera_centers,
            view_ids=ordinal_ids,
            parent_model=np.asarray(parent_manifest.model),
            parent_sample_key=np.asarray(parent_prediction.sample_key),
            parent_project_commit=np.asarray(parent_manifest.provenance.project_commit if parent_manifest.provenance else ""),
            parent_run_id=np.asarray(parent_manifest.run_id),
        )
        if not invalid:
            _validate_zero_subset_npz(subset_path, item, parent_manifest=parent_manifest, parent_prediction=parent_prediction)
        zero_result_path = partial / "zero_update_result.json"
        _atomic_json(zero_result_path, {"schema_version": "zero-update-result-v1", "identity": item.identity, "invalid_prediction": invalid, "reason_code": reason_code, "subset_prediction_sha256": sha256_file(subset_path)})
        stage_payload = _stage_item_payload_for_context(item, fingerprint)
        stage_payload.update(
            {
                "subset_prediction_sha256": sha256_file(subset_path),
                "zero_update_result_sha256": sha256_file(zero_result_path),
            }
        )
        _atomic_json(partial / "stage_item.json", stage_payload)
        _rewrite_bundle_uri_payloads(bundle_dir, partial, work_dir=partial)
        if bundle_dir.exists():
            raise RunnerError("provenance-conflict: zero-update destination appeared before atomic commit")
        partial.replace(bundle_dir)
        runtime = time.perf_counter() - start
        result = RunResult(item.identity, "completed", bundle_dir, invalid, reason_code, runtime, peak_memory)
        _append_ledger(context.output_root, item.stage, {**asdict(result), "item_identity": item.identity, "timestamp": _utc_now(), "attempt": _next_ledger_attempt(context.output_root, item.stage, item.identity), "artifact_digests": _artifact_digest_map(bundle_dir, ("subset_prediction.npz", "zero_update_result.json", "stage_item.json")), "artifact_bytes": _dir_size(bundle_dir)})
        return result
    finally:
        _release_claim(claim)


def ensure_stage_freeze(context: RunnerContext, stage: str, *, dry_run: bool) -> dict[str, Any]:
    if stage != "test":
        return {"stage": stage, "frozen": False, "dry_run": dry_run}
    path = context.output_root / "stage" / "test" / "stage_freeze.json"
    try:
        payload = {**_stage_fingerprint(context, "test", full_schedule(context.root, "test")), "schema_version": "georeliab-test-stage-freeze-v2"}
    except RunnerError as exc:
        if path.exists():
            raise RunnerError(f"test stage parameters changed after freeze: {exc}") from exc
        raise
    if payload["schedule_count"] != 400:
        raise RunnerError("refusing to freeze a shrunk P3 test grid")
    if path.exists():
        existing = _load_json(path)
        if existing != payload:
            raise RunnerError("test stage parameters changed after freeze")
        return existing
    if not dry_run:
        _atomic_json(path, payload)
    return payload


def check_existing_stage_freeze(context: RunnerContext, stage: str, fingerprint: Mapping[str, Any]) -> None:
    if stage not in {"test", "zero-update"}:
        return
    path = context.output_root / "stage" / "test" / "stage_freeze.json"
    if not path.exists():
        return
    existing = _load_json(path)
    if stage == "test" and existing.get("stage_fingerprint") != fingerprint.get("stage_fingerprint"):
        raise RunnerError("test stage parameters changed after freeze")


def _stage_evidence_top_paths(context: RunnerContext) -> dict[str, Path]:
    candidates = {
        "split_view_manifest": context.root / "manifests" / "split_view_manifest.json",
        "corruption_calibration_qa": context.root / "manifests" / "corruption_calibration_qa.json",
        "test_render_lock": context.root / "manifests" / "test_render_lock.json",
        "tartanair_native_fog_sanity": context.root / "evidence" / "tartanair_native_fog_sanity.json",
        "test_prepared_input": context.root / "prepared" / "render_inputs_test.json",
        "tartanair_prepared_input": context.root / "prepared" / "tartanair_p000_pairs.json",
        "test_render_index": context.root / "manifests" / "test_render_index.json",
    }
    missing = [str(path) for path in candidates.values() if not path.exists()]
    if missing:
        raise RunnerError("cannot build P3 stage evidence; missing top-level artifacts: " + ", ".join(missing))
    return candidates


def build_stage_evidence_manifest(context: RunnerContext, *, p3_only: bool = False) -> Path | None:
    bundle_root = context.output_root / "stage" / "test" / "bundles"
    if not bundle_root.exists():
        return None
    freeze_path = context.output_root / "stage" / "test" / "stage_freeze.json"
    if not freeze_path.exists():
        return None
    current_freeze = _load_json(freeze_path)
    entries: list[dict[str, str]] = []
    for manifest_path in sorted(bundle_root.glob("*/*/run_manifest.json")):
        bundle_dir = manifest_path.parent
        paths = {
            "manifest": manifest_path,
            "prediction": bundle_dir / "prediction_artifact.json",
            "audit": bundle_dir / "audit_record.json",
            "scene_summary": bundle_dir / "scene_summary.json",
        }
        if not all(path.exists() for path in paths.values()):
            continue
        manifest, prediction, audit = load_completed_bundle(bundle_dir)
        if manifest.mode is not RunMode.REAL or manifest.split != "test" or SampleKey.parse(prediction.sample_key).split != "test":
            continue
        stage_item = _load_json(bundle_dir / "stage_item.json")
        freeze = stage_item.get("freeze")
        if not isinstance(freeze, dict) or stage_item.get("stage_fingerprint") != freeze.get("stage_fingerprint"):
            continue
        if stage_item.get("stage_fingerprint") != current_freeze.get("stage_fingerprint") or freeze != current_freeze:
            continue
        summary = _load_json(paths["scene_summary"])
        if summary.get("audit_sha256") != sha256_file(paths["audit"]):
            continue
        entries.append({f"{name}_path": str(path) for name, path in paths.items()} | {f"{name}_sha256": sha256_file(path) for name, path in paths.items()})
    if len(entries) != 400:
        return None
    top = _stage_evidence_top_paths(context)
    corruption_qa = _load_json(top["corruption_calibration_qa"])
    materialization = context.root / "manifests" / "frozen_materialization.json"
    payload: dict[str, Any] = {
        "schema_version": "stage-evidence-v1",
        "parameter_manifest_sha256": str(corruption_qa.get("parameter_manifest_sha256", sha256_file(context.root / "manifests" / "corruption_calibration.json"))),
        "materialization_sha256": sha256_file(materialization),
        "bundle_index": entries,
        "downstream_index": [] if p3_only else _evidence_index(context.output_root / "stage" / "test" / "downstream", "*.json"),
        "zero_update_index": [] if p3_only else _evidence_index(context.output_root / "stage" / "test" / "zero_update", "*.json"),
    }
    for name, path in top.items():
        payload[f"{name}_path"] = str(path)
        payload[f"{name}_sha256"] = sha256_file(path)
    output_name = "stage_evidence_p3.json" if p3_only else "stage_evidence.json"
    output = context.output_root / "stage" / "test" / output_name
    if p3_only and output.exists():
        existing = _load_json(output)
        if existing != payload:
            raise RunnerError("immutable P3 stage_evidence_p3.json would change")
        return output
    _atomic_json(output, payload)
    return output


def _evidence_index(root: Path, pattern: str) -> list[dict[str, str]]:
    if not root.exists():
        return []
    return [{"evidence_path": str(path), "evidence_sha256": sha256_file(path)} for path in sorted(root.glob(pattern))]


def _severity2_source_keys(model: str, condition: str) -> list[str]:
    return [f"dtu/test/scan{scene}/views-0001/{condition}/2/0" for scene in TEST_SCENES]


def build_downstream_evidence(context: RunnerContext) -> list[Path]:
    output_root = context.output_root / "stage" / "test" / "downstream"
    written: list[Path] = []
    for model in MODELS:
        for condition in ("fog", "low-light-noise", "defocus"):
            source_keys = _severity2_source_keys(model, condition)
            found = []
            for manifest_path in (context.output_root / "stage" / "test" / "bundles" / model.lower()).glob("*/run_manifest.json"):
                prediction = read_json_artifact(manifest_path.with_name("prediction_artifact.json"), PredictionArtifact)
                if prediction.sample_key in source_keys:
                    load_completed_bundle(manifest_path.parent)
                    found.append(prediction.sample_key)
            if sorted(found) != sorted(source_keys):
                raise RunnerError(f"downstream evidence requires complete P3 severity-2 bundles for {model}/{condition}")
            path = output_root / f"{model.lower()}_{condition.replace('-', '_')}.json"
            _atomic_json(path, {"schema_version": "downstream-harm-v1", "model": model, "condition": f"{condition}-s2", "source_sample_keys": source_keys, "coverage": [0.9, 0.7, 0.5, 0.3], "random_mask_count": 100, "n_resamples": 10000})
            written.append(path)
    return written


def _zero_update_terminal_failure_path(context: RunnerContext) -> Path:
    return context.output_root / "stage" / "zero-update" / "terminal_failure.json"


def _load_zero_update_terminal_failure(
    context: RunnerContext,
    *,
    current_freeze: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = _zero_update_terminal_failure_path(context)
    if not path.exists():
        return None
    payload = _load_json(path)
    if (
        payload.get("schema_version") != "zero-update-terminal-failure-v1"
        or payload.get("status") != "FAIL"
        or payload.get("reason_code") != "P5_INVALID_SUBSET_PREDICTION"
        or payload.get("test_stage_fingerprint") != current_freeze.get("stage_fingerprint")
    ):
        raise RunnerError("zero-update terminal failure record is invalid or belongs to another freeze")
    freeze_path = context.output_root / "stage" / "test" / "stage_freeze.json"
    if payload.get("stage_freeze_sha256") != sha256_file(freeze_path):
        raise RunnerError("zero-update terminal failure record is not bound to the P3 freeze digest")
    invalid = payload.get("invalid_subset")
    if not isinstance(invalid, dict):
        raise RunnerError("zero-update terminal failure record lacks invalid subset evidence")
    for prefix in ("stage_item", "zero_update_result", "subset_prediction"):
        artifact_path = Path(str(invalid.get(f"{prefix}_path", "")))
        expected_sha = invalid.get(f"{prefix}_sha256")
        if not artifact_path.is_file() or expected_sha != sha256_file(artifact_path):
            raise RunnerError("zero-update terminal failure evidence digest mismatch")
    zero_result = _load_json(Path(str(invalid["zero_update_result_path"])))
    if zero_result.get("invalid_prediction") is not True or zero_result.get("identity") != invalid.get("identity"):
        raise RunnerError("zero-update terminal failure does not reference an invalid subset")
    return payload


def _write_zero_update_terminal_failure(
    context: RunnerContext,
    *,
    current_freeze: Mapping[str, Any],
    item: ScheduleItem,
    bundle: Path,
    stage_item_path: Path,
    result_path: Path,
    artifact: Path,
    reason_code: str,
) -> Path:
    output = _zero_update_terminal_failure_path(context)
    if output.exists():
        _load_zero_update_terminal_failure(context, current_freeze=current_freeze)
        return output
    freeze_path = context.output_root / "stage" / "test" / "stage_freeze.json"
    payload = {
        "schema_version": "zero-update-terminal-failure-v1",
        "status": "FAIL",
        "reason_code": "P5_INVALID_SUBSET_PREDICTION",
        "test_stage_fingerprint": current_freeze.get("stage_fingerprint"),
        "stage_freeze_sha256": sha256_file(freeze_path),
        "invalid_subset": {
            "model": item.model,
            "condition": item.condition,
            "sample_key": str(item.sample_key),
            "identity": item.identity,
            "subset": list(item.subset) if item.subset is not None else None,
            "adapter_reason_code": reason_code,
            "bundle_path": str(bundle),
            "stage_item_path": str(stage_item_path),
            "stage_item_sha256": sha256_file(stage_item_path),
            "zero_update_result_path": str(result_path),
            "zero_update_result_sha256": sha256_file(result_path),
            "subset_prediction_path": str(artifact),
            "subset_prediction_sha256": sha256_file(artifact),
        },
    }
    _atomic_json(output, payload)
    return output


def build_zero_update_evidence(context: RunnerContext) -> list[Path] | None:
    root = context.output_root / "stage" / "zero-update" / "bundles"
    if not root.exists():
        return None
    freeze_path = context.output_root / "stage" / "test" / "stage_freeze.json"
    if not freeze_path.exists():
        return None
    current_freeze = _load_json(freeze_path)
    output_root = context.output_root / "stage" / "test" / "zero_update"
    written: list[Path] = []
    omitted_pairs = [list(pair) for pair in ZERO_UPDATE_SUBSETS]
    canonical_gate = {"schema_version": "native-phenomenon-gate-v1", "status": "PASS"}
    expected_items = build_zero_update_schedule(context.root, canonical_gate, model="all")
    indexed: dict[tuple[str, str, str, tuple[int, int]], dict[str, str]] = {}
    missing = False
    for item in expected_items:
        bundle = _zero_update_bundle_dir(context.output_root, item)
        if not bundle.exists():
            missing = True
            continue
        stage_item_path = bundle / "stage_item.json"
        result_path = bundle / "zero_update_result.json"
        artifact = bundle / "subset_prediction.npz"
        if not stage_item_path.exists() or not result_path.exists() or not artifact.exists():
            raise RunnerError("zero-update committed bundle is incomplete")
        stage_item = _load_json(stage_item_path)
        expected_linkage = {
            "identity": item.identity,
            "stage": "zero-update",
            "model": item.model,
            "sample_key": str(item.sample_key),
            "condition": item.condition,
            "severity": item.severity,
            "subset": list(item.subset) if item.subset is not None else None,
            "parent_identity": item.parent_identity,
            "stage_fingerprint": current_freeze.get("stage_fingerprint"),
        }
        if any(stage_item.get(key) != value for key, value in expected_linkage.items()):
            raise RunnerError("zero-update committed bundle linkage conflicts with the frozen schedule")
        if stage_item.get("freeze") != current_freeze:
            raise RunnerError("zero-update committed bundle is not bound to the exact P3 freeze")
        zero_result = _load_json(result_path)
        artifact_sha = sha256_file(artifact)
        result_sha = sha256_file(result_path)
        if (
            zero_result.get("schema_version") != "zero-update-result-v1"
            or zero_result.get("identity") != item.identity
            or zero_result.get("subset_prediction_sha256") != artifact_sha
            or stage_item.get("subset_prediction_sha256") != artifact_sha
            or stage_item.get("zero_update_result_sha256") != result_sha
        ):
            raise RunnerError("zero-update evidence artifact digest or identity mismatch")
        _parent_dir, parent_manifest, parent_prediction, _parent_audit = _parent_bundle_for_zero_update(context, item)
        if zero_result.get("invalid_prediction") is True:
            terminal = _write_zero_update_terminal_failure(
                context,
                current_freeze=current_freeze,
                item=item,
                bundle=bundle,
                stage_item_path=stage_item_path,
                result_path=result_path,
                artifact=artifact,
                reason_code=str(zero_result.get("reason_code", "INVALID_PREDICTION")),
            )
            raise ZeroUpdateTerminalFailure(terminal)
        _validate_zero_subset_npz(
            artifact,
            item,
            parent_manifest=parent_manifest,
            parent_prediction=parent_prediction,
        )
        key = (item.model, item.condition, item.sample_key.scene, tuple(item.subset or ()))
        if key in indexed:
            raise RunnerError("duplicate zero-update subset artifact")
        indexed[key] = {"artifact_path": str(artifact), "artifact_sha256": artifact_sha}
    if missing:
        return None
    for model in MODELS:
        for condition in ("fog", "low-light-noise", "defocus"):
            source_keys = _severity2_source_keys(model, condition)
            subset_artifacts: dict[str, list[dict[str, str]]] = {f"scan{scene}": [] for scene in TEST_SCENES}
            for scene in (f"scan{scene_id}" for scene_id in TEST_SCENES):
                for subset in ZERO_UPDATE_SUBSETS:
                    artifact = indexed.get((model, condition, scene, subset))
                    if artifact is not None:
                        subset_artifacts[scene].append(artifact)
            if any(len(rows) != 4 for rows in subset_artifacts.values()):
                raise RunnerError("zero-update evidence index is incomplete after complete schedule validation")
            path = output_root / f"{model.lower()}_{condition.replace('-', '_')}.json"
            _atomic_json(path, {"schema_version": "zero-update-v1", "model": model, "condition": f"{condition}-s2", "n_resamples": 10000, "omitted_view_pairs": omitted_pairs, "subset_artifacts": subset_artifacts, "source_sample_keys": source_keys})
            written.append(path)
    return written


def check_budget(
    context: RunnerContext,
    stage: str,
    *,
    next_stage_gpu_hours: float,
    next_stage_bytes: int,
    completed_gpu_hours: float,
    completed_bytes: int,
) -> dict[str, Any]:
    remaining_gpu = GPU_HOUR_LIMIT - float(completed_gpu_hours)
    remaining_bytes = STORAGE_BYTE_LIMIT - int(completed_bytes)
    if next_stage_gpu_hours > remaining_gpu or next_stage_bytes > remaining_bytes:
        return {
            "status": "BLOCKED_RESOURCE_BUDGET",
            "stage": stage,
            "remaining_gpu_hours": remaining_gpu,
            "remaining_bytes": remaining_bytes,
            "estimated_gpu_hours": next_stage_gpu_hours,
            "estimated_bytes": next_stage_bytes,
        }
    return {"status": "OK", "stage": stage, "remaining_gpu_hours": remaining_gpu, "remaining_bytes": remaining_bytes}


def _observed_usage(output_root: Path) -> tuple[float, int]:
    gpu = 0.0
    ledger_roots = [output_root / "stage"] if (output_root / "stage").exists() else []
    ledger_roots.extend(output_root.glob("preflight-real/*/stage"))
    for root in ledger_roots:
        for ledger in root.glob("*/ledger.jsonl"):
            for line in ledger.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                gpu += float(row.get("runtime_seconds", 0.0)) / 3600.0
    bytes_used = 0
    if output_root.exists():
        for path in output_root.rglob("*"):
            if path.is_file():
                try:
                    bytes_used += path.stat().st_size
                except OSError:
                    pass
    return gpu, bytes_used


def _budget_observation_rows(context: RunnerContext, stage: str) -> list[dict[str, Any]]:
    if stage == "smoke":
        allowed = {"preflight"}
    elif stage == "test":
        allowed = {"smoke", "preflight"}
    elif stage == "zero-update":
        allowed = {"test", "smoke", "preflight"}
    else:
        allowed = {stage}
    rows: dict[str, dict[str, Any]] = {}
    stage_root = context.output_root / "stage"
    roots = [stage_root] if stage_root.exists() else []
    roots.extend(context.output_root.glob("preflight-real/*/stage"))
    for root in roots:
        for ledger in root.glob("*/ledger.jsonl"):
            source_stage = ledger.parent.name
            if source_stage not in allowed:
                continue
            for line in ledger.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("state") in {"completed", "skipped"}:
                    key = f"{source_stage}:{row.get('item_identity', row.get('identity', len(rows)))}"
                    rows[key] = row
    return list(rows.values())


def estimate_stage_budget(context: RunnerContext, stage: str, items: Sequence[ScheduleItem], override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    completed_gpu, completed_bytes = _observed_usage(context.output_root)
    if override is not None:
        next_gpu = float(override.get("next_stage_gpu_hours", 0.0))
        next_bytes = int(override.get("next_stage_bytes", 0))
    else:
        observation_rows = _budget_observation_rows(context, stage)
        if not observation_rows and stage != "preflight":
            return {
                "status": "BLOCKED_RESOURCE_BUDGET",
                "stage": stage,
                "reason_code": "BUDGET_ESTIMATE_UNAVAILABLE",
                "scheduled": len(items),
                "completed_gpu_hours": completed_gpu,
                "completed_bytes": completed_bytes,
            }
        completed = max(1, len(observation_rows))
        observed_stage_gpu = sum(float(row.get("runtime_seconds", 0.0)) for row in observation_rows) / 3600.0
        next_gpu = observed_stage_gpu / completed * len(items) if observation_rows else 0.0
        stage_dir = context.output_root / "stage" / stage
        observed_stage_bytes = sum(path.stat().st_size for path in stage_dir.rglob("*") if path.is_file()) if stage_dir.exists() else 0
        if observed_stage_bytes == 0 and observation_rows:
            observed_stage_bytes = int(sum(int(row.get("artifact_bytes", 0)) for row in observation_rows))
        next_bytes = int(observed_stage_bytes / completed * len(items)) if observation_rows else 0
    return check_budget(context, stage, next_stage_gpu_hours=next_gpu, next_stage_bytes=next_bytes, completed_gpu_hours=completed_gpu, completed_bytes=completed_bytes)


_P0_STATE_SPECS = (
    ("p0_download.json", "download", None),
    ("p0_verify.json", "verify", None),
    ("p0_index.json", "index", None),
    ("p0_manifests.json", "manifests", None),
    ("p0_prepared.json", "prepared", None),
    ("p0_calibration.json", "calibration", None),
    ("p0_render_smoke.json", "rendering", "smoke"),
    ("p0_render_test.json", "rendering", "test"),
    ("p0_sanity.json", "sanity", None),
)


def _pending_stage(reason_code: str, **details: Any) -> dict[str, Any]:
    return {
        "status": "BLOCKED_PENDING_EVIDENCE",
        "reason_code": reason_code,
        **details,
    }


def p0_completion_status(root: Path) -> dict[str, Any]:
    """Verify that every non-scientific P0 operation finished and is linked."""

    artifacts = root / "artifacts"
    states: dict[str, dict[str, Any]] = {}
    for filename, operation, stage in _P0_STATE_SPECS:
        path = artifacts / filename
        if not path.is_file():
            return _pending_stage("P0_STATE_MISSING", state_path=str(path))
        try:
            payload = _load_json(path)
        except RunnerError:
            return _pending_stage("P0_STATE_INVALID", state_path=str(path))
        expected = {
            "schema_version": "preparation-state-v5",
            "operation": operation,
            "stage": stage,
            "dry_run": False,
            "scientific_ready": False,
            "state_transition": f"{operation}:completed",
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            return _pending_stage("P0_STATE_INVALID", state_path=str(path))
        states[filename] = payload

    if states["p0_verify.json"].get("resources_ready") is not True:
        return _pending_stage(
            "P0_STATE_INVALID",
            state_path=str(artifacts / "p0_verify.json"),
        )

    calibration_path = root / "manifests" / "corruption_calibration.json"
    qa_path = root / "manifests" / "corruption_calibration_qa.json"
    sanity_path = root / "evidence" / "tartanair_native_fog_sanity.json"
    missing = [
        str(path)
        for path in (calibration_path, qa_path, sanity_path)
        if not path.is_file()
    ]
    if missing:
        return _pending_stage("P0_ARTIFACT_MISSING", missing=missing)
    try:
        qa = _load_json(qa_path)
        sanity = _load_json(sanity_path)
        calibration_sha = sha256_file(calibration_path)
        qa_sha = sha256_file(qa_path)
    except RunnerError:
        return _pending_stage("P0_ARTIFACT_INVALID")
    if (
        qa.get("passed") is not True
        or qa.get("parameter_manifest_sha256") != calibration_sha
        or sanity.get("passed") is not True
        or sanity.get("reason_code") != "TARTANAIR_NATIVE_FOG_SANITY"
        or sanity.get("evaluated_frames") != 100
        or not isinstance(sanity.get("negative_frames"), int)
        or sanity.get("negative_frames", 0) < 80
        or sanity.get("calibration_qa_sha256") != qa_sha
    ):
        return _pending_stage("P0_ARTIFACT_INVALID")

    calibration_state = states["p0_calibration.json"]
    if (
        calibration_state.get("qa_passed") is not True
        or calibration_state.get("parameter_manifest_sha256") != calibration_sha
    ):
        return _pending_stage(
            "P0_STATE_INVALID",
            state_path=str(artifacts / "p0_calibration.json"),
        )
    for filename, expected_stage, expected_split, expected_count in (
        ("p0_render_smoke.json", "smoke", "dev", 800),
        ("p0_render_test.json", "test", "test", 1600),
    ):
        state = states[filename]
        if (
            state.get("stage") != expected_stage
            or state.get("split") != expected_split
            or state.get("rendered_count") != expected_count
            or state.get("parameter_manifest_sha256") != calibration_sha
        ):
            return _pending_stage(
                "P0_STATE_INVALID",
                state_path=str(artifacts / filename),
            )
    sanity_state = states["p0_sanity.json"]
    if (
        sanity_state.get("passed") is not True
        or sanity_state.get("reason_code") != "TARTANAIR_NATIVE_FOG_SANITY"
        or sanity_state.get("calibration_qa_sha256") != qa_sha
    ):
        return _pending_stage(
            "P0_STATE_INVALID",
            state_path=str(artifacts / "p0_sanity.json"),
        )
    return {
        "status": "PASS",
        "reason_code": "P0_COMPLETE",
        "state_count": len(states),
        "calibration_sha256": calibration_sha,
        "calibration_qa_sha256": qa_sha,
        "tartanair_sanity_sha256": sha256_file(sanity_path),
    }


def p1_completion_status(
    root: Path,
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Verify the canonical dual-model, two-repeat P1 summary and artifacts."""

    summary_path = root / "artifacts" / "p1_preflight.json"
    if not summary_path.is_file():
        return _pending_stage("P1_SUMMARY_MISSING", summary_path=str(summary_path))
    try:
        summary = _load_json(summary_path)
    except RunnerError:
        return _pending_stage("P1_SUMMARY_INVALID", summary_path=str(summary_path))
    if summary.get("status") != "OK" or summary.get("stage") != "preflight":
        return _pending_stage("P1_SUMMARY_NOT_PASSED", summary_path=str(summary_path))
    workers = summary.get("workers")
    if not isinstance(workers, list):
        return _pending_stage("P1_SUMMARY_INVALID", summary_path=str(summary_path))
    by_model = {
        worker.get("model_worker"): worker
        for worker in workers
        if isinstance(worker, dict)
    }
    if set(by_model) != {"vggt", "mast3r"} or len(workers) != 2:
        return _pending_stage("P1_MODEL_GRID_INCOMPLETE")
    try:
        schedule = full_schedule(root, "preflight")
        expected_fingerprint = _stage_fingerprint(
            RunnerContext(
                root=root,
                output_root=root,
                config_path=config_path,
                device="cuda:0",
            ),
            "preflight",
            schedule,
        )["stage_fingerprint"]
    except (RunnerError, OSError, ValueError):
        return _pending_stage("P1_FINGERPRINT_UNVERIFIABLE")
    for model, worker in by_model.items():
        repeatability = worker.get("repeatability")
        if (
            worker.get("status") != "OK"
            or not isinstance(repeatability, dict)
            or repeatability.get("passed") is not True
            or repeatability.get("reason_code") != "OK"
        ):
            return _pending_stage(
                "P1_REPEATABILITY_NOT_PASSED",
                model=model,
            )
        repeats = worker.get("repeats")
        if (
            not isinstance(repeats, list)
            or len(repeats) != 2
            or any(
                not isinstance(repeat, dict)
                or repeat.get("status") != "OK"
                or repeat.get("stage_fingerprint") != expected_fingerprint
                for repeat in repeats
            )
        ):
            return _pending_stage("P1_REPEAT_GRID_INVALID", model=model)
    for label in ("repeat-a", "repeat-b"):
        counts = stage_progress_counts(
            root / "preflight-real" / label,
            "preflight",
            schedule,
        )
        if (
            counts.get("scheduled") != 8
            or counts.get("completed") != 8
            or counts.get("missing") != 0
            or counts.get("invalid") != 0
        ):
            return _pending_stage(
                "P1_SCHEDULE_INCOMPLETE",
                repeat=label,
                schedule_counts=counts,
            )
    return {
        "status": "PASS",
        "reason_code": "P1_COMPLETE",
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "stage_fingerprint": expected_fingerprint,
    }


def _require_stage_decision(decision: Mapping[str, Any], stage: str) -> None:
    if decision.get("status") != "PASS":
        raise RunnerError(
            f"{stage}_LOCKED: "
            + json.dumps(dict(decision), sort_keys=True, default=str)
        )


def _verify_stage_readiness(context: RunnerContext, stage: str, model: str) -> None:
    _verify_output_root_policy(context)
    _verify_clean_source_tree()
    if stage == "preflight":
        _require_stage_decision(p0_completion_status(context.root), "P1")
    elif stage == "smoke":
        _require_stage_decision(
            p1_completion_status(context.root, config_path=context.config_path),
            "P2",
        )
    elif stage == "test":
        smoke_counts = stage_progress_counts(
            context.output_root,
            "smoke",
            full_schedule(context.root, "smoke"),
        )
        if (
            smoke_counts.get("scheduled") != 200
            or smoke_counts.get("completed") != 200
            or smoke_counts.get("missing") != 0
        ):
            raise RunnerError(
                "P3_LOCKED: "
                + json.dumps(smoke_counts, sort_keys=True, default=str)
            )
    required = [
        context.root / "manifests" / "split_view_manifest.json",
        context.root / "manifests" / "corruption_calibration.json",
        context.root / "manifests" / "corruption_calibration_qa.json",
        context.root / "evidence" / "tartanair_native_fog_sanity.json",
        context.root / "manifests" / "frozen_materialization.json",
    ]
    prepared_stage = "test" if stage in {"test", "zero-update"} else "smoke"
    required.append(context.root / "prepared" / f"render_inputs_{prepared_stage}.json")
    if stage in {"test", "zero-update"}:
        required.extend(
            (
                context.root / "manifests" / "test_render_lock.json",
                context.root / "manifests" / "test_render_index.json",
                context.root / "prepared" / "tartanair_p000_pairs.json",
            )
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RunnerError("readiness artifacts missing before real dispatch: " + ", ".join(missing))
    try:
        from .materialization import PreparationError, verify_materialization_manifest
        from .preparation_round2 import load_prepared_batch

        split_path = context.root / "manifests" / "split_view_manifest.json"
        materialization_path = context.root / "manifests" / "frozen_materialization.json"
        verify_materialization_manifest(materialization_path, split_manifest_path=split_path)
        load_prepared_batch(
            context.root / "prepared" / f"render_inputs_{prepared_stage}.json",
            expected_stage=prepared_stage,
        )
    except PreparationError as exc:
        raise RunnerError(f"Task2 prepared/materialization verification failed before real dispatch: {exc}") from exc
    _verify_frozen_runtimes(context, model)


def _parse_shard(value: str) -> tuple[int, int]:
    try:
        left, right = value.split("/", 1)
        return int(left), int(right)
    except (ValueError, AttributeError) as exc:
        raise RunnerError("--shard must use INDEX/TOTAL") from exc


def run_stage(
    context: RunnerContext,
    *,
    stage: str,
    model: str,
    shard: str,
    dry_run: bool,
    adapter_factory: AdapterFactory = default_adapter_factory,
    audit_factory: AuditFactory = production_audit_factory,
    budget_override: Mapping[str, Any] | None = None,
    native_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    shard_index, shard_total = _parse_shard(shard)
    if stage == "zero-update":
        canonical_gate = load_native_phenomenon_gate(context)
        if not dry_run:
            if native_gate is not None and dict(native_gate) != canonical_gate:
                native_gate = {"schema_version": "native-phenomenon-gate-v1", "status": "BLOCKED_PENDING_EVIDENCE", "reason_code": "NATIVE_CONFIDENCE_GATE_INJECTION_FORBIDDEN"}
            else:
                native_gate = canonical_gate
        elif native_gate is None:
            native_gate = canonical_gate
        allowed = zero_update_schedule_allowed(native_gate)
        if allowed["status"] != "PASS":
            return {"status": allowed["status"], "reason_code": allowed.get("reason_code"), "stage": stage, "dry_run": dry_run, "schedule_counts": {"scheduled": 0, "completed": 0, "missing": 0, "invalid": 0}}
        full = build_zero_update_schedule(context.root, native_gate, model="all")
        selected = build_zero_update_schedule(context.root, native_gate, model=model)
    else:
        full = full_schedule(context.root, "preflight" if stage == "preflight" else stage)
        selected = build_schedule(context.root, "preflight" if stage == "preflight" else stage, model=model)
    if stage == "test" and refuse_shrunk_test_grid(full) != "OK":
        raise RunnerError("refusing to shrink frozen P3 test grid")
    if not dry_run:
        _verify_stage_readiness(context, stage, model)
    if stage == "zero-update":
        freeze_path = context.output_root / "stage" / "test" / "stage_freeze.json"
        if not freeze_path.exists():
            raise RunnerError("zero-update requires an existing frozen P3 test stage")
        freeze = _load_json(freeze_path)
        fingerprint = dict(freeze)
    else:
        fingerprint = _stage_fingerprint(context, stage, full)
        check_existing_stage_freeze(context, stage, fingerprint)
        freeze = ensure_stage_freeze(context, "test", dry_run=dry_run) if stage == "test" else {"stage": stage, "frozen": False, "dry_run": dry_run}
        if stage == "test":
            fingerprint = dict(freeze)
    if stage == "zero-update" and not dry_run:
        terminal = _load_zero_update_terminal_failure(
            context,
            current_freeze=fingerprint,
        )
        if terminal is not None:
            terminal_path = _zero_update_terminal_failure_path(context)
            return {
                "status": "FAIL",
                "stage": stage,
                "dry_run": False,
                "reason_code": "P5_INVALID_SUBSET_PREDICTION",
                "schedule_counts": stage_progress_counts(context.output_root, stage, full),
                "terminal_failure_path": str(terminal_path),
                "terminal_failure_sha256": sha256_file(terminal_path),
            }
    budget = estimate_stage_budget(context, stage, full, budget_override)
    if budget["status"] != "OK" and not dry_run:
        return {"status": budget["status"], "stage": stage, "dry_run": dry_run, "budget": budget, "schedule_counts": stage_progress_counts(context.output_root, stage, full)}
    work = shard_schedule(selected, index=shard_index, total=shard_total)
    summary = {"status": "DRY_RUN" if dry_run else "OK", "stage": stage, "dry_run": dry_run, "shard": shard, "schedule_counts": stage_counts(full), "selected_count": len(selected), "dispatch_count": len(work), "budget": budget, "stage_fingerprint": fingerprint.get("stage_fingerprint"), "freeze": freeze}
    if dry_run:
        return summary
    if stage == "zero-update":
        build_downstream_evidence(context)
        results = [execute_zero_update_item(context, item, adapter_factory=adapter_factory, stage_fingerprint=fingerprint) for item in work]
        summary["results"] = [asdict(result) for result in results]
        summary["schedule_counts"] = stage_progress_counts(context.output_root, stage, full)
        summary["ledger"] = read_stage_ledger(context.output_root, stage)["counts"]
        try:
            zero_written = build_zero_update_evidence(context)
        except ZeroUpdateTerminalFailure as exc:
            summary.update(
                {
                    "status": "FAIL",
                    "reason_code": "P5_INVALID_SUBSET_PREDICTION",
                    "terminal_failure_path": str(exc.path),
                    "terminal_failure_sha256": sha256_file(exc.path),
                }
            )
            return summary
        if zero_written is not None:
            summary["zero_update_evidence_count"] = len(zero_written)
            evidence_path = build_stage_evidence_manifest(context, p3_only=True)
            if evidence_path is not None:
                summary["p3_stage_evidence_path"] = str(evidence_path)
                summary["p3_stage_evidence_sha256"] = sha256_file(evidence_path)
                final_path = build_stage_evidence_manifest(context, p3_only=False)
                if final_path is not None:
                    summary["stage_evidence_path"] = str(final_path)
                    summary["stage_evidence_sha256"] = sha256_file(final_path)
        return summary
    if audit_factory is production_audit_factory:
        audit_factory = make_production_audit_factory(context)
    results = [execute_item(context, item, adapter_factory=adapter_factory, audit_factory=audit_factory, stage_fingerprint=fingerprint) for item in work]
    summary["results"] = [asdict(result) for result in results]
    summary["schedule_counts"] = stage_progress_counts(context.output_root, stage, full)
    summary["ledger"] = read_stage_ledger(context.output_root, stage)["counts"]
    if stage == "test":
        evidence_path = build_stage_evidence_manifest(context, p3_only=True)
        if evidence_path is not None:
            summary["p3_stage_evidence_path"] = str(evidence_path)
            summary["p3_stage_evidence_sha256"] = sha256_file(evidence_path)
            final_path = build_stage_evidence_manifest(context, p3_only=False)
            if final_path is not None:
                summary["stage_evidence_path"] = str(final_path)
                summary["stage_evidence_sha256"] = sha256_file(final_path)
    return summary


def _preflight_repeat_snapshot(output_root: Path) -> dict[str, Any]:
    bundle_root = output_root / "stage" / "preflight" / "bundles"
    input_parts: list[str] = []
    errors: list[float] = []
    rhos: list[float] = []
    from .metrics import MetricError, spearman_correlation

    if bundle_root.exists():
        for audit_path in sorted(bundle_root.glob("*/*/audit_record.json")):
            manifest = read_json_artifact(audit_path.with_name("run_manifest.json"), RunManifest)
            audit = read_json_artifact(audit_path, AuditRecord)
            input_parts.append(f"{audit.sample_key}:{manifest.rgb_digest}")
            dense = _load_npz_uri(audit.metadata["dense_audit_uri"])
            gt_error = np.asarray(dense["gt_error"], dtype=np.float64)
            risk = np.asarray(dense["risk"], dtype=np.float64)
            finite_error = gt_error[np.isfinite(gt_error)]
            errors.append(float(np.mean(finite_error)) if finite_error.size else 1.0)
            if len(risk) >= 2 and len(gt_error) == len(risk):
                try:
                    rhos.append(float(spearman_correlation(risk.tolist(), gt_error.tolist())))
                except MetricError:
                    rhos.append(0.0)
            else:
                rhos.append(0.0)
    return {
        "input_digest": _sha_json(input_parts),
        "aggregate_error": float(np.mean(errors)) if errors else 0.0,
        "rho": float(np.mean(rhos)) if rhos else 0.0,
    }


def run_preflight_real(
    context: RunnerContext,
    *,
    dry_run: bool,
    model: str = "all",
    adapter_factory: AdapterFactory = default_adapter_factory,
    audit_factory: AuditFactory = production_audit_factory,
    budget_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if dry_run:
        return run_stage(
            context,
            stage="preflight",
            model=model,
            shard="0/1",
            dry_run=True,
            adapter_factory=adapter_factory,
            audit_factory=audit_factory,
            budget_override=budget_override,
        )
    repeats: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    for label in ("repeat-a", "repeat-b"):
        repeat_context = RunnerContext(
            root=context.root,
            output_root=context.output_root / "preflight-real" / label,
            config_path=context.config_path,
            device=context.device,
        )
        payload = run_stage(
            repeat_context,
            stage="preflight",
            model=model,
            shard="0/1",
            dry_run=False,
            adapter_factory=adapter_factory,
            audit_factory=audit_factory,
            budget_override=budget_override,
        )
        repeats.append(payload)
        if payload.get("status") != "OK":
            return {
                "status": payload.get("status", "PREFLIGHT_REPEAT_FAILED"),
                "stage": "preflight",
                "dry_run": False,
                "schedule_counts": payload.get("schedule_counts", {}),
                "repeats": repeats,
                "reason_code": payload.get("reason_code", payload.get("status", "PREFLIGHT_REPEAT_FAILED")),
            }
        snapshots.append(_preflight_repeat_snapshot(repeat_context.output_root))
    repeatability = evaluate_repeatability(snapshots[0], snapshots[1])
    return {
        "status": "OK" if repeatability["passed"] else "PREFLIGHT_REPEATABILITY_FAILED",
        "stage": "preflight",
        "dry_run": False,
        "schedule_counts": repeats[0].get("schedule_counts", {}),
        "repeats": repeats,
        "repeatability": repeatability,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="georeliab-runner")
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight-real")
    pre.add_argument("--config", type=Path, required=True)
    pre.add_argument("--output-root", type=Path, required=True)
    pre.add_argument("--device", default="cuda:0")
    pre.add_argument("--model", choices=("vggt", "mast3r", "all"), default="all", help=argparse.SUPPRESS)
    pre.add_argument("--summary-json", type=Path, help=argparse.SUPPRESS)
    pre.add_argument("--dry-run", action="store_true")
    run = sub.add_parser("run-georeliab")
    run.add_argument("--stage", choices=("smoke", "test", "zero-update"), required=True)
    run.add_argument("--model", choices=("vggt", "mast3r", "all"), required=True)
    run.add_argument("--device", required=True)
    run.add_argument("--shard", required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--summary-json", type=Path, help=argparse.SUPPRESS)
    run.add_argument("--dry-run", action="store_true")
    return parser


def _worker_python_path(context: RunnerContext, model: str) -> Path:
    payload = _overlay_payload(context)
    runtime = payload.get("runtime", {})
    env_key = "vggt_env" if model == "vggt" else "mast3r_env"
    env_root = Path(str(runtime[env_key]))
    for candidate in (env_root / "bin" / "python", env_root / "Scripts" / "python.exe"):
        if candidate.exists():
            return candidate
    return env_root / "bin" / "python"


def _verify_current_worker_runtime(context: RunnerContext, model: str) -> None:
    expected_python = _worker_python_path(context, model)
    try:
        if Path(sys.executable).resolve() != expected_python.resolve():
            raise RunnerError(f"{model} worker is not running inside frozen environment: {sys.executable} != {expected_python}")
    except OSError as exc:
        raise RunnerError(f"cannot verify {model} worker executable") from exc
    import platform
    import hashlib
    spec = frozen_model_spec("VGGT" if model == "vggt" else "MASt3R", context.config_path)
    typing_file = Path(spec.typing_extensions_site) / "typing_extensions.py"
    if not typing_file.is_file() or sha256_file(typing_file) != spec.typing_extensions_sha256:
        raise RunnerError(f"{model} worker typing_extensions.py digest mismatch")
    if str(Path(spec.typing_extensions_site)) not in sys.path:
        sys.path.insert(0, spec.typing_extensions_site)
    import typing_extensions
    if Path(typing_extensions.__file__).resolve() != typing_file.resolve():
        raise RunnerError(f"{model} worker typing_extensions import escaped frozen site")
    from importlib import metadata
    if metadata.version("typing_extensions") != spec.typing_extensions_version:
        raise RunnerError(f"{model} worker typing_extensions version mismatch")
    dist = metadata.distribution("typing_extensions")
    dist_root = Path(dist.locate_file("")).resolve()
    dist_info = Path(dist.locate_file(FROZEN_TYPING_EXTENSIONS_DIST_INFO)).resolve()
    expected_dist_info = (Path(spec.typing_extensions_site) / FROZEN_TYPING_EXTENSIONS_DIST_INFO).resolve()
    if dist_root != Path(spec.typing_extensions_site).resolve() or dist_info != expected_dist_info or not dist_info.is_dir():
        raise RunnerError(f"{model} worker typing_extensions dist-info origin mismatch")
    if dist.version != spec.typing_extensions_version:
        raise RunnerError(f"{model} worker typing_extensions distribution version mismatch")
    import torch

    if platform.python_version() != spec.python or torch.__version__ != spec.torch:
        raise RunnerError(f"{model} worker Python/Torch mismatch: {platform.python_version()}/{torch.__version__} != {spec.python}/{spec.torch}")


def _run_isolated_model_workers(args: argparse.Namespace, context: RunnerContext) -> dict[str, Any]:
    _verify_output_root_policy(context)
    payloads: list[dict[str, Any]] = []
    source_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    overlay_payload = _overlay_payload(context)
    typing_site = str(overlay_payload.get("runtime", {}).get("typing_extensions_site", FROZEN_TYPING_EXTENSIONS_SITE))
    pythonpath = os.pathsep.join([typing_site, str(source_root)])
    env["PYTHONPATH"] = pythonpath
    env["PYTHONNOUSERSITE"] = "1"
    bootstrap = (
        "import hashlib, sys\n"
        "from importlib import metadata\n"
        "from pathlib import Path\n"
        f"frozen_site = {str(FROZEN_TYPING_EXTENSIONS_SITE)!r}\n"
        f"frozen_sha = {FROZEN_TYPING_EXTENSIONS_SHA256!r}\n"
        f"frozen_version = {FROZEN_TYPING_EXTENSIONS_VERSION!r}\n"
        f"frozen_dist_info = {FROZEN_TYPING_EXTENSIONS_DIST_INFO!r}\n"
        "site = Path(sys.argv[1])\n"
        "project = Path(sys.argv[2])\n"
        "if site.as_posix().rstrip('/') != frozen_site:\n"
        "    raise SystemExit(f'typing_extensions site must equal frozen package cache: {site}')\n"
        "if site.resolve().as_posix().rstrip('/') != frozen_site:\n"
        "    raise SystemExit(f'typing_extensions site must resolve to frozen package cache: {site}')\n"
        "sys.path.insert(0, str(project))\n"
        "sys.path.insert(0, str(site))\n"
        "import typing_extensions\n"
        "actual_file = Path(typing_extensions.__file__).resolve()\n"
        "expected_file = (site / 'typing_extensions.py').resolve()\n"
        "if actual_file != expected_file or hashlib.sha256(actual_file.read_bytes()).hexdigest() != frozen_sha:\n"
        "    raise SystemExit(f'typing_extensions import escaped frozen site: {actual_file}')\n"
        "dist = metadata.distribution('typing_extensions')\n"
        "dist_root = Path(dist.locate_file('')).resolve()\n"
        "dist_info = Path(dist.locate_file(frozen_dist_info)).resolve()\n"
        "expected_dist_info = (site / frozen_dist_info).resolve()\n"
        "if dist_root != site.resolve() or dist_info != expected_dist_info or not dist_info.is_dir():\n"
        "    raise SystemExit(f'typing_extensions dist-info escaped frozen site: {dist_info}')\n"
        "if dist.version != frozen_version or metadata.version('typing_extensions') != frozen_version:\n"
        "    raise SystemExit(f'typing_extensions distribution version mismatch: {dist.version}')\n"
        "sys.argv = [sys.argv[0], *sys.argv[3:]]\n"
        "from georeliab_mve.cli import main\n"
        "raise SystemExit(main())\n"
    )
    for model in ("vggt", "mast3r"):
        python = _worker_python_path(context, model)
        shard_token = str(getattr(args, "shard", "0/1")).replace("/", "of")
        stage_token = str(getattr(args, "stage", "preflight"))
        device_token = str(args.device).replace(":", "-").replace("/", "-")
        invocation_token = f"{os.getpid()}-{time.time_ns()}-{device_token}"
        summary_path = context.output_root / "stage" / "_worker_summaries" / f"{args.command}-{stage_token}-{model}-{shard_token}-{invocation_token}.json"
        stdout_path = summary_path.with_suffix(".stdout.log")
        stderr_path = summary_path.with_suffix(".stderr.log")
        if args.command == "preflight-real":
            child_argv = [
                "preflight-real", "--config", str(args.config), "--output-root", str(args.output_root),
                "--device", args.device, "--model", model, "--summary-json", str(summary_path),
            ]
        else:
            child_argv = [
                "run-georeliab", "--stage", args.stage, "--model", model, "--device", args.device,
                "--shard", args.shard, "--config", str(args.config), "--output-root", str(args.output_root),
                "--summary-json", str(summary_path),
            ]
        result = subprocess.run(
            [str(python), "-I", "-B", "-c", bootstrap, typing_site, str(source_root), *child_argv],
            cwd=str(source_root), capture_output=True, text=True, timeout=None, env=env,
        )
        _atomic_text(stdout_path, result.stdout)
        _atomic_text(stderr_path, result.stderr)
        log_evidence = {
            "worker_stdout_path": str(stdout_path),
            "worker_stdout_sha256": sha256_file(stdout_path),
            "worker_stderr_path": str(stderr_path),
            "worker_stderr_sha256": sha256_file(stderr_path),
        }
        if result.returncode != 0:
            payloads.append({"model": model, "status": "WORKER_FAILED", "returncode": result.returncode, "stderr": result.stderr.strip(), "stdout": result.stdout.strip(), **log_evidence})
            continue
        try:
            payload = _load_json(summary_path)
        except RunnerError:
            payload = {"status": "WORKER_MISSING_SUMMARY", "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
        payload["model_worker"] = model
        payload.update(log_evidence)
        payloads.append(payload)
    status = "OK" if all(item.get("status") == "OK" for item in payloads) else "WORKER_FAILED"
    if args.command == "preflight-real" and all(item.get("status") == "OK" for item in payloads):
        status = "OK"
    scheduled = 8 if args.command == "preflight-real" else (400 if args.stage == "test" else 200 if args.stage == "smoke" else 480)
    return {"status": status, "stage": "preflight" if args.command == "preflight-real" else args.stage, "dry_run": False, "model_isolation": "per-frozen-env-python", "schedule_counts": {"scheduled": scheduled}, "workers": payloads}


def _cli_success(status: Any) -> bool:
    return status in {"OK", "DRY_RUN"}


def _emit_cli_payload(args: argparse.Namespace, payload: Mapping[str, Any]) -> None:
    summary_json = getattr(args, "summary_json", None)
    if summary_json is not None:
        _atomic_json(summary_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def cli_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        context = RunnerContext(root=args.output_root, output_root=args.output_root, config_path=args.config, device=args.device)
        if not args.dry_run and getattr(args, "model", None) == "all":
            payload = _run_isolated_model_workers(args, context)
            _emit_cli_payload(args, payload)
            return 0 if _cli_success(payload.get("status")) else 2
        if not args.dry_run and getattr(args, "model", None) in {"vggt", "mast3r"}:
            _verify_current_worker_runtime(context, args.model)
        if args.command == "preflight-real":
            payload = run_preflight_real(context, dry_run=args.dry_run, model=args.model)
        else:
            payload = run_stage(context, stage=args.stage, model=args.model, shard=args.shard, dry_run=args.dry_run)
        _emit_cli_payload(args, payload)
        return 0 if _cli_success(payload.get("status")) else 2
    except (RunnerError, OSError, ValueError, ContractError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def refuse_shrunk_test_grid(items: Sequence[ScheduleItem]) -> str:
    if len(items) != 400:
        return "TEST_GRID_SHRINK_FORBIDDEN"
    return "OK"


def evaluate_repeatability(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    same_input = first.get("input_digest") == second.get("input_digest")
    a = float(first.get("aggregate_error", 0.0))
    b = float(second.get("aggregate_error", 0.0))
    denom = max(abs(a), 1e-12)
    rel = abs(a - b) / denom
    rho_delta = abs(float(first.get("rho", 0.0)) - float(second.get("rho", 0.0)))
    passed = bool(same_input and rel < 0.001 and rho_delta < 0.005)
    return {
        "passed": passed,
        "reason_code": "OK" if passed else "PREFLIGHT_REPEATABILITY_FAILED",
        "same_input_digest": same_input,
        "aggregate_error_relative_difference": rel,
        "rho_absolute_difference": rho_delta,
    }


def stage_counts(items: Sequence[ScheduleItem], results: Sequence[RunResult] = ()) -> dict[str, int]:
    completed = sum(1 for result in results if result.state in {"completed", "skipped"})
    invalid = sum(1 for result in results if result.invalid_prediction)
    return {
        "scheduled": len(items),
        "completed": completed,
        "invalid": invalid,
        "missing": max(len(items) - completed, 0),
        "skipped": sum(1 for result in results if result.state == "skipped"),
        "retried": 0,
    }
