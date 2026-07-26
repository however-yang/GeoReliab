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
import tomllib
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .adapters import RenderedView
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


class RunnerError(RuntimeError):
    """Raised when runner governance must fail closed."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    checkpoint_sha256: str
    source_commit: str
    python: str
    torch: str
    environment_lock_sha256: str
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot read required runner JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RunnerError(f"runner JSON must be an object: {path}")
    return payload


def _git_object(args: Sequence[str], fallback: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return fallback
    value = result.stdout.strip().lower()
    return value or fallback


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
                sample = SampleKey("dtu", split, f"scan{scene:03d}", "fps8", condition, str(severity), "0")
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
    if native_gate.get("status") != "PASS":
        return {"status": "SHORT_CIRCUIT_P5", "reason_code": "NATIVE_CONFIDENCE_GATE_NOT_PASS"}
    return {"status": "PASS"}


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
        },
        "MASt3R": {
            "checkpoint_sha256": "0a615eb05fa9db654050aa655945ee5696e7c6c1b7f93f1ee8c37249010f6feb",
            "source_commit": "f5209afc300cec36239a7ac992263f36847bbba0",
            "python": "3.10.20",
            "torch": "2.5.1+cu121",
            "dust3r_source_commit": "3cc8c88c413bb9e34c41db0e0eef99c2ee010b12",
            "croco_source_commit": "d7de0705845239092414480bd829228723bf20de",
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
        if model == "VGGT":
            values["checkpoint_sha256"] = str(resources.get("vggt_checkpoint_sha256", values["checkpoint_sha256"]))
            values["source_commit"] = str(resources.get("vggt_source_commit", values["source_commit"]))
            values["python"] = str(runtime.get("vggt_python", values["python"]))
            values["torch"] = str(runtime.get("vggt_torch", values["torch"]))
        else:
            values["checkpoint_sha256"] = str(resources.get("mast3r_checkpoint_sha256", values["checkpoint_sha256"]))
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
        dust3r_source_commit=values.get("dust3r_source_commit"),
        croco_source_commit=values.get("croco_source_commit"),
    )


def _model_specs(config_path: Path | None) -> dict[str, ModelSpec]:
    return {model: frozen_model_spec(model, config_path) for model in MODELS}


def make_manifest(item: ScheduleItem, root: Path, model_specs: Mapping[str, ModelSpec], *, device: str) -> RunManifest:
    spec = model_specs[item.model]
    split_sha = sha256_file(root / "manifests" / "split_view_manifest.json")
    corruption_sha = sha256_file(root / "manifests" / "corruption_calibration.json")
    project_commit = _git_object(["rev-parse", "HEAD"], "0" * 40)
    project_tree = _git_object(["rev-parse", "HEAD^{tree}"], "1" * 40)
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
        environment={"device": device, "python": spec.python, "torch": spec.torch},
        rgb_digest=_sha_json([(view.view_id, view.png_sha256) for view in item.rendered_views]),
        prompt_digest="fixed-empty-prompt",
        decoder_digest="fixed-native-decoder",
        provenance=provenance,
    )


def _empty_invalid_prediction(manifest: RunManifest, sample_key: SampleKey, output_dir: Path, reason: str) -> PredictionArtifact:
    output_dir.mkdir(parents=True, exist_ok=True)
    geo = output_dir / "geometry_prediction.npz"
    conf = output_dir / "native_confidence.npz"
    mask = output_dir / "valid_mask.npz"
    np.savez(geo, points_world=np.empty((0, 3)), camera_c2w=np.empty((0, 4, 4)), intrinsics=np.empty((0, 3, 3)), pixel_xy=np.empty((0, 2)), view_id=np.empty((0,), dtype=np.int64))
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


def fixture_audit_factory(manifest: RunManifest, prediction: PredictionArtifact, output_dir: Path) -> AuditRecord:
    output_dir.mkdir(parents=True, exist_ok=True)
    dense = output_dir / "dense_audit.npz"
    if prediction.invalid_prediction:
        np.savez(
            dense,
            voxel_points=np.empty((0, 3)), raw_confidence=np.empty((0,)), risk=np.empty((0,)),
            gt_error=np.empty((0,)), failure_label=np.empty((0,), dtype=bool), provenance_count=np.empty((0,), dtype=np.int64),
        )
        return AuditRecord(
            manifest.run_id, prediction.sample_key, None, True, 1.0, 0.0, False, 0.0, True,
            {"dense_audit_uri": dense.as_uri(), "dense_audit_sha256": sha256_file(dense), "audit_factory": "fixture"},
        )
    np.savez(
        dense,
        voxel_points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        raw_confidence=np.array([2.0, 3.0]),
        risk=np.array([0.1, 0.2]),
        gt_error=np.array([0.5, 1.5]),
        failure_label=np.array([False, False]),
        provenance_count=np.array([1, 1], dtype=np.int64),
    )
    return AuditRecord(
        manifest.run_id, prediction.sample_key, 1.0, False, 0.1, 1.0, True, 1.0, False,
        {"dense_audit_uri": dense.as_uri(), "dense_audit_sha256": sha256_file(dense), "audit_factory": "fixture"},
    )


def default_adapter_factory(model: str, context: RunnerContext) -> Any:
    from .adapters import FrozenRuntime, MASt3RAdapter, VGGTAdapter

    if context.config_path is None or not context.config_path.exists():
        raise RunnerError("real adapter execution requires frozen A100 overlay config")
    payload = tomllib.loads(context.config_path.read_text(encoding="utf-8"))
    runtime = payload.get("runtime", {})
    resources = payload.get("resources", {})
    if model == "VGGT":
        frozen = FrozenRuntime(
            source=Path(runtime["vggt_source"]), source_commit=resources["vggt_source_commit"],
            environment=Path(runtime["vggt_env"]), python_version=runtime["vggt_python"], torch_version=runtime["vggt_torch"],
            checkpoint=Path(resources["vggt_checkpoint"]), checkpoint_sha256=resources["vggt_checkpoint_sha256"],
        )
        return VGGTAdapter(frozen, output_root=context.output_root, device=context.device)
    if model == "MASt3R":
        frozen = FrozenRuntime(
            source=Path(runtime["mast3r_source"]), source_commit=resources["mast3r_source_commit"],
            environment=Path(runtime["mast3r_env"]), python_version=runtime["mast3r_python"], torch_version=runtime["mast3r_torch"],
            checkpoint=Path(resources["mast3r_checkpoint"]), checkpoint_sha256=resources["mast3r_checkpoint_sha256"],
            config=Path(resources["mast3r_config"]), config_sha256=resources["mast3r_config_sha256"],
            dust3r_source=Path(runtime["dust3r_source"]), dust3r_source_commit=resources["dust3r_source_commit"],
            croco_source=Path(runtime["croco_source"]), croco_source_commit=resources["croco_source_commit"],
        )
        return MASt3RAdapter(frozen, output_root=context.output_root, device=context.device)
    raise RunnerError(f"unsupported model: {model}")


def _bundle_dir(output_root: Path, item: ScheduleItem) -> Path:
    return output_root / "stage" / item.stage / "bundles" / item.model.lower() / item.identity


def load_completed_bundle(bundle_dir: Path) -> tuple[RunManifest, PredictionArtifact, AuditRecord]:
    manifest = read_json_artifact(bundle_dir / "run_manifest.json", RunManifest)
    prediction = read_json_artifact(bundle_dir / "prediction_artifact.json", PredictionArtifact)
    audit = read_json_artifact(bundle_dir / "audit_record.json", AuditRecord)
    validate_artifact_bundle(manifest, prediction, audit)
    return manifest, prediction, audit


def _rewrite_bundle_uri_payloads(bundle_dir: Path, partial_dir: Path) -> None:
    partial_uri = partial_dir.as_uri()
    final_uri = bundle_dir.as_uri()
    for name in ("prediction_artifact.json", "audit_record.json"):
        path = bundle_dir / name
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


def claim_item(claim_root: Path, item: ScheduleItem, *, stale_seconds: int = 3600) -> ClaimResult:
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


def _stage_item_payload(item: ScheduleItem) -> dict[str, Any]:
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
    }


def _existing_bundle_state(bundle_dir: Path, item: ScheduleItem) -> str | None:
    if not bundle_dir.exists():
        return None
    try:
        payload = _load_json(bundle_dir / "stage_item.json")
        if payload.get("identity") != item.identity:
            raise RunnerError("provenance-conflict: existing valid artifact has different schedule identity")
        load_completed_bundle(bundle_dir)
    except (RunnerError, ContractError, OSError, ValueError) as exc:
        raise RunnerError(f"provenance-conflict: existing artifact is not a valid matching bundle: {exc}") from exc
    return "complete"


def _append_ledger(output_root: Path, stage: str, row: Mapping[str, Any]) -> None:
    path = output_root / "stage" / stage / "ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass


def read_stage_ledger(output_root: Path, stage: str) -> dict[str, Any]:
    path = output_root / "stage" / stage / "ledger.jsonl"
    rows: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    counts = {"scheduled": 0, "completed": 0, "invalid": 0, "skipped": 0, "retried": 0, "missing": 0}
    for row in rows:
        state = row.get("state")
        if state in counts:
            counts[state] += 1
        if row.get("invalid_prediction") is True:
            counts["invalid"] += 1
        if row.get("retried") is True:
            counts["retried"] += 1
    return {"schema_version": "georeliab-stage-ledger-v1", "stage": stage, "counts": counts, "rows": rows}


def execute_item(
    context: RunnerContext,
    item: ScheduleItem,
    *,
    adapter_factory: AdapterFactory = default_adapter_factory,
    audit_factory: AuditFactory = fixture_audit_factory,
) -> RunResult:
    bundle_dir = _bundle_dir(context.output_root, item)
    if _existing_bundle_state(bundle_dir, item) == "complete":
        result = RunResult(item.identity, "skipped", bundle_dir, False, "EXISTING_VALID_ARTIFACT", 0.0, 0.0)
        _append_ledger(context.output_root, item.stage, {**asdict(result), "timestamp": _utc_now()})
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
        try:
            adapter = adapter_factory(item.model, context)
            prediction = adapter.predict_sample(manifest, item.sample_key, item.rendered_views)
        except Exception as exc:
            prediction = _empty_invalid_prediction(manifest, item.sample_key, partial, f"{type(exc).__name__}: {exc}")
        audit = audit_factory(manifest, prediction, partial)
        write_json_artifact(partial / "run_manifest.json", manifest)
        write_json_artifact(partial / "prediction_artifact.json", prediction)
        write_json_artifact(partial / "audit_record.json", audit)
        _atomic_json(partial / "stage_item.json", _stage_item_payload(item))
        validate_artifact_bundle(manifest, prediction, audit)
        if bundle_dir.exists():
            raise RunnerError("provenance-conflict: destination appeared before atomic commit")
        partial.replace(bundle_dir)
        _rewrite_bundle_uri_payloads(bundle_dir, partial)
        load_completed_bundle(bundle_dir)
        runtime = time.perf_counter() - start
        result = RunResult(item.identity, "completed", bundle_dir, prediction.invalid_prediction, "OK", runtime, prediction.peak_memory_mb)
        _append_ledger(context.output_root, item.stage, {**asdict(result), "timestamp": _utc_now(), "retried": retried})
        return result
    finally:
        _release_claim(claim)


def ensure_stage_freeze(context: RunnerContext, stage: str, *, dry_run: bool) -> dict[str, Any]:
    if stage != "test":
        return {"stage": stage, "frozen": False, "dry_run": dry_run}
    split_path = context.root / "manifests" / "split_view_manifest.json"
    corruption_path = context.root / "manifests" / "corruption_calibration.json"
    payload = {
        "schema_version": "georeliab-test-stage-freeze-v1",
        "stage": "test",
        "split_view_manifest_sha256": sha256_file(split_path),
        "corruption_manifest_sha256": sha256_file(corruption_path),
        "schedule_count": len(full_schedule(context.root, "test")),
    }
    if payload["schedule_count"] != 400:
        raise RunnerError("refusing to freeze a shrunk P3 test grid")
    path = context.output_root / "stage" / "test" / "stage_freeze.json"
    if path.exists():
        existing = _load_json(path)
        if existing != payload:
            raise RunnerError("test stage parameters changed after freeze")
        return existing
    if not dry_run:
        _atomic_json(path, payload)
    return payload


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
