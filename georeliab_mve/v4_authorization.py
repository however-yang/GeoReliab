"""CPU-only GeoReliab v4 GPU preflight and execution authorization.

This module only inspects already-prepared inputs and hardware metadata. It does
not load model checkpoints, instantiate adapters, execute forwards, compute
scientific metrics, acquire the scientific execution lock, or touch the GPU
inference ledger.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import platform
import subprocess
import time
from typing import Any, Callable
from uuid import uuid4

from .v4_counterfactuals import (
    FOG_STATES,
    LIGHTING_STATES,
    SCIENTIFIC_MODELS,
    SCIENTIFIC_STATES,
    TEST_SCENE_IDS,
    parse_scientific_schedule,
    validate_scientific_schedule,
)
from .v4_execution import (
    BYTE_CATASTROPHE,
    BYTE_TARGET,
    GPU_CATASTROPHE_SECONDS,
    GPU_TARGET_SECONDS,
    SCIENTIFIC_MVE,
    V4ExecutionError,
    V4ExecutionReceipt,
)
from .v4_science_lock import V4_PROTOCOL_ID, V4_PROTOCOL_SHA256


IMPLEMENTATION_ANCHOR_COMMIT = "7381e60050143a78fca6a3ebde5706ae27d2c145"
IMPLEMENTATION_ANCHOR_TREE = "f4e2b1104496c817693aaa5989d0276d2ebe03e9"
V4_HARDWARE_PREFLIGHT_SCHEMA_VERSION = "georeliab-v4-hardware-preflight-1.0"
V4_EXECUTION_AUTHORIZATION_SCHEMA_VERSION = "georeliab-v4-execution-authorization-1.0"
V4_PREFLIGHT_DECISION_SCHEMA_VERSION = "georeliab-v4-preflight-decision-1.0"
AUTHORIZED_GPU_MODEL = "NVIDIA A100 80GB PCIe"
MIN_FREE_MEMORY_BYTES = 16 * 1024 * 1024 * 1024
FORMAL_SAMPLE_INTERVAL_SECONDS = 5.0
AUTHORIZED_FINALIZER = "georeliab_mve.v4_execution:finalize_v4_scientific_bundle"
AUTHORIZED_RESOURCE_KEYS = (
    "raw_data",
    "model_weights_config",
    "upstream_commits_manifests",
    "split",
    "state_inventory_200",
    "fog_manifest",
    "scientific_schedule_400",
    "environment_locks",
)
PROCESS_PRESENT_REASON_CODES = frozenset(
    {
        "V4_GPU_NON_GEORELIAB_COMPUTE_PROCESS_PRESENT",
        "V4_GPU_GEORELIAB_RESIDUAL_COMPUTE_PROCESS_PRESENT",
        "V4_GPU_ACTIVE_COMPUTE_PROCESS_IDENTITY_INCOMPLETE",
        "V4_GPU_ACTIVE_COMPUTE_PROCESS_STATE_UNSTABLE",
    }
)
NO_ACTIVE_PROCESS_REASON_CODES = frozenset(
    {"V4_GPU_RECEIPT_NO_ACTIVE_COMPUTE_PROCESS"}
)
PROCESS_EVIDENCE_REASON_CODES = PROCESS_PRESENT_REASON_CODES | frozenset(
    {"V4_GPU_UNEXPLAINED_ACTIVITY"}
)


@dataclass(frozen=True, slots=True)
class V4ExecutionAuthorization:
    receipt_path: str
    receipt_sha256: str
    hardware_preflight_path: str
    hardware_preflight_sha256: str
    implementation_commit: str
    implementation_tree: str
    protocol_id: str
    protocol_sha256: str
    schedule_sha256: str
    resource_inventory: tuple[tuple[str, str, str], ...]
    root: str
    run_root: str
    artifact_root: str
    final_evidence_path: str
    finalizer: str
    schema_version: str = V4_EXECUTION_AUTHORIZATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["resource_inventory"] = [
            {"key": key, "path": path, "sha256": digest}
            for key, path, digest in self.resource_inventory
        ]
        payload["authorized_scope"] = _authorized_scope()
        payload["authorization_sha256"] = _sha_json(
            {key: value for key, value in payload.items() if key != "authorization_sha256"}
        )
        return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), default=str)
        + "\n"
    ).encode("utf-8")


def _atomic_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    validator: Callable[[Mapping[str, Any]], None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        with partial.open("wb") as handle:
            handle.write(_json_bytes(payload))
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        reloaded = json.loads(partial.read_text(encoding="utf-8"))
        if not isinstance(reloaded, dict):
            raise V4ExecutionError("V4_ATOMIC_JSON_STAGING_VALIDATION_FAILED")
        validator(reloaded)
        partial.replace(path)
    finally:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass


def _runtime_proven(value: object) -> bool:
    return isinstance(value, str) and value.strip() != "" and value.strip().lower() != "unknown"


def _driver_proven(value: object) -> bool:
    return isinstance(value, str) and value.strip() != "" and value.strip().lower() != "unknown"


def _validate_preflight_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != V4_HARDWARE_PREFLIGHT_SCHEMA_VERSION:
        raise V4ExecutionError("V4_GPU_PREFLIGHT_SCHEMA_REQUIRED")
    if payload.get("status") not in {"PASS", "FAIL"} or not isinstance(payload.get("reason_code"), str):
        raise V4ExecutionError("V4_GPU_PREFLIGHT_SCHEMA_REQUIRED")
    for sample in payload.get("samples", ()):
        if not isinstance(sample, Mapping) or not _runtime_proven(sample.get("cuda_runtime")):
            raise V4ExecutionError("V4_GPU_CUDA_RUNTIME_UNPROVEN")
        if not _driver_proven(sample.get("driver_version")):
            raise V4ExecutionError("V4_GPU_DRIVER_VERSION_UNPROVEN")
    for device in payload.get("devices", ()):
        if not isinstance(device, Mapping) or not _runtime_proven(device.get("cuda_runtime")):
            raise V4ExecutionError("V4_GPU_CUDA_RUNTIME_UNPROVEN")
        if not _driver_proven(device.get("driver_version")):
            raise V4ExecutionError("V4_GPU_DRIVER_VERSION_UNPROVEN")
    _validate_evidence_reason_consistency(payload)
    if payload.get("status") == "PASS":
        decision = _evaluate_basic(payload.get("samples", ()))
        if decision["status"] != "PASS":
            raise V4ExecutionError(str(decision["reason_code"]))
        probe_decision = _evaluate_probes(payload.get("model_environment_probes", ()), payload["samples"][-1])
        if probe_decision["status"] != "PASS":
            raise V4ExecutionError(str(probe_decision["reason_code"]))


def _validate_receipt_payload(payload: Mapping[str, Any]) -> None:
    receipt = V4ExecutionReceipt.from_mapping(payload)
    _validate_receipt_contract(receipt)


def _validate_authorization_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != V4_EXECUTION_AUTHORIZATION_SCHEMA_VERSION:
        raise V4ExecutionError("V4_AUTHORIZATION_SCHEMA_REQUIRED")
    expected_sha = payload.get("authorization_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "authorization_sha256"}
    if not _is_sha(expected_sha) or _sha_json(unsigned) != expected_sha:
        raise V4ExecutionError("V4_AUTHORIZATION_TAMPER")
    if payload.get("finalizer") != AUTHORIZED_FINALIZER:
        raise V4ExecutionError("V4_AUTHORIZATION_FINALIZER_MISMATCH")


def _validate_preflight_decision_payload(
    payload: Mapping[str, Any],
    *,
    expected_snapshot_path: Path | None = None,
    expected_authorization_commit: str | None = None,
    expected_authorization_tree: str | None = None,
    expected_run_id: str | None = None,
) -> None:
    if payload.get("schema_version") != V4_PREFLIGHT_DECISION_SCHEMA_VERSION:
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_SCHEMA_REQUIRED")
    if payload.get("implementation_commit") != IMPLEMENTATION_ANCHOR_COMMIT or payload.get("implementation_tree") != IMPLEMENTATION_ANCHOR_TREE:
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_ANCHOR_MISMATCH")
    authorization_commit = payload.get("authorization_commit")
    authorization_tree = payload.get("authorization_tree")
    if not _is_sha(authorization_commit, length=40) or not _is_sha(authorization_tree, length=40):
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_REVISION_MISMATCH")
    if expected_authorization_commit is not None and authorization_commit != expected_authorization_commit:
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_REVISION_MISMATCH")
    if expected_authorization_tree is not None and authorization_tree != expected_authorization_tree:
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_REVISION_MISMATCH")
    run_id = payload.get("run_id")
    if not _is_sha(run_id, length=32):
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_SCHEMA_REQUIRED")
    if expected_run_id is not None and run_id != expected_run_id:
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_RUN_MISMATCH")
    if not isinstance(payload.get("requested_physical_index"), int) or isinstance(payload.get("requested_physical_index"), bool):
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_SCHEMA_REQUIRED")
    if payload.get("status") != "BLOCKED" or not isinstance(payload.get("reason_code"), str):
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_BLOCKED_REQUIRED")
    if payload.get("terminal_status") != "BLOCKED" or payload.get("terminal_reason_code") != payload.get("reason_code"):
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_TERMINAL_REQUIRED")
    if payload.get("scientific_result") != "NO_SCIENTIFIC_RESULT":
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_NO_SCIENTIFIC_RESULT_REQUIRED")
    snapshot_path = Path(str(payload.get("hardware_preflight_path")))
    if expected_snapshot_path is not None and snapshot_path.resolve() != expected_snapshot_path.resolve():
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_SNAPSHOT_PATH_MISMATCH")
    if not snapshot_path.is_file():
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_SNAPSHOT_REQUIRED")
    snapshot_sha = payload.get("hardware_preflight_sha256")
    if not _is_sha(snapshot_sha) or sha256_file(snapshot_path) != snapshot_sha:
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_SNAPSHOT_TAMPER")
    snapshot = load_json(snapshot_path)
    if snapshot.get("schema_version") != V4_HARDWARE_PREFLIGHT_SCHEMA_VERSION or snapshot.get("status") != "FAIL":
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_SNAPSHOT_MISMATCH")
    if snapshot.get("reason_code") != payload.get("reason_code"):
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_SNAPSHOT_MISMATCH")
    if snapshot.get("project_commit") != IMPLEMENTATION_ANCHOR_COMMIT or snapshot.get("project_tree") != IMPLEMENTATION_ANCHOR_TREE:
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_SNAPSHOT_MISMATCH")
    if snapshot.get("authorization_commit") != authorization_commit or snapshot.get("authorization_tree") != authorization_tree:
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_SNAPSHOT_MISMATCH")
    if snapshot.get("requested_physical_index") != payload.get("requested_physical_index"):
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_SNAPSHOT_MISMATCH")
    if snapshot.get("run_id") != run_id:
        raise V4ExecutionError("V4_PREFLIGHT_DECISION_RUN_MISMATCH")
    _validate_evidence_reason_consistency(snapshot)


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V4ExecutionError("V4_AUTHORIZATION_JSON_UNREADABLE") from exc
    if not isinstance(payload, dict):
        raise V4ExecutionError("V4_AUTHORIZATION_JSON_NOT_OBJECT")
    return payload


def _is_sha(value: object, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _current_authorization_revision() -> tuple[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        commit = _run_text_command(("git", "-C", str(repo_root), "rev-parse", "HEAD")).strip()
        tree = _run_text_command(("git", "-C", str(repo_root), "rev-parse", "HEAD^{tree}")).strip()
    except Exception as exc:
        raise V4ExecutionError("V4_AUTHORIZATION_REVISION_UNRESOLVED") from exc
    if not _is_sha(commit, length=40) or not _is_sha(tree, length=40):
        raise V4ExecutionError("V4_AUTHORIZATION_REVISION_UNRESOLVED")
    return commit, tree


def _resolve_authorization_revision(authorization_commit: str | None, authorization_tree: str | None) -> tuple[str, str]:
    if authorization_commit is None and authorization_tree is None:
        return _current_authorization_revision()
    if not _is_sha(authorization_commit, length=40) or not _is_sha(authorization_tree, length=40):
        raise V4ExecutionError("V4_AUTHORIZATION_REVISION_REQUIRED")
    return str(authorization_commit), str(authorization_tree)


def _resolve_under_root(root: Path, target: Path, *, must_exist: bool = False) -> Path:
    resolved_root = root.resolve()
    if resolved_root == Path(resolved_root.anchor):
        raise V4ExecutionError("V4_AUTHORIZATION_ROOT_INVALID")
    resolved = target.resolve(strict=must_exist)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise V4ExecutionError("V4_AUTHORIZATION_PATH_ESCAPE") from exc
    if not relative.parts:
        raise V4ExecutionError("V4_AUTHORIZATION_PATH_ESCAPE")
    return resolved


def _run_text_command(command: Sequence[str], *, env: Mapping[str, str] | None = None) -> str:
    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update({key: str(value) for key, value in env.items()})
    return subprocess.run(
        list(command),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=merged_env,
    ).stdout


def _parse_csv_row(text: str, expected_columns: int, *, reason_code: str) -> list[str]:
    rows = [row.strip() for row in text.splitlines() if row.strip()]
    if len(rows) != 1:
        raise V4ExecutionError(reason_code)
    values = [item.strip() for item in rows[0].split(",")]
    if len(values) != expected_columns:
        raise V4ExecutionError(reason_code)
    return values


def _int_from_smi(value: str) -> int:
    stripped = value.strip()
    if stripped in {"", "N/A", "[Not Supported]", "Not Supported"}:
        return 0
    return int(float(stripped))


def _bytes_from_mib(value: str) -> int:
    return _int_from_smi(value) * 1024 * 1024


def _cuda_runtime_from_nvidia_smi_banner(text: str) -> str:
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)*)", text)
    if match is None:
        raise V4ExecutionError("V4_GPU_CUDA_RUNTIME_UNPROVEN")
    return f"CUDA Version {match.group(1)}"


def _nvidia_smi_cuda_runtime(command_runner: Callable[..., str]) -> str:
    try:
        runtime = _cuda_runtime_from_nvidia_smi_banner(command_runner(("nvidia-smi",)))
    except V4ExecutionError:
        raise
    except Exception as exc:
        raise V4ExecutionError("V4_GPU_CUDA_RUNTIME_UNPROVEN") from exc
    if not _runtime_proven(runtime):
        raise V4ExecutionError("V4_GPU_CUDA_RUNTIME_UNPROVEN")
    return runtime


def _process_owner(pid: int) -> str | None:
    if os.name == "nt":
        return None
    try:
        import pwd

        uid_text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8").split("Uid:")[1].splitlines()[0].strip().split()[0]
        return pwd.getpwuid(int(uid_text)).pw_name
    except (OSError, IndexError, KeyError, ValueError):
        return None


def _process_cwd(pid: int) -> str | None:
    if os.name == "nt":
        return None
    try:
        return str(Path(f"/proc/{pid}/cwd").resolve())
    except OSError:
        return None


def _process_cmdline(pid: int) -> str | None:
    if os.name == "nt":
        return None
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except OSError:
        return None


def nvidia_smi_hardware_sample(
    requested_physical_index: int,
    *,
    command_runner: Callable[..., str] = _run_text_command,
) -> dict[str, object]:
    base = _parse_csv_row(
        command_runner((
            "nvidia-smi",
            "-i",
            str(requested_physical_index),
            "--query-gpu=index,uuid,name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu,driver_version",
            "--format=csv,noheader,nounits",
        )),
        9,
        reason_code="V4_GPU_BASIC_SAMPLE_UNAVAILABLE",
    )
    mig_ecc = _parse_csv_row(
        command_runner((
            "nvidia-smi",
            "-i",
            str(requested_physical_index),
            "--query-gpu=mig.mode.current,ecc.errors.uncorrected.volatile.total",
            "--format=csv,noheader,nounits",
        )),
        2,
        reason_code="V4_GPU_HEALTH_SAMPLE_UNAVAILABLE",
    )
    cuda_runtime = _nvidia_smi_cuda_runtime(command_runner)
    processes: list[dict[str, object]] = []
    try:
        proc_text = command_runner((
            "nvidia-smi",
            "-i",
            str(requested_physical_index),
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ))
    except Exception as exc:
        raise V4ExecutionError("V4_GPU_PROCESS_ENUMERATION_UNPROVEN") from exc
    for row in [line.strip() for line in proc_text.splitlines() if line.strip()]:
        gpu_uuid, pid_raw, process_name, used_raw = [item.strip() for item in row.split(",", 3)]
        pid = _int_from_smi(pid_raw)
        processes.append({
            "gpu_uuid": gpu_uuid,
            "pid": pid,
            "process_name": process_name,
            "owner": _process_owner(pid),
            "cwd": _process_cwd(pid),
            "cmdline": _process_cmdline(pid),
            "used_memory_bytes": _bytes_from_mib(used_raw),
        })
    return {
        "host": platform.node(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "requested_physical_index": requested_physical_index,
        "resolved_physical_index": _int_from_smi(base[0]),
        "device_uuid": base[1],
        "device_model": base[2],
        "total_memory_bytes": _bytes_from_mib(base[3]),
        "free_memory_bytes": _bytes_from_mib(base[4]),
        "used_memory_bytes": _bytes_from_mib(base[5]),
        "utilization_gpu_percent": _int_from_smi(base[6]),
        "temperature_c": _int_from_smi(base[7]),
        "driver_version": base[8],
        "cuda_runtime": cuda_runtime,
        "mig_mode": mig_ecc[0],
        "ecc_health": "OK" if _int_from_smi(mig_ecc[1]) == 0 else "ERROR",
        "compute_processes": processes,
    }


def _frozen_python_for_model(model_id: str) -> str:
    env_key = {"VGGT": "GEORELIAB_V4_VGGT_PYTHON", "MASt3R": "GEORELIAB_V4_MAST3R_PYTHON"}.get(model_id)
    if env_key is None:
        raise V4ExecutionError("V4_GPU_TORCH_PROBE_MODEL_SET_MISMATCH")
    value = os.environ.get(env_key)
    if not value:
        raise V4ExecutionError(f"V4_GPU_TORCH_PROBE_FROZEN_ENV_REQUIRED:{env_key}")
    return value


def _default_torch_probe(model_id: str, requested_physical_index: int, expected_sample: Mapping[str, object]) -> dict[str, object]:
    code = (
        "import json, torch;"
        "props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() and torch.cuda.device_count()==1 else None;"
        "payload={'torch_cuda_available': torch.cuda.is_available(),"
        "'torch_device_count': torch.cuda.device_count(),"
        "'torch_current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,"
        "'device_name': props.name if props else None,"
        "'total_memory_bytes': props.total_memory if props else None};"
        "print(json.dumps(payload, sort_keys=True))"
    )
    payload = json.loads(_run_text_command((_frozen_python_for_model(model_id), "-c", code), env={"CUDA_VISIBLE_DEVICES": str(requested_physical_index)}))
    post_probe_sample = nvidia_smi_hardware_sample(requested_physical_index)
    return {
        "model_id": model_id,
        "torch_device_count": payload.get("torch_device_count"),
        "torch_cuda_available": payload.get("torch_cuda_available"),
        "torch_current_device": payload.get("torch_current_device"),
        "mapped_device_uuid": post_probe_sample.get("device_uuid"),
        "mapped_device_model": payload.get("device_name"),
        "mapped_total_memory_bytes": payload.get("total_memory_bytes"),
        "post_probe_physical_model": post_probe_sample.get("device_model"),
        "post_probe_physical_total_memory_bytes": post_probe_sample.get("total_memory_bytes"),
        "compute_process_count": len(post_probe_sample.get("compute_processes", ())),
    }


def _fail(reason_code: str, **extra: object) -> dict[str, object]:
    return {"status": "FAIL", "reason_code": reason_code, **extra}


def _active_process_identity(
    process: Mapping[str, object],
) -> tuple[int, str, str] | None:
    pid = process.get("pid")
    owner = process.get("owner")
    cmdline = process.get("cmdline")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 0
        or not isinstance(owner, str)
        or not owner.strip()
        or not isinstance(cmdline, str)
        or not cmdline.strip()
    ):
        return None
    return pid, owner.strip(), cmdline.strip()


def _is_georeliab_process(process: Mapping[str, object]) -> bool:
    provenance = " ".join(
        str(process.get(key) or "")
        for key in ("process_name", "cwd", "cmdline")
    ).lower()
    return "georeliab" in provenance


def _classify_active_compute_processes(
    samples: Sequence[Mapping[str, object]],
) -> str | None:
    process_rows: list[list[Mapping[str, object]]] = []
    for sample in samples:
        raw_processes = sample.get("compute_processes", ())
        if not isinstance(raw_processes, Sequence) or isinstance(
            raw_processes, (str, bytes)
        ):
            return "V4_GPU_ACTIVE_COMPUTE_PROCESS_IDENTITY_INCOMPLETE"
        rows: list[Mapping[str, object]] = []
        for process in raw_processes:
            if not isinstance(process, Mapping):
                return "V4_GPU_ACTIVE_COMPUTE_PROCESS_IDENTITY_INCOMPLETE"
            rows.append(process)
            if _active_process_identity(process) is None:
                return "V4_GPU_ACTIVE_COMPUTE_PROCESS_IDENTITY_INCOMPLETE"
        process_rows.append(rows)

    if not any(process_rows):
        if any(sample.get("utilization_gpu_percent") != 0 for sample in samples):
            return "V4_GPU_UNEXPLAINED_ACTIVITY"
        return None

    identity_sets = [
        {_active_process_identity(process) for process in rows}
        for rows in process_rows
    ]
    if any(identities != identity_sets[0] for identities in identity_sets[1:]):
        return "V4_GPU_ACTIVE_COMPUTE_PROCESS_STATE_UNSTABLE"
    if any(
        not _is_georeliab_process(process)
        for rows in process_rows
        for process in rows
    ):
        return "V4_GPU_NON_GEORELIAB_COMPUTE_PROCESS_PRESENT"
    return "V4_GPU_GEORELIAB_RESIDUAL_COMPUTE_PROCESS_PRESENT"


def _validate_evidence_reason_consistency(payload: Mapping[str, Any]) -> None:
    samples = payload.get("samples", ())
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise V4ExecutionError("V4_GPU_EVIDENCE_REASON_COUNT_CONTRADICTION")
    if not samples:
        if payload.get("reason_code") in (
            PROCESS_PRESENT_REASON_CODES | NO_ACTIVE_PROCESS_REASON_CODES
        ):
            raise V4ExecutionError("V4_GPU_EVIDENCE_REASON_COUNT_CONTRADICTION")
        return
    if any(not isinstance(sample, Mapping) for sample in samples):
        raise V4ExecutionError("V4_GPU_EVIDENCE_REASON_COUNT_CONTRADICTION")

    observed_count = max(
        len(sample.get("compute_processes", ())) for sample in samples
    )
    declared_count = payload.get("compute_process_count")
    if declared_count is not None and (
        not isinstance(declared_count, int)
        or isinstance(declared_count, bool)
        or declared_count != observed_count
    ):
        raise V4ExecutionError("V4_GPU_EVIDENCE_REASON_COUNT_CONTRADICTION")
    for device in payload.get("devices", ()):
        if (
            isinstance(device, Mapping)
            and "compute_process_count" in device
            and device.get("compute_process_count") != observed_count
        ):
            raise V4ExecutionError("V4_GPU_EVIDENCE_REASON_COUNT_CONTRADICTION")

    reason_code = payload.get("reason_code")
    if (
        reason_code in (PROCESS_PRESENT_REASON_CODES | NO_ACTIVE_PROCESS_REASON_CODES)
        and declared_count is None
    ):
        raise V4ExecutionError("V4_GPU_EVIDENCE_REASON_COUNT_CONTRADICTION")
    if reason_code in PROCESS_PRESENT_REASON_CODES and observed_count == 0:
        raise V4ExecutionError("V4_GPU_EVIDENCE_REASON_COUNT_CONTRADICTION")
    if reason_code in NO_ACTIVE_PROCESS_REASON_CODES and observed_count > 0:
        raise V4ExecutionError("V4_GPU_EVIDENCE_REASON_COUNT_CONTRADICTION")
    expected_reason = _classify_active_compute_processes(samples)
    if expected_reason is not None and reason_code != expected_reason:
        raise V4ExecutionError("V4_GPU_EVIDENCE_REASON_COUNT_CONTRADICTION")
    if expected_reason is None and reason_code in PROCESS_EVIDENCE_REASON_CODES:
        raise V4ExecutionError("V4_GPU_EVIDENCE_REASON_COUNT_CONTRADICTION")
    if reason_code in NO_ACTIVE_PROCESS_REASON_CODES:
        raise V4ExecutionError("V4_GPU_LEGACY_INVERTED_REASON_FORBIDDEN")


def _evaluate_basic(samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(samples) != 2:
        return _fail("V4_GPU_PREFLIGHT_TWO_SAMPLES_REQUIRED")
    first, second = samples
    for sample in samples:
        if sample.get("requested_physical_index") != sample.get("resolved_physical_index"):
            return _fail("V4_GPU_INDEX_RESOLUTION_MISMATCH")
        if not isinstance(sample.get("device_uuid"), str) or not sample.get("device_uuid"):
            return _fail("V4_GPU_UUID_MISSING")
        if sample.get("device_model") != AUTHORIZED_GPU_MODEL:
            return _fail("V4_GPU_MODEL_NOT_AUTHORIZED")
        if not _driver_proven(sample.get("driver_version")):
            return _fail("V4_GPU_DRIVER_VERSION_UNPROVEN")
        if str(sample.get("mig_mode")).lower() not in {"disabled", "off", "0"}:
            return _fail("V4_GPU_MIG_ENABLED")
        if str(sample.get("ecc_health")).upper() not in {"OK", "PASS", "NONE", "0"}:
            return _fail("V4_GPU_HEALTH_ERROR")
        if not isinstance(sample.get("free_memory_bytes"), int) or int(sample["free_memory_bytes"]) < MIN_FREE_MEMORY_BYTES:
            return _fail("V4_GPU_FREE_MEMORY_INSUFFICIENT")
    active_process_reason = _classify_active_compute_processes(samples)
    if active_process_reason is not None:
        return _fail(active_process_reason)
    for key, reason in (("resolved_physical_index", "V4_GPU_INDEX_DRIFT"), ("device_uuid", "V4_GPU_UUID_DRIFT"), ("device_model", "V4_GPU_MODEL_DRIFT"), ("driver_version", "V4_GPU_DRIVER_VERSION_DRIFT")):
        if first.get(key) != second.get(key):
            return _fail(reason)
    return {"status": "PASS", "reason_code": "V4_GPU_BASIC_PREFLIGHT_PASS"}


def _evaluate_probes(probes: Sequence[Mapping[str, object]], sample: Mapping[str, object]) -> dict[str, object]:
    seen: set[str] = set()
    if len(probes) != len(SCIENTIFIC_MODELS):
        return _fail("V4_GPU_TORCH_PROBE_MODEL_SET_MISMATCH")
    for probe in probes:
        model_id = probe.get("model_id")
        if model_id not in SCIENTIFIC_MODELS or str(model_id) in seen:
            return _fail("V4_GPU_TORCH_PROBE_MODEL_SET_MISMATCH")
        seen.add(str(model_id))
        if probe.get("torch_cuda_available") is not True or probe.get("torch_device_count") != 1 or probe.get("torch_current_device") != 0:
            return _fail("V4_GPU_TORCH_PROBE_VISIBLE_DEVICE_MISMATCH")
        if (
            not isinstance(probe.get("mapped_device_uuid"), str)
            or not isinstance(probe.get("mapped_device_model"), str)
            or not isinstance(probe.get("mapped_total_memory_bytes"), int)
            or not isinstance(probe.get("post_probe_physical_model"), str)
            or not isinstance(probe.get("post_probe_physical_total_memory_bytes"), int)
            or not isinstance(probe.get("compute_process_count"), int)
        ):
            return _fail("V4_GPU_TORCH_PROBE_SCHEMA_REQUIRED")
        if (
            probe["mapped_device_uuid"] != sample.get("device_uuid")
            or probe["mapped_device_model"] != sample.get("device_model")
            or probe["mapped_total_memory_bytes"] != sample.get("total_memory_bytes")
            or probe["post_probe_physical_model"] != sample.get("device_model")
            or probe["post_probe_physical_total_memory_bytes"] != sample.get("total_memory_bytes")
        ):
            return _fail("V4_GPU_TORCH_PROBE_PHYSICAL_DEVICE_MISMATCH")
        if probe["compute_process_count"] != 0:
            return _fail("V4_GPU_TORCH_PROBE_LEFT_PROCESS")
    return {"status": "PASS", "reason_code": "V4_GPU_PREFLIGHT_PASS"}


def _remove_owned_preflight_siblings(output_path: Path) -> None:
    for name in (
        "v4-execution-receipt.json",
        "v4-execution-authorization.json",
        "authorization.json",
        "v4-execution-schedule.json",
        "v4-state-inventory.json",
    ):
        sibling = output_path.with_name(name)
        try:
            sibling.unlink()
        except FileNotFoundError:
            pass


def _preflight_decision_path(output_path: Path) -> Path:
    return output_path.with_name("v4-preflight-decision.json")


def _write_blocked_preflight_decision(
    *,
    output_path: Path,
    requested_physical_index: int,
    reason_code: str,
    authorization_commit: str,
    authorization_tree: str,
    run_id: str,
) -> dict[str, object]:
    snapshot_sha = sha256_file(output_path)
    decision_path = _preflight_decision_path(output_path)
    payload = {
        "schema_version": V4_PREFLIGHT_DECISION_SCHEMA_VERSION,
        "implementation_commit": IMPLEMENTATION_ANCHOR_COMMIT,
        "implementation_tree": IMPLEMENTATION_ANCHOR_TREE,
        "authorization_commit": authorization_commit,
        "authorization_tree": authorization_tree,
        "run_id": run_id,
        "requested_physical_index": requested_physical_index,
        "hardware_preflight_path": str(output_path),
        "hardware_preflight_sha256": snapshot_sha,
        "status": "BLOCKED",
        "reason_code": reason_code,
        "terminal_status": "BLOCKED",
        "terminal_reason_code": reason_code,
        "scientific_result": "NO_SCIENTIFIC_RESULT",
    }
    _atomic_json(
        decision_path,
        payload,
        validator=lambda staged: _validate_preflight_decision_payload(
            staged,
            expected_snapshot_path=output_path,
            expected_authorization_commit=authorization_commit,
            expected_authorization_tree=authorization_tree,
            expected_run_id=run_id,
        ),
    )
    return {
        "preflight_decision_path": str(decision_path),
        "preflight_decision_sha256": sha256_file(decision_path),
    }


def create_hardware_preflight(
    *,
    output_path: Path,
    requested_physical_index: int,
    project_commit: str = IMPLEMENTATION_ANCHOR_COMMIT,
    project_tree: str = IMPLEMENTATION_ANCHOR_TREE,
    scope: str = SCIENTIFIC_MVE,
    stage: str = SCIENTIFIC_MVE,
    schedule_sha256: str | None = None,
    authorization_commit: str | None = None,
    authorization_tree: str | None = None,
    sample_interval_seconds: float = 5.0,
    sampler: Callable[[int], Mapping[str, object]] = nvidia_smi_hardware_sample,
    sleeper: Callable[[float], None] = time.sleep,
    probe_runner: Callable[[str, int, Mapping[str, object]], Mapping[str, object]] = _default_torch_probe,
) -> dict[str, object]:
    if not isinstance(requested_physical_index, int) or isinstance(requested_physical_index, bool) or requested_physical_index < 0:
        raise V4ExecutionError("V4_GPU_INDEX_INVALID")
    if float(sample_interval_seconds) != FORMAL_SAMPLE_INTERVAL_SECONDS:
        raise V4ExecutionError("V4_GPU_SAMPLE_INTERVAL_MUST_BE_5_SECONDS")
    resolved_authorization_commit, resolved_authorization_tree = _resolve_authorization_revision(authorization_commit, authorization_tree)
    run_id = uuid4().hex
    samples: list[dict[str, object]] = []
    try:
        samples.append(dict(sampler(requested_physical_index)))
        sleeper(float(sample_interval_seconds))
        samples.append(dict(sampler(requested_physical_index)))
        decision = _evaluate_basic(samples)
    except V4ExecutionError as exc:
        decision = _fail(str(exc), error=str(exc))
    except Exception as exc:
        decision = _fail("V4_GPU_BASIC_SAMPLE_UNAVAILABLE", error=str(exc))
    probes: list[dict[str, object]] = []
    if decision["status"] == "PASS":
        try:
            probes = [dict(probe_runner(model, requested_physical_index, samples[-1])) for model in SCIENTIFIC_MODELS]
            decision = _evaluate_probes(probes, samples[-1])
        except V4ExecutionError as exc:
            decision = _fail(str(exc), error=str(exc))
        except Exception as exc:
            decision = _fail("V4_GPU_TORCH_PROBE_FAILED", error=str(exc))
    selected = samples[-1] if samples else {}
    evidence_process_count = max(
        (len(sample.get("compute_processes", ())) for sample in samples),
        default=0,
    )
    receipt_path = output_path.with_name("v4-execution-receipt.json")
    if decision["status"] != "PASS":
        _remove_owned_preflight_siblings(output_path)
    snapshot = {
        "schema_version": V4_HARDWARE_PREFLIGHT_SCHEMA_VERSION,
        "status": decision["status"],
        "reason_code": decision["reason_code"],
        "project_commit": project_commit,
        "project_tree": project_tree,
        "scope": scope,
        "stage": stage,
        "authorization_commit": resolved_authorization_commit,
        "authorization_tree": resolved_authorization_tree,
        "run_id": run_id,
        "requested_physical_index": requested_physical_index,
        "resolved_physical_index": selected.get("resolved_physical_index"),
        "sample_interval_seconds": float(sample_interval_seconds),
        "visible_gpu_count": 1 if decision["status"] == "PASS" else 0,
        "compute_process_count": evidence_process_count,
        "stable_sample_count": len(samples),
        "environment": {"CUDA_VISIBLE_DEVICES": str(requested_physical_index), "GEORELIAB_PHYSICAL_GPU_DEVICE": f"cuda:{requested_physical_index}"},
        "devices": [{
            "physical_index": selected.get("resolved_physical_index"),
            "uuid": selected.get("device_uuid"),
            "model": selected.get("device_model"),
            "driver_version": selected.get("driver_version"),
            "total_memory_bytes": selected.get("total_memory_bytes"),
            "free_memory_bytes": selected.get("free_memory_bytes"),
            "used_memory_bytes": selected.get("used_memory_bytes"),
            "utilization_gpu_percent": selected.get("utilization_gpu_percent"),
            "temperature_c": selected.get("temperature_c"),
            "cuda_runtime": selected.get("cuda_runtime"),
            "mig_mode": selected.get("mig_mode"),
            "ecc_health": selected.get("ecc_health"),
            "compute_process_count": evidence_process_count,
        }] if selected else [],
        "samples": [{"sample_index": index, **sample} for index, sample in enumerate(samples)],
        "model_environment_probes": probes,
    }
    if "error" in decision:
        snapshot["error"] = decision["error"]
    _atomic_json(output_path, snapshot, validator=_validate_preflight_payload)
    result: dict[str, object] = {
        "status": snapshot["status"],
        "reason_code": snapshot["reason_code"],
        "hardware_preflight_path": str(output_path),
        "hardware_preflight_sha256": sha256_file(output_path),
    }
    if snapshot["status"] != "PASS":
        result.update(_write_blocked_preflight_decision(
            output_path=output_path,
            requested_physical_index=requested_physical_index,
            reason_code=str(snapshot["reason_code"]),
            authorization_commit=resolved_authorization_commit,
            authorization_tree=resolved_authorization_tree,
            run_id=run_id,
        ))
        return result
    try:
        _preflight_decision_path(output_path).unlink()
    except FileNotFoundError:
        pass
    receipt = V4ExecutionReceipt(
        explicit_user_selection=True,
        project_commit=project_commit,
        project_tree=project_tree,
        protocol_id=V4_PROTOCOL_ID,
        protocol_sha256=V4_PROTOCOL_SHA256,
        scope=scope,
        stage=stage,
        schedule_sha256=schedule_sha256,
        hardware_preflight_path=str(output_path),
        hardware_preflight_sha256=str(result["hardware_preflight_sha256"]),
        requested_physical_index=requested_physical_index,
        resolved_physical_index=int(selected["resolved_physical_index"]),
        device_uuid=str(selected["device_uuid"]),
        device_model=str(selected["device_model"]),
        driver_version=str(selected["driver_version"]),
        total_memory_bytes=int(selected["total_memory_bytes"]),
        max_concurrent_gpus=1,
        sequential_model_execution=True,
        sequential_unit_execution=True,
        fallback_allowed=False,
        device_switch_allowed=False,
        retry_allowed=False,
        nonce=uuid4().hex,
    )
    _atomic_json(receipt_path, receipt.to_dict(), validator=_validate_receipt_payload)
    result["receipt_path"] = str(receipt_path)
    result["receipt_sha256"] = sha256_file(receipt_path)
    return result


def _rich_preflight_receipt_args(preflight_path: Path, receipt: V4ExecutionReceipt) -> dict[str, object]:
    payload = load_json(preflight_path)
    if payload.get("schema_version") != V4_HARDWARE_PREFLIGHT_SCHEMA_VERSION or payload.get("status") != "PASS":
        raise V4ExecutionError("V4_GPU_PREFLIGHT_NOT_PASS")
    return {
        "project_commit": receipt.project_commit,
        "project_tree": receipt.project_tree,
        "scope": receipt.scope,
        "stage": receipt.stage,
        "schedule_sha256": receipt.schedule_sha256,
        "hardware_preflight_path": preflight_path,
        "hardware_preflight_sha256": receipt.hardware_preflight_sha256,
        "requested_physical_index": receipt.requested_physical_index,
        "visible_gpu_count": payload.get("visible_gpu_count"),
        "active_gpu_count": payload.get("compute_process_count"),
        "lock_active": False,
    }


def validate_rich_preflight_for_receipt(preflight_path: Path, receipt: V4ExecutionReceipt) -> None:
    payload = load_json(preflight_path)
    if payload.get("schema_version") != V4_HARDWARE_PREFLIGHT_SCHEMA_VERSION or payload.get("status") != "PASS":
        raise V4ExecutionError("V4_GPU_PREFLIGHT_NOT_PASS")
    if payload.get("project_commit") != receipt.project_commit or payload.get("project_tree") != receipt.project_tree:
        raise V4ExecutionError("V4_GPU_RECEIPT_GIT_MISMATCH")
    if payload.get("requested_physical_index") != receipt.requested_physical_index or payload.get("resolved_physical_index") != receipt.resolved_physical_index:
        raise V4ExecutionError("V4_GPU_RECEIPT_DEVICE_MISMATCH")
    devices = payload.get("devices")
    if not isinstance(devices, list) or len(devices) != 1:
        raise V4ExecutionError("V4_GPU_RECEIPT_SINGLE_VISIBLE_GPU_REQUIRED")
    device = devices[0]
    if not isinstance(device, Mapping):
        raise V4ExecutionError("V4_GPU_RECEIPT_DEVICE_MISMATCH")
    if device.get("uuid") != receipt.device_uuid or device.get("model") != receipt.device_model or device.get("driver_version") != receipt.driver_version or device.get("total_memory_bytes") != receipt.total_memory_bytes:
        raise V4ExecutionError("V4_GPU_RECEIPT_DEVICE_MISMATCH")
    decision = _evaluate_basic(payload.get("samples", ()))
    if decision["status"] != "PASS":
        raise V4ExecutionError(str(decision["reason_code"]))
    probe_decision = _evaluate_probes(payload.get("model_environment_probes", ()), payload["samples"][-1])
    if probe_decision["status"] != "PASS":
        raise V4ExecutionError(str(probe_decision["reason_code"]))


def _load_receipt(receipt_path: Path, expected_sha256: str | None = None) -> V4ExecutionReceipt:
    if expected_sha256 is not None and sha256_file(receipt_path) != expected_sha256:
        raise V4ExecutionError("V4_RECEIPT_TAMPER")
    return V4ExecutionReceipt.from_mapping(load_json(receipt_path))


def _validate_receipt_contract(receipt: V4ExecutionReceipt) -> None:
    if receipt.explicit_user_selection is not True:
        raise V4ExecutionError("GPU_SELECTION_REQUIRED")
    if receipt.protocol_id != V4_PROTOCOL_ID or receipt.protocol_sha256 != V4_PROTOCOL_SHA256:
        raise V4ExecutionError("V4_GPU_RECEIPT_PROTOCOL_MISMATCH")
    if receipt.scope != SCIENTIFIC_MVE or receipt.stage != SCIENTIFIC_MVE:
        raise V4ExecutionError("V4_GPU_RECEIPT_SCOPE_MISMATCH")
    if receipt.max_concurrent_gpus != 1:
        raise V4ExecutionError("V4_GPU_RECEIPT_SINGLE_GPU_REQUIRED")
    if receipt.sequential_model_execution is not True or receipt.sequential_unit_execution is not True:
        raise V4ExecutionError("V4_GPU_RECEIPT_SEQUENTIAL_REQUIRED")
    if receipt.fallback_allowed is not False or receipt.device_switch_allowed is not False or receipt.retry_allowed is not False:
        raise V4ExecutionError("V4_GPU_RECEIPT_NO_FALLBACK_REQUIRED")
    if receipt.device_model != AUTHORIZED_GPU_MODEL:
        raise V4ExecutionError("V4_GPU_MODEL_NOT_AUTHORIZED")


def _validate_schedule_path(schedule_path: Path, expected_sha256: str) -> None:
    if sha256_file(schedule_path) != expected_sha256:
        raise V4ExecutionError("V4_SCHEDULE_TAMPER")
    schedule = validate_scientific_schedule(parse_scientific_schedule(schedule_path.read_text(encoding="utf-8")))
    if schedule.models != SCIENTIFIC_MODELS or len(schedule.units) != 400:
        raise V4ExecutionError("V4_AUTHORIZATION_SCOPE_EXPANDED")


def _inventory_from_manifest(root: Path, manifest_path: Path) -> tuple[tuple[str, str, str], ...]:
    manifest = load_json(manifest_path)
    if set(manifest) != set(AUTHORIZED_RESOURCE_KEYS):
        raise V4ExecutionError("V4_RESOURCE_INVENTORY_SCHEMA_REQUIRED")
    rows: list[tuple[str, str, str]] = []
    for key in AUTHORIZED_RESOURCE_KEYS:
        value = manifest[key]
        if isinstance(value, str):
            path = _resolve_under_root(root, Path(value), must_exist=True)
            expected_sha = sha256_file(path)
        elif isinstance(value, Mapping):
            path = _resolve_under_root(root, Path(str(value.get("path"))), must_exist=True)
            expected_sha = str(value.get("sha256"))
            if not _is_sha(expected_sha) or sha256_file(path) != expected_sha:
                raise V4ExecutionError("V4_RESOURCE_TAMPER")
        else:
            raise V4ExecutionError("V4_RESOURCE_INVENTORY_SCHEMA_REQUIRED")
        rows.append((key, str(path), expected_sha))
    return tuple(rows)


def _authorized_scope() -> dict[str, object]:
    return {
        "model_set": list(SCIENTIFIC_MODELS),
        "dataset": "DTU",
        "test_scene_ids": list(TEST_SCENE_IDS),
        "state_ids": list(SCIENTIFIC_STATES),
        "lighting_states": list(LIGHTING_STATES),
        "fog_states": list(FOG_STATES),
        "model_independent_states": 200,
        "execution_units": 400,
        "l3_reused_once": True,
        "fog_model": "Koschmieder",
        "severity_family": "beta-only",
        "primary_endpoint": "Pose",
        "supporting_endpoints": ["Fusion", "F-score"],
        "authorized_stop_gpu_seconds": GPU_TARGET_SECONDS,
        "authorized_stop_logical_bytes": BYTE_TARGET,
        "authorized_stop_allocated_bytes": BYTE_TARGET,
        "hard_ceiling_gpu_seconds": GPU_CATASTROPHE_SECONDS,
        "hard_ceiling_logical_bytes": BYTE_CATASTROPHE,
        "hard_ceiling_allocated_bytes": BYTE_CATASTROPHE,
        "forbidden": [
            "UAVLight",
            "v4.1",
            "third model",
            "second corruption family",
            "seed expansion",
            "severity expansion",
            "grid expansion",
            "automatic fallback",
            "parallel GPU execution",
        ],
        "max_concurrency": 1,
        "sequential_models": True,
        "fallback_allowed": False,
        "device_switch_allowed": False,
        "retry_allowed": False,
    }


def create_execution_authorization(
    *,
    root: Path,
    receipt_path: Path,
    resource_inventory_path: Path,
    run_root: Path,
    artifact_root: Path,
    final_evidence_path: Path,
    output_path: Path,
) -> dict[str, object]:
    try:
        output_path.unlink()
    except FileNotFoundError:
        pass
    try:
        output_path.with_name(output_path.name + ".partial").unlink()
    except FileNotFoundError:
        pass
    resolved_root = root.resolve()
    receipt = _load_receipt(receipt_path)
    if receipt.project_commit != IMPLEMENTATION_ANCHOR_COMMIT or receipt.project_tree != IMPLEMENTATION_ANCHOR_TREE:
        raise V4ExecutionError("V4_AUTHORIZATION_STALE_ANCHOR")
    _validate_receipt_contract(receipt)
    if receipt.schedule_sha256 is None:
        raise V4ExecutionError("V4_AUTHORIZATION_SCHEDULE_REQUIRED")
    preflight_path = Path(receipt.hardware_preflight_path)
    if not preflight_path.is_absolute():
        preflight_path = receipt_path.parent / preflight_path
    validate_rich_preflight_for_receipt(preflight_path, receipt)
    _validate_schedule_path(_resolve_under_root(resolved_root, Path(load_json(resource_inventory_path)["scientific_schedule_400"]["path"] if isinstance(load_json(resource_inventory_path).get("scientific_schedule_400"), Mapping) else load_json(resource_inventory_path)["scientific_schedule_400"]), must_exist=True), receipt.schedule_sha256)
    inventory = _inventory_from_manifest(resolved_root, resource_inventory_path)
    authorized = V4ExecutionAuthorization(
        receipt_path=str(receipt_path.resolve()),
        receipt_sha256=sha256_file(receipt_path),
        hardware_preflight_path=str(preflight_path.resolve()),
        hardware_preflight_sha256=sha256_file(preflight_path),
        implementation_commit=receipt.project_commit,
        implementation_tree=receipt.project_tree,
        protocol_id=V4_PROTOCOL_ID,
        protocol_sha256=V4_PROTOCOL_SHA256,
        schedule_sha256=receipt.schedule_sha256,
        resource_inventory=inventory,
        root=str(resolved_root),
        run_root=str(_resolve_under_root(resolved_root, run_root)),
        artifact_root=str(_resolve_under_root(resolved_root, artifact_root)),
        final_evidence_path=str(_resolve_under_root(resolved_root, final_evidence_path)),
        finalizer=AUTHORIZED_FINALIZER,
    )
    payload = authorized.to_dict()
    _atomic_json(output_path, payload, validator=_validate_authorization_payload)
    return {"status": "PASS", "authorization_path": str(output_path), "authorization_sha256": sha256_file(output_path)}


def validate_execution_authorization(path: Path) -> dict[str, object]:
    payload = load_json(path)
    if payload.get("schema_version") != V4_EXECUTION_AUTHORIZATION_SCHEMA_VERSION:
        raise V4ExecutionError("V4_AUTHORIZATION_SCHEMA_REQUIRED")
    expected_sha = payload.get("authorization_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "authorization_sha256"}
    if not _is_sha(expected_sha) or _sha_json(unsigned) != expected_sha:
        raise V4ExecutionError("V4_AUTHORIZATION_TAMPER")
    if payload.get("implementation_commit") != IMPLEMENTATION_ANCHOR_COMMIT or payload.get("implementation_tree") != IMPLEMENTATION_ANCHOR_TREE:
        raise V4ExecutionError("V4_AUTHORIZATION_STALE_ANCHOR")
    if payload.get("protocol_id") != V4_PROTOCOL_ID or payload.get("protocol_sha256") != V4_PROTOCOL_SHA256:
        raise V4ExecutionError("V4_AUTHORIZATION_PROTOCOL_MISMATCH")
    if payload.get("finalizer") != AUTHORIZED_FINALIZER:
        raise V4ExecutionError("V4_AUTHORIZATION_FINALIZER_MISMATCH")
    root = Path(str(payload.get("root"))).resolve()
    _resolve_under_root(root, Path(str(payload.get("run_root"))), must_exist=True)
    _resolve_under_root(root, Path(str(payload.get("artifact_root"))), must_exist=True)
    _resolve_under_root(root, Path(str(payload.get("final_evidence_path"))))
    scope = payload.get("authorized_scope")
    if scope != _authorized_scope():
        raise V4ExecutionError("V4_AUTHORIZATION_SCOPE_EXPANDED")
    receipt_path = Path(str(payload.get("receipt_path")))
    receipt = _load_receipt(receipt_path, str(payload.get("receipt_sha256")))
    if receipt.project_commit != payload.get("implementation_commit") or receipt.project_tree != payload.get("implementation_tree") or receipt.schedule_sha256 != payload.get("schedule_sha256"):
        raise V4ExecutionError("V4_AUTHORIZATION_RECEIPT_MISMATCH")
    _validate_receipt_contract(receipt)
    preflight_path = Path(str(payload.get("hardware_preflight_path")))
    if sha256_file(preflight_path) != payload.get("hardware_preflight_sha256"):
        raise V4ExecutionError("V4_GPU_PREFLIGHT_TAMPER")
    validate_rich_preflight_for_receipt(preflight_path, receipt)
    for row in payload.get("resource_inventory", ()):
        if not isinstance(row, Mapping) or row.get("key") not in AUTHORIZED_RESOURCE_KEYS:
            raise V4ExecutionError("V4_RESOURCE_INVENTORY_SCHEMA_REQUIRED")
        resource_path = _resolve_under_root(root, Path(str(row.get("path"))), must_exist=True)
        digest = row.get("sha256")
        if not _is_sha(digest) or sha256_file(resource_path) != digest:
            raise V4ExecutionError("V4_RESOURCE_TAMPER")
    if [row.get("key") for row in payload.get("resource_inventory", ())] != list(AUTHORIZED_RESOURCE_KEYS):
        raise V4ExecutionError("V4_RESOURCE_INVENTORY_SCHEMA_REQUIRED")
    return {"status": "PASS", "authorization_path": str(path), "authorization_sha256": sha256_file(path)}
