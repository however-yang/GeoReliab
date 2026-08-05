
from __future__ import annotations

import json
from pathlib import Path
import signal

import pytest

import georeliab_mve.v4_attempt05_recovery as recovery
from georeliab_mve.v4_attempt05_recovery import (
    Attempt05RecoveryError,
    FailureEnvelope,
    HeartbeatSupervisor,
    ImmutableUnionManifest,
    install_signal_exit_handlers,
    JournaledLedger,
    UnitTransactionStore,
    atomic_write_bytes,
    attempt06_gate,
    build_monitor_status,
    build_immutable_union_manifest,
    build_missing_unit_schedule,
    checkpoint_due,
    identity_only_audit,
    read_hash_chain,
    reconcile_unit_transaction,
    repair_torn_ledger_tail,
    write_forensic_bundle,
)


def _schedule(count: int) -> list[tuple[str, int, str]]:
    return [("model", index, "L3") for index in range(count)]


def _append_valid_unit(ledger: JournaledLedger, run_root: Path, key: tuple[str, int, str]) -> Path:
    model, scene, state = key
    record = run_root / "records" / f"{scene}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_bytes(b'{"opaque_scientific_payload":{"auc":0.123}}\n')
    digest = __import__("hashlib").sha256(record.read_bytes()).hexdigest()
    payload = {
        "model_id": model,
        "scene_id": scene,
        "state_id": state,
        "task_record_sha256": digest,
        "projection_path": str(record.relative_to(run_root)),
        "adapter_sha256": "adapter",
        "config_sha256": "config",
        "split_sha256": "split",
        "corruption_sha256": "corruption",
        "gpu_uuid": "GPU-0",
        "physical_gpu_index": 0,
        "tooling_commit": "commit",
        "tooling_tree": "tree",
    }
    ledger.append("CANONICAL_RECORD_PROJECTION", payload)
    completion_payload = dict(payload)
    completion_payload.pop("projection_path", None)
    ledger.append("SCIENTIFIC_UNIT_COMPLETE", completion_payload)
    return record


def test_failure_envelope_roundtrip_and_traceback_tamper() -> None:
    try:
        raise ValueError("fault")
    except ValueError as exc:
        envelope = FailureEnvelope.from_exception(
            exc,
            attempt_id="attempt-06",
            stage="rename",
            unit_key=("m", 1, "L3"),
            reason_code="V4_ATTEMPT06_RENAME_FAILED",
        )
    restored = FailureEnvelope.from_mapping(envelope.to_dict())
    assert restored.reason_code == "V4_ATTEMPT06_RENAME_FAILED"
    assert restored.unit_key == "m|1|L3"
    tampered = envelope.to_dict()
    tampered["traceback"] = "other"
    with pytest.raises(Attempt05RecoveryError, match="TRACE_TAMPER"):
        FailureEnvelope.from_mapping(tampered)


def test_atomic_write_fault_leaves_partial_and_success_has_no_partial(tmp_path: Path) -> None:
    target = tmp_path / "unit" / "canonical.json"
    def fail(stage: str) -> None:
        if stage == "file_fsync":
            raise OSError("injected fsync")
    with pytest.raises(OSError):
        atomic_write_bytes(target, b"payload", fault_injector=fail)
    assert list(target.parent.glob("*.partial"))
    atomic_write_bytes(target, b"payload")
    assert target.read_bytes() == b"payload"
    # A prior failed transaction remains forensic evidence by design.
    assert list(target.parent.glob("*.partial"))


def test_journaled_ledger_recovers_pending_and_rejects_torn_tail(tmp_path: Path) -> None:
    ledger = JournaledLedger(tmp_path / "ledger.jsonl")
    def fail(stage: str) -> None:
        if stage == "ledger_append":
            raise OSError("injected append")
    with pytest.raises(OSError):
        ledger.append("UNIT", {"unit_key": "m|1|L3"}, fault_injector=fail)
    assert list(ledger.pending_root.glob("*.partial"))
    repaired = ledger.reconcile_pending()
    assert len(repaired) == 1
    assert len(read_hash_chain(ledger.ledger_path)) == 1
    with ledger.ledger_path.open("ab") as handle:
        handle.write(b"torn")
    removed = repair_torn_ledger_tail(ledger.ledger_path)
    assert removed == 4
    assert len(read_hash_chain(ledger.ledger_path)) == 1


def test_unit_transaction_is_exactly_once_and_recoverable(tmp_path: Path) -> None:
    ledger = JournaledLedger(tmp_path / "ledger.jsonl")
    store = UnitTransactionStore(tmp_path / "units", attempt_id="attempt-06", ledger=ledger)
    receipt = store.prepare_unit(
        idempotency_key="attempt-06:m|1|L3",
        unit_key=("m", 1, "L3"),
        stage="prediction",
        canonical_dir=tmp_path / "units" / "m-1-L3",
        files={"prediction.bin": b"prediction"},
        ledger_events=(
            {
                "event_type": "UNIT_COMPLETE",
                "payload": {"model_id": "m", "scene_id": 1, "state_id": "L3"},
            },
        ),
    )
    committed = store.commit_unit(receipt)
    repeated = store.commit_unit(committed)
    assert committed.state == "LEDGER_COMMITTED"
    assert repeated.state == "LEDGER_COMMITTED"
    assert len(read_hash_chain(tmp_path / "ledger.jsonl")) == 1
    assert (tmp_path / "units" / "m-1-L3").is_dir()
    classified = reconcile_unit_transaction(
        canonical_dir=tmp_path / "units" / "m-1-L3",
        ledger_path=tmp_path / "ledger.jsonl",
        idempotency_key="attempt-06:m|1|L3",
    )
    assert classified["state"] == "COMPLETE"


def test_reconciler_distinguishes_incomplete_orphan_and_torn_tail(tmp_path: Path) -> None:
    ledger = JournaledLedger(tmp_path / "ledger.jsonl")
    store = UnitTransactionStore(tmp_path / "units", attempt_id="attempt-06", ledger=ledger)
    receipt = store.prepare_unit(
        idempotency_key="attempt-06:m|1|L3",
        unit_key=("m", 1, "L3"),
        stage="prediction",
        canonical_dir=tmp_path / "units" / "m-1-L3",
        files={"prediction.bin": b"prediction"},
    )
    assert reconcile_unit_transaction(
        canonical_dir=tmp_path / "units" / "m-1-L3",
        ledger_path=tmp_path / "ledger.jsonl",
        idempotency_key="attempt-06:m|1|L3",
    )["state"] == "INCOMPLETE"
    store.promote_unit(receipt)
    assert reconcile_unit_transaction(
        canonical_dir=tmp_path / "units" / "m-1-L3",
        ledger_path=tmp_path / "ledger.jsonl",
        idempotency_key="attempt-06:m|1|L3",
    )["state"] == "INCOMPLETE"
    orphan = tmp_path / "units" / "orphan"
    orphan.mkdir()
    assert reconcile_unit_transaction(
        canonical_dir=orphan,
        ledger_path=tmp_path / "ledger.jsonl",
        idempotency_key="orphan",
    )["state"] == "ORPHAN"
    with (tmp_path / "ledger.jsonl").open("ab") as handle:
        handle.write(b"torn")
    assert reconcile_unit_transaction(
        canonical_dir=tmp_path / "units" / "m-1-L3",
        ledger_path=tmp_path / "ledger.jsonl",
        idempotency_key="attempt-06:m|1|L3",
    )["state"] == "TORN_LEDGER_TAIL"


def test_identity_only_audit_allows_verified_partial_but_blocks_hash_mismatch(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    ledger = JournaledLedger(tmp_path / "ledger.jsonl")
    schedule = _schedule(3)
    for key in schedule[:2]:
        _append_valid_unit(ledger, run_root, key)
    eligible = identity_only_audit(
        attempt_id="attempt-05",
        schedule_keys=schedule,
        ledger_path=tmp_path / "ledger.jsonl",
        run_root=run_root,
        schedule_sha256="fd6bfd",
        expected_schedule_sha256="fd6bfd",
        input_closure_schedule_sha256="fd6bfd",
        expected_verified_count=2,
    )
    assert eligible.verdict == "V4_ATTEMPT05_PARTIAL_CORPUS_RESUME_ELIGIBLE"
    assert len(eligible.verified_unit_keys) == 2
    assert len(eligible.missing_unit_keys) == 1
    assert eligible.corpus_classification == "IMMUTABLE_PARTIAL_PREDICTION_CORPUS"
    report = json.dumps(eligible.to_dict(), sort_keys=True)
    assert "0.123" not in report
    blocked = identity_only_audit(
        attempt_id="attempt-05",
        schedule_keys=schedule,
        ledger_path=tmp_path / "ledger.jsonl",
        run_root=run_root,
        schedule_sha256="fd6bfd",
        expected_schedule_sha256="70fec",
        input_closure_schedule_sha256="70fec",
        expected_verified_count=2,
    )
    assert blocked.verdict == "V4_ATTEMPT05_PARTIAL_CORPUS_NOT_RESUMABLE"


def test_identity_audit_strictly_checks_binding_evidence_without_metrics(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    ledger = JournaledLedger(tmp_path / "ledger.jsonl")
    key = _schedule(1)[0]
    _append_valid_unit(ledger, run_root, key)
    eligible = identity_only_audit(
        attempt_id="attempt-05",
        schedule_keys=[key],
        ledger_path=tmp_path / "ledger.jsonl",
        run_root=run_root,
        schedule_sha256="x",
        expected_verified_count=1,
        require_binding_evidence=True,
    )
    assert eligible.verdict == "V4_ATTEMPT05_PARTIAL_CORPUS_RESUME_ELIGIBLE"
    assert eligible.checks["binding_evidence_present"] is True

    stripped = JournaledLedger(tmp_path / "stripped-ledger.jsonl")
    record = run_root / "records" / "0.json"
    digest = __import__("hashlib").sha256(record.read_bytes()).hexdigest()
    base = {
        "model_id": "model",
        "scene_id": 0,
        "state_id": "L3",
        "task_record_sha256": digest,
        "projection_path": str(record.relative_to(run_root)),
    }
    stripped.append("CANONICAL_RECORD_PROJECTION", base)
    complete = dict(base)
    complete.pop("projection_path")
    stripped.append("SCIENTIFIC_UNIT_COMPLETE", complete)
    blocked = identity_only_audit(
        attempt_id="attempt-05",
        schedule_keys=[key],
        ledger_path=tmp_path / "stripped-ledger.jsonl",
        run_root=run_root,
        schedule_sha256="x",
        expected_verified_count=1,
        require_binding_evidence=True,
    )
    assert blocked.verdict == "V4_ATTEMPT05_PARTIAL_CORPUS_NOT_RESUMABLE"
    assert blocked.checks["binding_evidence_missing_units"][0]["unit_key"] == "model|0|L3"
    assert "adapter" in blocked.checks["binding_evidence_missing_units"][0]["fields"]


def test_identity_audit_detects_duplicate_and_partial(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    ledger = JournaledLedger(tmp_path / "ledger.jsonl")
    key = _schedule(1)[0]
    _append_valid_unit(ledger, run_root, key)
    ledger.append(
        "CANONICAL_RECORD_PROJECTION",
        {"model_id": "model", "scene_id": 0, "state_id": "L3", "projection_path": "records/0.json", "task_record_sha256": "bad"},
    )
    (run_root / "orphan.partial").write_bytes(b"x")
    result = identity_only_audit(
        attempt_id="attempt-05",
        schedule_keys=[key],
        ledger_path=tmp_path / "ledger.jsonl",
        run_root=run_root,
        schedule_sha256="x",
        expected_verified_count=1,
    )
    assert result.verdict == "V4_ATTEMPT05_PARTIAL_CORPUS_NOT_RESUMABLE"
    assert result.checks["duplicate_units"] == ["model|0|L3"]
    assert result.checks["partials_or_temps"] == ["orphan.partial"]


def test_missing_schedule_union_and_attempt06_gate(tmp_path: Path) -> None:
    schedule = _schedule(3)
    run_root = tmp_path / "run"
    ledger = JournaledLedger(tmp_path / "ledger.jsonl")
    _append_valid_unit(ledger, run_root, schedule[0])
    eligibility = identity_only_audit(
        attempt_id="attempt-05",
        schedule_keys=schedule,
        ledger_path=tmp_path / "ledger.jsonl",
        run_root=run_root,
        schedule_sha256="x",
        expected_verified_count=1,
    )
    assert build_missing_unit_schedule(schedule, eligibility) == ("model|1|L3", "model|2|L3")
    union = build_immutable_union_manifest(
        schedule_sha256="x",
        schedule_keys=schedule,
        source_units={"attempt-05": [schedule[0]], "attempt-06": schedule[1:]},
    )
    assert isinstance(union, ImmutableUnionManifest)
    assert union.expected_count == 3
    decision = attempt06_gate(
        eligibility=eligibility,
        root_cause_resolved=True,
        fault_injection_passed=True,
        gpu_smoke_passed=True,
        explicit_authorization=False,
    )
    assert decision["status"] == "V4_ATTEMPT06_BLOCKED_RECOVERY_GATE"
    assert decision["launch_performed"] is False
    closed = attempt06_gate(
        eligibility=eligibility,
        root_cause_resolved=True,
        fault_injection_passed=True,
        gpu_smoke_passed=True,
        explicit_authorization=True,
        missing_schedule_keys=("model|1|L3", "model|2|L3"),
        expected_total_count=3,
        expected_missing_count=2,
    )
    assert closed["status"] == "V4_ATTEMPT06_AUTHORIZATION_READY"
    assert closed["launch_performed"] is False


def test_forensic_bundle_and_supervisor_are_separate_from_source(tmp_path: Path) -> None:
    source = tmp_path / "source.log"
    source.write_text("raw log", encoding="utf-8")
    run_root = tmp_path / "run"
    ledger = JournaledLedger(tmp_path / "ledger.jsonl")
    key = _schedule(1)[0]
    _append_valid_unit(ledger, run_root, key)
    eligibility = identity_only_audit(
        attempt_id="attempt-05",
        schedule_keys=[key],
        ledger_path=tmp_path / "ledger.jsonl",
        run_root=run_root,
        schedule_sha256="x",
        expected_verified_count=1,
    )
    forensic = tmp_path / "forensics"
    outputs = write_forensic_bundle(
        forensic,
        source_paths={"raw-log": source, "run-root": run_root},
        eligibility=eligibility,
        postmortem={"root_cause": "V4_ATTEMPT05_ROOT_CAUSE_UNOBSERVABLE_DUE_TO_EXCEPTION_COLLAPSE"},
    )
    assert (forensic / "source-manifest.json").is_file()
    assert outputs["resume-eligibility.json"].is_file()
    assert source.read_text(encoding="utf-8") == "raw log"
    supervisor = HeartbeatSupervisor(tmp_path / "heartbeat.json", attempt_id="attempt-06")
    supervisor.beat(stage="idle")
    assert supervisor.heartbeat_path.is_file()
    supervisor.write_exit(
        receipt_path=tmp_path / "exit-receipt.json",
        return_code=1,
        signal_number=None,
        heartbeat_age_seconds=0.1,
    )
    assert (tmp_path / "exit-receipt.json").is_file()


def test_signal_handler_writes_terminal_and_supervisor_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = JournaledLedger(tmp_path / "ledger.jsonl")
    supervisor = HeartbeatSupervisor(tmp_path / "heartbeat.json", attempt_id="attempt-06")
    registered: dict[int, object] = {}

    def fake_signal(number: int, handler: object) -> object:
        registered[number] = handler
        return None

    monkeypatch.setattr(recovery.signal, "signal", fake_signal)
    install_signal_exit_handlers(
        attempt_id="attempt-06",
        ledger=ledger,
        supervisor=supervisor,
        exit_receipt_path=tmp_path / "exit-receipt.json",
        stage_getter=lambda: "prediction",
        unit_key_getter=lambda: ("m", 1, "L3"),
    )
    handler = registered[int(signal.SIGTERM)]
    with pytest.raises(SystemExit) as raised:
        handler(signal.SIGTERM, None)  # type: ignore[misc]
    assert raised.value.code == 128 + int(signal.SIGTERM)
    assert read_hash_chain(tmp_path / "ledger.jsonl")[0]["event_type"] == "ATTEMPT_TERMINAL"
    exit_payload = json.loads((tmp_path / "exit-receipt.json").read_text(encoding="utf-8"))
    assert exit_payload["scientific_result"] == "NO_SCIENTIFIC_RESULT"


def test_hourly_monitor_status_is_identity_and_budget_only() -> None:
    status = build_monitor_status(
        attempt05_valid_completed=199,
        attempt06_valid_completed=1,
        attempt06_elapsed_seconds=3600.0,
        cumulative_materialization_elapsed_seconds=7200.0,
        gpu0_owners=("pid-2", "pid-1"),
        gpu1_owners=(),
        cumulative_gpu_inference_seconds=36 * 3600,
        cumulative_storage_bytes=151 * 1024**3,
    )
    assert status["external_monitor_interval_seconds"] == 3600
    assert status["internal_heartbeat_interval_seconds"] == 60
    assert status["overall_progress"] == 0.5
    assert status["gpu0_owners"] == ["pid-1", "pid-2"]
    assert status["gpu1_owners"] == []
    assert status["budget_status"]["gpu_target_exceeded"] is True
    assert status["budget_status"]["gpu_hard_ceiling_exceeded"] is False
    assert status["scientific_result"] == "NO_SCIENTIFIC_RESULT"


def test_checkpoint_interval() -> None:
    assert checkpoint_due(5)
    assert checkpoint_due(10)
    assert not checkpoint_due(6)
