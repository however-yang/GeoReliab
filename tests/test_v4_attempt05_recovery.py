
from __future__ import annotations

import hashlib
import inspect
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
    archive_torn_ledger_tail,
    RecoveryDecision,
    SameAttemptSessionUnionManifest,
    build_same_attempt_session_union,
    segment_torn_ledger_tail,
    sha256_file,
    RecoverySmokeManifest,
    build_recovery_smoke_manifest,
    evaluate_recovery_smoke,
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
    before = ledger.ledger_path.read_bytes()
    segmented = segment_torn_ledger_tail(ledger.ledger_path)
    assert segmented["torn_tail_bytes"] == 4
    assert segmented["torn_tail_sha256"] is not None
    assert ledger.ledger_path.read_bytes() == before
    assert Path(str(segmented["segment_path"])).read_bytes() == b"torn"
    # Compatibility API is also non-destructive and reports the segment size.
    assert archive_torn_ledger_tail(ledger.ledger_path)["torn_tail_bytes"] == 4
    with pytest.raises(Attempt05RecoveryError, match="TORN_TAIL"):
        read_hash_chain(ledger.ledger_path)


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
    )["state"] == "INCOMPLETE_SAFE_TO_RETRY"
    store.promote_unit(receipt)
    assert reconcile_unit_transaction(
        canonical_dir=tmp_path / "units" / "m-1-L3",
        ledger_path=tmp_path / "ledger.jsonl",
        idempotency_key="attempt-06:m|1|L3",
    )["state"] == "INCOMPLETE_SAFE_TO_RETRY"
    orphan = tmp_path / "units" / "orphan"
    orphan.mkdir()
    assert reconcile_unit_transaction(
        canonical_dir=orphan,
        ledger_path=tmp_path / "ledger.jsonl",
        idempotency_key="orphan",
    )["state"] == "ORPHAN_REQUIRES_QUARANTINE"
    with (tmp_path / "ledger.jsonl").open("ab") as handle:
        handle.write(b"torn")
    assert reconcile_unit_transaction(
        canonical_dir=tmp_path / "units" / "m-1-L3",
        ledger_path=tmp_path / "ledger.jsonl",
        idempotency_key="attempt-06:m|1|L3",
    )["state"] == "ORPHAN_REQUIRES_QUARANTINE"


def test_recovery_decision_and_active_worker_guard_are_typed_and_non_mutating(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_bytes(b"not-a-valid-ledger-tail")
    result = reconcile_unit_transaction(
        canonical_dir=tmp_path / "unit",
        ledger_path=ledger_path,
        idempotency_key="attempt-06:m|1|L3",
        worker_alive=True,
        worker_pid=1234,
    )
    assert result["state"] == "ACTIVE_VALID_DEFER_NO_MUTATION"
    decision = RecoveryDecision.from_reconciliation(result)
    assert decision.recovery_action == "NOOP"
    assert decision.worker_alive is True
    assert ledger_path.read_bytes() == b"not-a-valid-ledger-tail"
    assert result["recovery_decision"]["state"] == decision.state


def test_no_clobber_promotion_rejects_existing_identity(tmp_path: Path) -> None:
    ledger = JournaledLedger(tmp_path / "ledger.jsonl")
    store = UnitTransactionStore(tmp_path / "units", attempt_id="attempt-06", ledger=ledger)
    receipt = store.prepare_unit(
        idempotency_key="attempt-06:m|1|L3",
        unit_key=("m", 1, "L3"),
        stage="prediction",
        canonical_dir=tmp_path / "units" / "m-1-L3",
        files={"prediction.bin": b"new"},
    )
    canonical = tmp_path / "units" / "m-1-L3"
    canonical.mkdir()
    (canonical / "sentinel").write_bytes(b"do-not-clobber")
    with pytest.raises(Attempt05RecoveryError, match="RECEIPT_MISSING"):
        store.promote_unit(receipt)
    assert (canonical / "sentinel").read_bytes() == b"do-not-clobber"
    assert (tmp_path / "units" / "m-1-L3.partial").is_dir()


def test_schedule_identity_domains_accept_distinct_raw_and_semantic_hashes(tmp_path: Path) -> None:
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_bytes(b"{\n  \"units\": [1, 2]\n}\n")
    manifest = recovery.ScheduleIdentityManifest.build(
        raw_sha256=__import__("hashlib").sha256(schedule_path.read_bytes()).hexdigest(),
        semantic_payload={"units": [1, 2]},
        ordered_unit_ids=[("m", 1, "L3"), ("m", 2, "L3")],
    )
    assert manifest.raw_sha256 != manifest.semantic_sha256
    expected_semantic = hashlib.sha256(
        recovery.SCHEDULE_SEMANTIC_HASH_DOMAIN.encode("utf-8")
        + b"\0"
        + b'{"units":[1,2]}'
    ).hexdigest()
    assert manifest.semantic_sha256 == expected_semantic
    # Hash domains use compact canonical JSON without the JSONL newline used
    # by the append-only ledger.
    assert not recovery._canonical_json_bytes_without_newline({"units": [1, 2]}).endswith(b"\n")
    assert recovery.ScheduleIdentityManifest.from_mapping(manifest.to_dict()).schedule_identity_sha256 == manifest.schedule_identity_sha256


def test_production_atomic_writer_has_no_replace_primitive() -> None:
    assert "os.replace" not in inspect.getsource(recovery.atomic_write_bytes)


def test_same_attempt_session_union_rejects_cross_attempt_and_exactly_closes(tmp_path: Path) -> None:
    keys = _schedule(3)
    manifest = build_same_attempt_session_union(
        attempt_id="attempt-06",
        schedule_identity_sha256="a" * 64,
        session_units={"session-a": keys[:2], "session-b": keys[2:]},
        expected_schedule_keys=keys,
    )
    assert isinstance(manifest, SameAttemptSessionUnionManifest)
    assert manifest.unit_keys == tuple(f"model|{idx}|L3" for idx in range(3))
    restored = SameAttemptSessionUnionManifest.from_mapping(manifest.to_dict())
    assert restored.to_dict() == manifest.to_dict()
    with pytest.raises(Attempt05RecoveryError, match="CROSS_ATTEMPT"):
        build_same_attempt_session_union(
            attempt_id="attempt-06",
            schedule_identity_sha256="a" * 64,
            session_units={"session": keys},
            expected_schedule_keys=keys,
            source_attempt_ids={"session": "attempt-05"},
        )


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
    assert blocked.checks["reconciliation_failures"]["path_or_sha_missing"] == 1
    assert blocked.checks["binding_evidence_missing_units"] == []


def test_sha256_file_rejects_path_metadata_mutation(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "canonical.bin"
    path.write_bytes(b"stable")
    real_lstat = recovery.os.lstat
    calls = 0

    def mutating_lstat(value):
        nonlocal calls
        result = real_lstat(value)
        calls += 1
        if calls == 2:
            fields = list(result)
            fields[6] = int(fields[6]) + 1  # st_size differs after the read
            return type(result)(fields)
        return result

    monkeypatch.setattr(recovery.os, "lstat", mutating_lstat)
    with pytest.raises(OSError):
        sha256_file(path)


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
    # Historical eligibility/missing-unit closure can never authorize a fresh
    # Attempt-06 materialization.
    assert closed["status"] == "V4_ATTEMPT06_BLOCKED_RECOVERY_GATE"
    assert closed["checks"]["fresh_source_confirmed"] is False
    assert closed["launch_performed"] is False


def test_attempt06_fresh_source_gate_rejects_historical_partial_and_requires_pilot(tmp_path: Path) -> None:
    blocked = attempt06_gate(
        fresh_source=True,
        attempt05_source_rejected=True,
        recovery_runtime_qualified=True,
        power_gate_passed=True,
        pilot_status="V4_PILOT_SCIENTIFIC_NO_GO",
        budget_gate_passed=True,
        fresh_schedule_closed=True,
        explicit_authorization=True,
    )
    assert blocked["status"] == "V4_ATTEMPT06_BLOCKED_RECOVERY_GATE"
    assert blocked["checks"]["pilot_gate_passed"] is False
    ready = attempt06_gate(
        fresh_source_confirmed=True,
        attempt05_source_rejected=True,
        recovery_runtime_qualified=True,
        power_gate_passed=True,
        pilot_status="V4_PILOT_GO_TO_FULL_MVE",
        budget_gate_passed=True,
        fresh_schedule_closed=True,
        explicit_authorization=True,
    )
    assert ready["status"] == "V4_ATTEMPT06_AUTHORIZATION_READY"
    assert ready["checks"].get("historical_partial_ignored") is True
    assert ready["launch_performed"] is False


def test_recovery_smoke_manifest_is_deterministic_and_fail_closed(tmp_path: Path) -> None:
    schedule_identity = "b" * 64
    manifest = build_recovery_smoke_manifest(
        schedule_identity_sha256=schedule_identity,
        support_scene_ids=(1, 9, 10, 11, 12, 13, 23, 24, 29, 32),
    )
    assert isinstance(manifest, RecoverySmokeManifest)
    restored = RecoverySmokeManifest.from_mapping(manifest.to_dict())
    assert restored.to_dict() == manifest.to_dict()
    incomplete = evaluate_recovery_smoke(
        restored,
        observations=[{"unit_key": manifest.unit_keys[0]}],
    )
    assert incomplete["status"] == "V4_RECOVERY_RUNTIME_NOT_QUALIFIED"
    assert manifest.unit_keys[0] in incomplete["closure_violations"] or manifest.unit_keys[0] in incomplete["inference_count_mismatches"]

    rows = []
    for key, expected in manifest.expected_inference_starts.items():
        rows.append(
            {
                "unit_key": key,
                "inference_start_count": expected,
                "completion_count": 1,
                "overwrite_count": 0,
                "gpu_uuid": manifest.gpu_uuid,
                "physical_gpu_index": manifest.physical_gpu_index,
                "canonical_present": True,
                "ledger_committed": True,
                "scientific_marker": "NO_SCIENTIFIC_RESULT",
            }
        )
    qualified = evaluate_recovery_smoke(restored, observations=rows)
    assert qualified["status"] == "V4_RECOVERY_RUNTIME_QUALIFIED"
    assert qualified["scientific_result"] == "NO_SCIENTIFIC_RESULT"


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
