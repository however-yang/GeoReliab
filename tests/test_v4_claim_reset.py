from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

from georeliab_mve.v4_science_lock import (
    GEORELIAB_V4_PROTOCOL_READY,
    GPU_SELECTION_REQUIRED,
    NO_SCIENTIFIC_RESULT,
    SUPERSEDED_BY_PRIOR_ART_CHANGE,
    V1_IMMUTABLE_SCIENCE_FILES,
    V4_BASE_COMMIT,
    V4_ARTIFACT_RECORD_SCHEMA_VERSION,
    V4_LOCKED_SCIENCE_FILES,
    V4_PROTOCOL_ID,
    V4_PROTOCOL_SHA256,
    V4_SCIENTIFIC_BUNDLE_SCHEMA_VERSION,
    V4ScienceLockError,
    load_v4_protocol,
    v4_protocol_provenance,
    v4_record_origin,
    validate_v4_science_lock,
    validate_v4_scientific_bundle,
    validate_v4_scientific_bundle_structure,
)


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def _copy_locked_tree(target: Path) -> None:
    for relative in (*V4_LOCKED_SCIENCE_FILES, *V1_IMMUTABLE_SCIENCE_FILES):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE_ROOT / relative, destination)


def _v4_bundle(artifact: dict[str, object]) -> dict[str, object]:
    provenance = v4_protocol_provenance(SOURCE_ROOT)
    return {
        "schema_version": V4_SCIENTIFIC_BUNDLE_SCHEMA_VERSION,
        "project_line": "v4",
        "scientific_validity": "SCIENTIFIC",
        "protocol_provenance": provenance,
        "artifacts": [
            {
                "project_line": "v4",
                "protocol_provenance": provenance,
                "artifact": artifact,
            }
        ],
    }


def _v4_record(
    *,
    data: dict[str, object] | None = None,
    record_kind: str = "artifact",
) -> dict[str, object]:
    return {
        "schema_version": V4_ARTIFACT_RECORD_SCHEMA_VERSION,
        "record_kind": record_kind,
        "origin": v4_record_origin(SOURCE_ROOT),
        "source_uri": "file:///new-v4/source.json",
        "source_sha256": "a" * 64,
        "source_schema_version": "1.1",
        "data": data or {"derived_under_v4": True},
    }


def test_v4_science_lock_freezes_claim_reset_and_initial_status() -> None:
    report = validate_v4_science_lock(SOURCE_ROOT)

    assert report["status"] == GEORELIAB_V4_PROTOCOL_READY
    assert report["execution_status"] == GPU_SELECTION_REQUIRED
    assert report["scientific_result_status"] == NO_SCIENTIFIC_RESULT
    assert report["v1_status"] == SUPERSEDED_BY_PRIOR_ART_CHANGE
    assert report["v1_scientific_result_status"] == NO_SCIENTIFIC_RESULT
    assert report["protocol_id"] == V4_PROTOCOL_ID
    assert report["protocol_sha256"] == V4_PROTOCOL_SHA256
    assert {
        row["path"]: row["sha256"] for row in report["files"]
    } == dict(V4_LOCKED_SCIENCE_FILES)
    assert {
        row["path"]: row["sha256"] for row in report["unchanged_v1_files"]
    } == dict(V1_IMMUTABLE_SCIENCE_FILES)


def test_v4_protocol_freezes_claims_routes_scope_gates_budgets_and_dates() -> None:
    protocol = load_v4_protocol(SOURCE_ROOT)

    assert protocol["title"] == (
        "GeoReliab: Ranking Is Not Warning — Paired Counterfactual Reliability "
        "and Task Transfer in Geometric Foundation Models"
    )
    assert protocol["core_proposition"] == (
        "Point-level confidence may rank geometry errors correctly within a "
        "fixed condition, yet fail to warn when the same scene becomes harder "
        "and fail to transfer into a reliable warning of relative-pose failure."
    )
    assert protocol["main_line"] == "v4"
    assert protocol["sole_main_line"] is True
    assert protocol["protocol_status"] == GEORELIAB_V4_PROTOCOL_READY
    assert protocol["execution_status"] == GPU_SELECTION_REQUIRED
    assert protocol["scientific_result_status"] == NO_SCIENTIFIC_RESULT
    assert protocol["official_deadline"] == "TBA"
    assert protocol["route"] == {
        "v4": "SOLE_CVPR_2027_MAIN_LINE",
        "v1": SUPERSEDED_BY_PRIOR_ART_CHANGE,
        "v1_scientific_result": NO_SCIENTIFIC_RESULT,
        "v1_fail_classification_forbidden": True,
        "geometry_causal_audit": "INDEPENDENT_BACKLOG_NON_BLOCKING",
        "deformable_world": "INDEPENDENT_BACKLOG_NON_BLOCKING",
    }
    assert protocol["claims"] == {
        "c1": (
            "Paired counterfactual protocol: hold scene, geometry, cameras and "
            "view identities fixed while changing only illumination or "
            "synthetic fog."
        ),
        "c2": (
            "Ranking–warning separation: distinguish static point-level "
            "ranking from cross-condition task-level warning."
        ),
        "c3": (
            "Point-to-pose transfer: measure whether native point confidence "
            "transfers to warning relative-pose failure."
        ),
        "c4": (
            "Reliability boundary map: identify model-, task- and "
            "condition-specific silent-failure boundaries."
        ),
        "c5": (
            "Conditional method contribution: only after MVE GO, create a "
            "separate protocol for View-Set Admission Control."
        ),
    }
    assert protocol["claim_boundaries"]["excluded"] == [
        "first GFM UQ",
        "first corruption benchmark",
        "universal confidence failure",
        "conformal guarantees",
        "non-rejection means robustness",
        "synthetic-only extrapolation to real deployment",
    ]

    novelty = {item["work"]: item for item in protocol["novelty_boundary"]}
    assert novelty["Trust3R"]["url"] == "https://arxiv.org/abs/2605.19539"
    assert novelty["VGGT-UQ"]["secondary_url"] == (
        "https://doi.org/10.5194/isprs-annals-XI-2-2026-665-2026"
    )
    assert novelty["RL3DEdit"]["url"] == "https://arxiv.org/abs/2603.03143"
    assert "do not claim first confidence-to-pose use" in (
        novelty["RL3DEdit"]["boundary"]
    )
    assert novelty["DTU official dataset"]["boundary"].endswith(
        "not an ordered severity ladder"
    )

    mve = protocol["mve"]
    assert mve["models"] == ["VGGT", "MASt3R"]
    assert mve["test_scene_count"] == 20
    assert mve["unique_state_count"] == 10
    assert mve["scientific_unit_count"] == 400
    assert mve["lighting_axis"] == "UNORDERED_DISCRETE"
    assert mve["lighting_severity_order_forbidden"] is True
    assert mve["boundary_lag_legal_sequence"] == [
        "L3",
        "fog-s1",
        "fog-s2",
        "fog-s3",
    ]
    assert mve["real_paired_dtu_required_for_go"] is True
    assert mve["synthetic_only_decision"] == "MVE_SCIENTIFIC_NO_GO"
    assert mve["tartanair_role"] == "PHYSICAL_SANITY_ONLY"
    assert protocol["splits"] == {
        "names": ["dev", "reference", "calibration", "test"],
        "deterministic": True,
        "complete": True,
        "scene_disjoint": True,
        "excluded_scene_ids": [4, 15],
        "assignment_hash": 'SHA256("GeoReliab-v4:scan<ID>")',
    }
    reuse = protocol["reuse"]
    assert reuse["v1_runtime_artifacts_in_v4_scientific_evidence"] is False
    assert reuse["v1_1_source_reuse_requires_new_v4_record"] is True
    assert reuse["v4_record_requires_exact_protocol_origin"] is True
    assert reuse["v4_record_requires_source_uri_and_sha256"] is True
    assert reuse["raw_v1_runtime_objects_in_v4_records"] is False
    assert reuse["prior_gpu_selection_receipts_reusable"] is False
    assert protocol["pose_task"]["unordered_pair_count"] == 28
    assert protocol["pose_task"]["scene_failure"] == "median pair error >10deg"
    assert protocol["native_warning"]["threshold"] == (
        "model-specific calibration-L3 90th percentile"
    )
    assert protocol["metrics"]["naurc"] == (
        "(AURC_signal-AURC_oracle)/(AURC_random-AURC_oracle)"
    )
    assert protocol["statistics"]["bootstrap_resamples"] == 10000
    assert protocol["statistics"]["repeats_are_independent_observations"] is False
    assert protocol["gate"]["ranking_warning_branch"] == (
        "StaticRank lower >=0.35, CRR-pose upper <=0.15, RWG lower >=0.20"
    )
    assert protocol["gate"]["transfer_branch"] == (
        "nAURC-point upper <=0.50, TTG-pose lower >=0.20"
    )
    assert protocol["gate"]["shared_requirement"] == "SFR-pose lower >=0.30"
    assert protocol["gate"]["real_paired_dtu_required"] is True

    assert protocol["budgets"] == {
        "authorization_stop_gpu_hours": 35,
        "authorization_stop_logical_gb": 150,
        "authorization_stop_allocated_gb": 150,
        "hard_catastrophe_gpu_hours": 50,
        "hard_catastrophe_storage_gb": 1000,
        "extra_gpu_hours_not_automatic": 15,
        "extra_storage_gb_not_automatic": 850,
    }
    assert protocol["repeats"] == {
        "trigger_distance_to_main_threshold": 0.02,
        "maximum_exact_recipe_repeats": 2,
        "purpose": "NUMERICAL_REPRODUCIBILITY_ONLY",
        "included_in_ci_sample_size": False,
    }
    assert protocol["post_go_v4_1"]["enabled_only_after"] == (
        "MVE_GO_TO_EXTERNAL_VALIDATION"
    )
    assert protocol["post_go_v4_1"]["writes_back_into_v4_mve"] is False
    assert protocol["dates"] == {
        "claim_reset_novelty_protocol": "2026-07-30/2026-08-04",
        "pair_identity_lighting_metric_tests": "2026-08-05/2026-08-10",
        "gpu_request_then_dual_model_sanity": "2026-08-11/2026-08-13",
        "mve_400_units": "2026-08-14/2026-08-21",
        "statistics_and_gate": "2026-08-22/2026-08-24",
        "post_go_v4_1_uavlight": "2026-08-25/2026-09-15",
        "internal_evidence_freeze": "2026-09-15",
        "cvpr_2027_deadline": "TBA",
        "cvpr_2026_extrapolation_forbidden": True,
    }


def test_v4_lock_is_line_ending_invariant_and_fails_closed_on_tamper(
    tmp_path: Path,
) -> None:
    _copy_locked_tree(tmp_path)
    for relative in (*V4_LOCKED_SCIENCE_FILES, *V1_IMMUTABLE_SCIENCE_FILES):
        path = tmp_path / relative
        canonical = (
            path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        )
        path.write_bytes(canonical)

    assert validate_v4_science_lock(tmp_path)["status"] == (
        GEORELIAB_V4_PROTOCOL_READY
    )
    (tmp_path / "configs" / "georeliab_v4_protocol.toml").write_text(
        "tampered = true\n", encoding="utf-8"
    )
    with pytest.raises(V4ScienceLockError, match="v4 science lock mismatch"):
        validate_v4_science_lock(tmp_path)


def test_old_v1_config_and_lock_git_blobs_are_byte_for_byte_unchanged() -> None:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--no-ext-diff",
            "--exit-code",
            V4_BASE_COMMIT,
            "--",
            *V1_IMMUTABLE_SCIENCE_FILES,
        ],
        cwd=SOURCE_ROOT,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout.decode(
        "utf-8", errors="replace"
    ) + result.stderr.decode("utf-8", errors="replace")


def test_v1_artifact_or_evidence_cannot_enter_v4_scientific_bundle() -> None:
    legacy_stage_evidence = _old_stage_evidence()
    with pytest.raises(V4ScienceLockError, match="closed schema.*missing"):
        validate_v4_scientific_bundle(SOURCE_ROOT, legacy_stage_evidence)

    wrapped_legacy = _v4_bundle(legacy_stage_evidence)
    with pytest.raises(V4ScienceLockError, match="closed schema.*missing"):
        validate_v4_scientific_bundle(SOURCE_ROOT, wrapped_legacy)

    missing_artifact_binding = _v4_bundle(_v4_record())
    del missing_artifact_binding["artifacts"][0]["protocol_provenance"]  # type: ignore[index]
    with pytest.raises(V4ScienceLockError, match="closed schema.*missing"):
        validate_v4_scientific_bundle(SOURCE_ROOT, missing_artifact_binding)


def test_v4_scientific_bundle_artifacts_must_be_json_list() -> None:
    bundle = _v4_bundle(_v4_record())
    artifacts = bundle["artifacts"]
    assert isinstance(artifacts, list)
    bundle["artifacts"] = tuple(artifacts)

    with pytest.raises(V4ScienceLockError, match="artifacts.*JSON list"):
        validate_v4_scientific_bundle(SOURCE_ROOT, bundle)


def _old_prediction_artifact() -> dict[str, object]:
    return {
        "schema_version": "1.1",
        "run_id": "old-v1-run",
        "sample_key": "dtu/test/scan1/fps8/L3/0/0",
        "geometry_prediction_uri": "file:///old-v1/geometry.npz",
        "native_confidence_uri": "file:///old-v1/confidence.npz",
        "valid_mask_uri": "file:///old-v1/mask.npz",
        "hook_location": None,
        "runtime_seconds": 1.0,
        "peak_memory_mb": 1024.0,
        "invalid_prediction": False,
        "payload_digests": {
            "geometry_prediction_uri": "b" * 64,
            "native_confidence_uri": "c" * 64,
            "valid_mask_uri": "d" * 64,
        },
    }


def _old_audit_record() -> dict[str, object]:
    return {
        "schema_version": "1.1",
        "run_id": "old-v1-run",
        "sample_key": "dtu/test/scan1/fps8/L3/0/0",
        "gt_error": 1.5,
        "failure_label": False,
        "selection_score": 0.9,
        "coverage": 0.8,
        "accepted": True,
        "downstream_outcome": 0.75,
        "invalid_prediction": False,
        "metadata": {"dense_audit_uri": "file:///old-v1/dense.npz"},
    }


def _old_stage_evidence() -> dict[str, object]:
    return {
        "schema_version": "stage-evidence-v1",
        "scientific_validity": "SCIENTIFIC",
        "bundle_index": [],
    }


def _old_path_digest_wrapper() -> dict[str, object]:
    return {
        "downstream_index": [
            {
                "evidence_path": "stage/test/downstream/v1.json",
                "evidence_sha256": "e" * 64,
            }
        ]
    }


@pytest.mark.parametrize("injection_layer", ["bundle", "envelope"])
@pytest.mark.parametrize(
    "legacy_payload",
    [
        _old_stage_evidence(),
        _old_prediction_artifact(),
        _old_audit_record(),
        _old_path_digest_wrapper(),
    ],
    ids=[
        "stage-evidence-v1",
        "prediction-artifact-v1.1",
        "audit-record-v1.1",
        "path-digest-wrapper",
    ],
)
def test_closed_bundle_and_envelope_schemas_reject_legacy_extra_fields(
    injection_layer: str,
    legacy_payload: dict[str, object],
) -> None:
    bundle = _v4_bundle(_v4_record())
    if injection_layer == "bundle":
        target = bundle
    else:
        target = bundle["artifacts"][0]  # type: ignore[index]
        assert isinstance(target, dict)
    target["legacy_extra"] = legacy_payload

    with pytest.raises(
        V4ScienceLockError,
        match="closed schema.*unexpected.*legacy_extra",
    ):
        validate_v4_scientific_bundle(SOURCE_ROOT, bundle)


@pytest.mark.parametrize(
    ("layer", "missing_key"),
    [
        ("bundle", "schema_version"),
        ("bundle", "artifacts"),
        ("envelope", "project_line"),
        ("envelope", "protocol_provenance"),
        ("record", "schema_version"),
        ("record", "data"),
    ],
)
def test_closed_v4_schemas_reject_missing_keys(
    layer: str,
    missing_key: str,
) -> None:
    record = _v4_record()
    bundle = _v4_bundle(record)
    if layer == "bundle":
        del bundle[missing_key]
    elif layer == "envelope":
        envelope = bundle["artifacts"][0]  # type: ignore[index]
        assert isinstance(envelope, dict)
        del envelope[missing_key]
    else:
        del record[missing_key]

    with pytest.raises(
        V4ScienceLockError,
        match=rf"closed schema.*missing.*{missing_key}",
    ):
        validate_v4_scientific_bundle(SOURCE_ROOT, bundle)


@pytest.mark.parametrize(
    "extra_value",
    [7, object()],
    ids=["scalar", "custom-object"],
)
def test_closed_v4_record_rejects_unknown_fields(extra_value: object) -> None:
    record = _v4_record()
    record["ignored_extra"] = extra_value

    with pytest.raises(
        V4ScienceLockError,
        match="closed schema.*unexpected.*ignored_extra",
    ):
        validate_v4_scientific_bundle(SOURCE_ROOT, _v4_bundle(record))


@pytest.mark.parametrize(
    "non_json_value",
    [
        b"bytes",
        object(),
        {1: "non-string JSON key"},
        range(2),
        (1, 2),
        float("nan"),
        float("inf"),
    ],
    ids=[
        "bytes",
        "custom-object",
        "non-string-key",
        "range",
        "tuple",
        "nan",
        "infinity",
    ],
)
def test_v4_record_data_rejects_non_json_values(
    non_json_value: object,
) -> None:
    bundle = _v4_bundle(
        _v4_record(data={"nested": [non_json_value]})
    )

    with pytest.raises(V4ScienceLockError, match="non-JSON"):
        validate_v4_scientific_bundle(SOURCE_ROOT, bundle)


@pytest.mark.parametrize(
    "legacy_payload",
    [_old_prediction_artifact(), _old_audit_record()],
)
def test_wrapped_old_runtime_objects_remain_forbidden(
    legacy_payload: dict[str, object],
) -> None:
    bundle = _v4_bundle(
        _v4_record(data={"legacy_runtime_object": legacy_payload})
    )

    with pytest.raises(V4ScienceLockError, match="legacy schema.*1.1"):
        validate_v4_scientific_bundle(SOURCE_ROOT, bundle)


def test_old_runtime_signature_remains_forbidden_without_schema_marker() -> None:
    legacy_payload = _old_prediction_artifact()
    del legacy_payload["schema_version"]
    bundle = _v4_bundle(
        _v4_record(data={"legacy_runtime_object": legacy_payload})
    )

    with pytest.raises(V4ScienceLockError, match="raw legacy PredictionArtifact"):
        validate_v4_scientific_bundle(SOURCE_ROOT, bundle)


def test_nested_old_evidence_path_digest_wrapper_remains_forbidden() -> None:
    old_wrapper = _old_path_digest_wrapper()
    bundle = _v4_bundle(_v4_record(data=old_wrapper, record_kind="evidence"))

    with pytest.raises(V4ScienceLockError, match="path/digest wrapper"):
        validate_v4_scientific_bundle(SOURCE_ROOT, bundle)


def test_new_v4_record_requires_exact_origin_and_source_binding() -> None:
    missing_origin = _v4_record()
    del missing_origin["origin"]
    with pytest.raises(V4ScienceLockError, match="closed schema.*missing.*origin"):
        validate_v4_scientific_bundle(
            SOURCE_ROOT, _v4_bundle(missing_origin)
        )

    wrong_protocol_hash = _v4_record()
    origin = wrong_protocol_hash["origin"]
    assert isinstance(origin, dict)
    provenance = origin["protocol_provenance"]
    assert isinstance(provenance, dict)
    provenance["protocol_sha256"] = "0" * 64
    with pytest.raises(V4ScienceLockError, match="exact v4 record origin"):
        validate_v4_scientific_bundle(
            SOURCE_ROOT, _v4_bundle(wrong_protocol_hash)
        )

    invalid_source_digest = _v4_record()
    invalid_source_digest["source_sha256"] = "NOT-A-DIGEST"
    with pytest.raises(V4ScienceLockError, match="source_sha256"):
        validate_v4_scientific_bundle(
            SOURCE_ROOT, _v4_bundle(invalid_source_digest)
        )


@pytest.mark.parametrize(
    "marker",
    [
        {"project_line": "v1"},
        {"protocol_id": "georeliab-v1"},
        {"project_route": "v1"},
        {"route": "v1"},
        {"protocol_version": "1.0"},
    ],
)
def test_nested_non_v4_route_or_protocol_markers_fail_closed(
    marker: dict[str, object],
) -> None:
    bundle = _v4_bundle(_v4_record(data={"nested": [marker]}))

    with pytest.raises(V4ScienceLockError, match="non-v4"):
        validate_v4_scientific_bundle(SOURCE_ROOT, bundle)


def test_v1_1_data_is_reusable_only_as_non_admitting_digest_bound_record() -> None:
    raw_old_artifact = _old_prediction_artifact()

    with pytest.raises(V4ScienceLockError, match="closed schema"):
        validate_v4_scientific_bundle(SOURCE_ROOT, raw_old_artifact)

    bundle = _v4_bundle(
        _v4_record(
            data={
                "reuse_mode": "source-bytes-only",
                "source_contract": "PredictionArtifact v1.1",
            }
        )
    )
    structural = validate_v4_scientific_bundle_structure(
        SOURCE_ROOT,
        bundle,
    )
    assert structural == {
        "status": "V4_BUNDLE_STRUCTURE_VALID_ONLY",
        "protocol_id": V4_PROTOCOL_ID,
        "protocol_sha256": V4_PROTOCOL_SHA256,
        "artifact_count": 1,
        "scientific_admission": False,
    }
    with pytest.raises(
        V4ScienceLockError,
        match="canonical finalizer.*exact 400",
    ):
        validate_v4_scientific_bundle(SOURCE_ROOT, bundle)
