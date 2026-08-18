from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from georeliab_mve.contracts import (
    PredictionArtifact,
    RunManifest,
    RunMode,
    SampleKey,
    ScientificProvenance,
    ScientificValidity,
)
from georeliab_mve.v4_attempt05_recovery import (
    AUTHORIZED_GPU_UUID,
    NO_SCIENTIFIC_RESULT,
    build_recovery_smoke_manifest,
)


HARNESS_PATH = Path(__file__).with_name("gate2_gpu_smoke_harness.py")
SPEC = importlib.util.spec_from_file_location("gate2_gpu_smoke_harness", HARNESS_PATH)
assert SPEC is not None and SPEC.loader is not None
harness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harness)


def test_gate2_manifest_has_exact_kill_plan_and_inference_counts() -> None:
    smoke = build_recovery_smoke_manifest(
        schedule_identity_sha256="a" * 64,
        support_scene_ids=tuple(range(1, 21)),
    )
    assert len(smoke.unit_keys) == 12
    assert set(smoke.interruption_plan.values()) == harness.INTERRUPTION_PHASES
    assert smoke.expected_inference_starts[smoke.unit_keys[10]] == 2
    assert all(
        smoke.expected_inference_starts[key] == 1
        for key in smoke.unit_keys
        if key != smoke.unit_keys[10]
    )
    assert smoke.gpu_uuid == AUTHORIZED_GPU_UUID


def test_safe_key_and_idempotency_are_deterministic() -> None:
    first = harness._safe_key("VGGT|1|L3")
    second = harness._safe_key("VGGT|1|L3")
    assert first == second
    assert "/" not in first and "|" not in first
    assert harness._idempotency_key("b" * 64, "VGGT|1|L3") == harness._idempotency_key(
        "b" * 64, "VGGT|1|L3"
    )
    assert harness._idempotency_key("b" * 64, "VGGT|1|L3") != harness._idempotency_key(
        "b" * 64, "MASt3R|1|L3"
    )


def test_home_output_is_rejected() -> None:
    with pytest.raises(harness.Gate2HarnessError, match="UNDER_HOME"):
        harness._refuse_home_output(Path("/home/example/gate2"))


def test_local_output_requires_current_home_descendant() -> None:
    harness._require_home_output(Path.home() / "georeliab-gate2-local" / "runs" / "one")
    with pytest.raises(harness.Gate2HarnessError, match="OUTSIDE_HOME"):
        harness._require_home_output(Path("/tmp/georeliab-gate2-local"))


def test_local_run_parser_sets_development_boundary() -> None:
    args = harness.build_parser().parse_args(
        [
            "local-run",
            "--input-closure-dir",
            "/home/example/local/manifests",
            "--overlay-config",
            "/home/example/local/overlay.toml",
            "--output-root",
            "/home/example/local/runs/one",
            "--authorization-note",
            "user-local-development-smoke",
        ]
    )
    assert args.local_development is True

    formal = harness.build_parser().parse_args(
        [
            "run",
            "--input-closure-dir",
            "/srv/input",
            "--overlay-config",
            "/srv/overlay.toml",
            "--output-root",
            "/srv/output",
            "--authorization-note",
            "formal",
        ]
    )
    assert formal.local_development is False


def test_gpu_exclusivity_policy_differs_only_for_local_development() -> None:
    assert harness._non_target_gpu_processes_forbidden(local_development=False) is True
    assert harness._non_target_gpu_processes_forbidden(local_development=True) is False


def test_gate2_imported_production_source_matches_canonical() -> None:
    repo = Path(__file__).resolve().parents[1]
    manifest = harness._source_manifest(repo)
    assert manifest["canonical_commit"] == harness.CANONICAL_COMMIT
    assert manifest["canonical_tree"] == harness.CANONICAL_TREE
    assert manifest["production_source_zero_drift"] is True
    assert manifest["excluded_production_paths"] == list(
        harness.EXCLUDED_PRODUCTION_PATHS
    )
    assert len(manifest["production_files"]) >= 40


def test_transaction_files_bind_all_prediction_payloads(tmp_path: Path) -> None:
    geometry = tmp_path / "geometry.npz"
    confidence = tmp_path / "confidence.npz"
    valid = tmp_path / "valid.npz"
    geometry.write_bytes(b"geometry")
    confidence.write_bytes(b"confidence")
    valid.write_bytes(b"valid")
    provenance = ScientificProvenance(
        project_commit="a" * 40,
        project_tree="b" * 40,
        model_source_commit="c" * 40,
        environment_lock_sha256="d" * 64,
        corruption_manifest_sha256="e" * 64,
        split_view_manifest_sha256="f" * 64,
    )
    manifest = RunManifest(
        run_id="gate2-vggt-scan001-l3",
        mode=RunMode.SMOKE,
        scientific_validity=ScientificValidity.NON_SCIENTIFIC_SMOKE,
        model="VGGT",
        checkpoint_hash="1" * 64,
        dataset="dtu",
        split="gate2",
        seed=0,
        intervention_version="none",
        corruption_version="georeliab-v4-gate2-l3",
        environment={"device": "cuda:0"},
        rgb_digest="rgb",
        prompt_digest="prompt",
        decoder_digest="decoder",
        provenance=provenance,
    )
    sample = SampleKey("dtu", "gate2", "scan001", "views-0001", "L3", "0", "0")
    prediction = PredictionArtifact(
        run_id=manifest.run_id,
        sample_key=str(sample),
        geometry_prediction_uri=geometry.as_uri(),
        native_confidence_uri=confidence.as_uri(),
        valid_mask_uri=valid.as_uri(),
        hook_location=None,
        runtime_seconds=1.0,
        peak_memory_mb=2.0,
        invalid_prediction=False,
        payload_digests={},
    )
    files = harness._transaction_files(manifest, prediction)
    assert set(files) == {
        "geometry.npz",
        "confidence.npz",
        "valid-mask.npz",
        "run-manifest.json",
        "prediction-artifact.json",
    }
    assert b"NON_SCIENTIFIC_SMOKE" in files["run-manifest.json"]
    assert NO_SCIENTIFIC_RESULT.encode() not in files["prediction-artifact.json"]


def test_local_manifest_is_explicitly_non_scientific() -> None:
    views = tuple(
        harness.RenderedView(
            view_id=index,
            png_path=Path(f"/home/example/{index}.png"),
            png_sha256=f"{index:064x}",
        )
        for index in range(1, 9)
    )
    manifest, sample = harness._smoke_run_manifest(
        model="VGGT",
        scene_id=1,
        views=views,
        model_spec={
            "checkpoint": "1" * 64,
            "source_commit": "2" * 40,
            "python": "3.10.20",
            "torch": "2.3.1+cu121",
            "environment_lock": "3" * 64,
            "dust3r_commit": None,
            "croco_commit": None,
        },
        runtime_binding_sha="4" * 64,
        schedule_identity_sha="5" * 64,
        local_development=True,
    )
    assert manifest.scientific_validity == ScientificValidity.NON_SCIENTIFIC_SMOKE
    assert manifest.split == "local-gate2"
    assert manifest.run_id.startswith("local-gate2-")
    assert "local-gate2" in str(sample)


def test_scientific_marker_scanner_is_fail_closed(tmp_path: Path) -> None:
    safe = tmp_path / "safe.json"
    safe.write_text('{"scientific_result":"NO_SCIENTIFIC_RESULT"}\n', encoding="utf-8")
    assert harness._scan_for_scientific_markers(tmp_path) == []
    forbidden = tmp_path / "forbidden.json"
    forbidden.write_text('{"event_type":"MVE_FINALIZED"}\n', encoding="utf-8")
    violations = harness._scan_for_scientific_markers(tmp_path)
    assert len(violations) == 1
    assert "MVE_FINALIZED" in violations[0]
