from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from georeliab_mve.artifact_storage import (
    ArtifactStorageError,
    assert_npz_equivalent,
    finalize_mast3r_cache,
    finalize_zero_update_adapter,
    load_npz_arrays,
    npz_fingerprint,
    reencode_npz_lossless,
    sha256_file,
    validate_retention_receipt,
    write_deterministic_npz,
    write_shared_gt,
)


def _arrays() -> dict[str, np.ndarray]:
    return {
        "float64": np.arange(2400, dtype=np.float64).reshape(800, 3),
        "bool": np.asarray([True, False, True], dtype=bool),
        "scalar": np.asarray("bound-parent"),
    }


def test_deterministic_npz_rejects_unsafe_array_names(tmp_path: Path):
    with pytest.raises(ArtifactStorageError, match="unsafe NPZ array name"):
        write_deterministic_npz(tmp_path / "bad.npz", {"../escape": np.ones(1)})


def test_deterministic_npz_can_preserve_frozen_member_order(tmp_path: Path):
    path = tmp_path / "ordered.npz"
    arrays = {
        "failure_label": np.asarray([False, True], dtype=bool),
        "voxel_points": np.arange(6, dtype=np.float64).reshape(2, 3),
        "risk": np.asarray([0.1, 0.9], dtype=np.float64),
    }
    order = ("voxel_points", "risk", "failure_label")

    write_deterministic_npz(path, arrays, member_order=order)

    with np.load(path, allow_pickle=False) as payload:
        assert tuple(payload.files) == order
    with pytest.raises(ArtifactStorageError, match="every array exactly once"):
        write_deterministic_npz(
            tmp_path / "incomplete.npz",
            arrays,
            member_order=("risk",),
        )


def test_deterministic_npz_is_compressed_and_exact(tmp_path: Path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    write_deterministic_npz(first, _arrays())
    write_deterministic_npz(second, dict(reversed(list(_arrays().items()))))

    assert sha256_file(first) == sha256_file(second)
    assert assert_npz_equivalent(first, second) == npz_fingerprint(first)
    loaded = load_npz_arrays(first)
    for name, expected in _arrays().items():
        assert loaded[name].dtype == expected.dtype
        assert loaded[name].shape == expected.shape
        assert np.array_equal(loaded[name], expected)
        assert (
            np.ascontiguousarray(loaded[name]).tobytes(order="C")
            == np.ascontiguousarray(expected).tobytes(order="C")
        )
    with zipfile.ZipFile(first) as archive:
        assert archive.infolist()
        assert all(
            item.compress_type == zipfile.ZIP_DEFLATED
            for item in archive.infolist()
        )


def test_lossless_reencode_preserves_source_and_decoded_semantics(tmp_path: Path):
    source = tmp_path / "source.npz"
    destination = tmp_path / "destination.npz"
    np.savez(source, **_arrays())
    source_before = source.read_bytes()

    report = reencode_npz_lossless(source, destination)

    assert source.read_bytes() == source_before
    assert report["array_fingerprint"] == npz_fingerprint(source)
    assert assert_npz_equivalent(source, destination) == report["array_fingerprint"]
    assert report["source_sha256"] != report["destination_sha256"]


def test_shared_gt_deduplicates_and_detects_tampering(tmp_path: Path):
    points = np.arange(300, dtype=np.float64).reshape(100, 3)
    first = write_shared_gt(tmp_path, points)
    second = write_shared_gt(tmp_path, points.copy())

    assert first.path == second.path
    assert first.sha256 == second.sha256
    assert list((tmp_path / "shared_gt").glob("*.npz")) == [first.path]

    first.path.write_bytes(b"tampered")
    with pytest.raises(ArtifactStorageError, match="digest drift"):
        write_shared_gt(tmp_path, points)


def _cache(bundle: Path) -> Path:
    cache = bundle / "adapter" / "mast3r_cache"
    (cache / "forward").mkdir(parents=True)
    (cache / "forward" / "a.pth").write_bytes(b"a" * 128)
    (cache / "corres_b.pth").write_bytes(b"b" * 64)
    return cache


def test_mast3r_cache_requires_inventory_unless_adapter_exception(tmp_path: Path):
    bundle = tmp_path / "empty.partial"
    bundle.mkdir()

    with pytest.raises(ArtifactStorageError, match="non-empty verified cache inventory"):
        finalize_mast3r_cache(bundle, stage="smoke")

    receipt = finalize_mast3r_cache(bundle, stage="smoke", allow_empty=True)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["policy"] == "adapter-exception-empty-cache"
    assert payload["members"] == []
    assert validate_retention_receipt(bundle) == payload


def test_p2_cache_keeps_inventory_receipt_only(tmp_path: Path):
    bundle = tmp_path / "smoke.partial"
    cache = _cache(bundle)

    receipt_path = finalize_mast3r_cache(bundle, stage="smoke")

    assert not cache.exists()
    assert not (bundle / "mast3r_pairwise_cache.tar.gz").exists()
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["policy"] == "p2-inventory-only"
    assert [row["relative_path"] for row in payload["members"]] == [
        "corres_b.pth",
        "forward/a.pth",
    ]
    assert validate_retention_receipt(bundle) == payload


def test_p3_cache_archive_is_deterministic_and_tamper_detected(tmp_path: Path):
    first = tmp_path / "first.partial"
    second = tmp_path / "second.partial"
    _cache(first)
    _cache(second)

    finalize_mast3r_cache(first, stage="test")
    finalize_mast3r_cache(second, stage="test")

    first_archive = first / "mast3r_pairwise_cache.tar.gz"
    second_archive = second / "mast3r_pairwise_cache.tar.gz"
    assert sha256_file(first_archive) == sha256_file(second_archive)
    validate_retention_receipt(first)

    first_archive.write_bytes(b"tampered")
    with pytest.raises(ArtifactStorageError, match="digest mismatch"):
        validate_retention_receipt(first)


def test_p5_retains_only_subset_and_receipt_from_adapter_tree(tmp_path: Path):
    bundle = tmp_path / "zero.partial"
    cache = _cache(bundle)
    (cache.parent / "geometry.npz").write_bytes(b"large-adapter-output")
    subset = bundle / "subset_prediction.npz"
    write_deterministic_npz(
        subset,
        {
            "points": np.ones((3, 3), dtype=np.float64),
            "parent_model": np.asarray("VGGT"),
        },
    )

    receipt_path = finalize_zero_update_adapter(bundle, subset_path=subset)

    assert not (bundle / "adapter").exists()
    assert subset.exists()
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["subset_sha256"] == sha256_file(subset)
    assert validate_retention_receipt(bundle) == payload


def test_retention_refuses_symlink_members(tmp_path: Path):
    bundle = tmp_path / "smoke.partial"
    cache = bundle / "adapter" / "mast3r_cache"
    cache.mkdir(parents=True)
    outside = tmp_path / "outside.pth"
    outside.write_bytes(b"outside")
    link = cache / "escape.pth"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symbolic links are not available in this environment")

    with pytest.raises(ArtifactStorageError, match="symbolic links"):
        finalize_mast3r_cache(bundle, stage="smoke")
    assert outside.read_bytes() == b"outside"
