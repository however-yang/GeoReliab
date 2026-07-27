from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from georeliab_mve.materialization import (
    DTU_SCAN_IDS,
    FROZEN_TYPING_EXTENSIONS_SITE,
    build_dtu_archive_inventory,
    materialize_frozen_selection,
    validate_rectified_index,
    validate_tartanair_indexes,
    verify_frozen_overlay_identities,
    verify_materialization_manifest,
)
from georeliab_mve.preparation import PreparationError
from georeliab_mve.preparation_round2 import (
    build_split_view_manifest,
    load_prepared_batch,
    load_tartanair_prepared_pairs,
)
from georeliab_mve.tartanair_range import (
    RemoteZipEntry,
    RemoteZipIndex,
    extract_range_members_evidence,
)


def _zip_index(names: list[str], *, payload: bytes | None = b"payload") -> tuple[bytes, RemoteZipIndex]:
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.writestr(name, name.encode() if payload is None else payload)
    raw = stream.getvalue()
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        entries = {
            info.filename: RemoteZipEntry(
                name=info.filename,
                compression=info.compress_type,
                compressed_size=info.compress_size,
                uncompressed_size=info.file_size,
                crc32=info.CRC,
                local_offset=info.header_offset,
            )
            for info in archive.infolist()
        }
    return raw, RemoteZipIndex(
        content_length=len(raw),
        etag='"fixture-etag"',
        entries=entries,
        central_directory_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _rectified_names() -> list[str]:
    return [
        f"Rectified/scan{scene}/rect_{view:03d}_3_r5000.png"
        for scene in DTU_SCAN_IDS
        for view in range(1, 50)
    ]


def _tartan_names(modality: str) -> list[str]:
    suffix = "_lcam_front.png" if modality == "image" else "_lcam_front_depth.png"
    return [
        f"GreatMarsh/Data_easy/P000/{modality}_lcam_front/{frame:06d}{suffix}"
        for frame in range(3537)
    ]


def _tartan_directory_entry(modality: str) -> str:
    return f"GreatMarsh/Data_easy/P000/{modality}_lcam_front/"


def test_remote_inventory_validators_require_exact_official_members():
    _, rectified = _zip_index(_rectified_names())
    by_scene = validate_rectified_index(rectified)
    assert set(by_scene) == set(DTU_SCAN_IDS)
    assert all(len(members) == 49 for members in by_scene.values())

    _, image = _zip_index(_tartan_names("image") + [_tartan_directory_entry("image")])
    _, depth = _zip_index(_tartan_names("depth") + [_tartan_directory_entry("depth")])
    frames, selected_image, selected_depth = validate_tartanair_indexes(image, depth)
    assert len(frames) == len(selected_image) == len(selected_depth) == 100
    assert frames[0] == "000000"


def test_remote_inventory_rejects_missing_extra_and_misnamed_members():
    names = _rectified_names()
    names[-1] = "Rectified/scan128/rect_049_3_r5000-extra.png"
    _, bad_rectified = _zip_index(names)
    with pytest.raises(PreparationError, match="6076"):
        validate_rectified_index(bad_rectified)

    _, depth = _zip_index(_tartan_names("depth"))

    missing_image_names = _tartan_names("image")[:-1]
    _, missing_image = _zip_index(missing_image_names)
    with pytest.raises(PreparationError, match="exactly 3537"):
        validate_tartanair_indexes(missing_image, depth)

    extra_image_names = _tartan_names("image") + [
        "GreatMarsh/Data_easy/P000/image_lcam_front/999999_lcam_front.png"
    ]
    _, extra_image = _zip_index(extra_image_names)
    with pytest.raises(PreparationError, match="exactly 3537"):
        validate_tartanair_indexes(extra_image, depth)

    misnamed_image_names = _tartan_names("image")
    misnamed_image_names[-1] = misnamed_image_names[-1].replace(".png", ".jpg")
    _, misnamed_image = _zip_index(misnamed_image_names)
    with pytest.raises(PreparationError, match="misnamed P000 member"):
        validate_tartanair_indexes(misnamed_image, depth)

    wrong_range_image_names = _tartan_names("image")
    wrong_range_image_names[-1] = wrong_range_image_names[-1].replace("003536", "999999")
    _, wrong_range_image = _zip_index(wrong_range_image_names)
    with pytest.raises(PreparationError, match="000000..003536"):
        validate_tartanair_indexes(wrong_range_image, depth)


class _FakeHttpResponse:
    def __init__(
        self,
        *,
        data: bytes = b"",
        fail_read: bool = False,
        read_error: BaseException | None = None,
    ):
        self.status = 206
        self.headers = {"Content-Range": "bytes 2-5/10", "ETag": '"fixture-etag"'}
        self._data = data
        self._fail_read = fail_read
        self._read_error = read_error

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        return False

    def read(self) -> bytes:
        if self._fail_read:
            raise TimeoutError("range read stalled")
        if self._read_error is not None:
            raise self._read_error
        return self._data


def test_range_request_retries_transient_timeout_and_preserves_invariants(monkeypatch):
    import georeliab_mve.tartanair_range as range_module

    calls = []

    def urlopen(request, *, timeout):
        calls.append((request, timeout))
        return _FakeHttpResponse(data=b"cdef", fail_read=len(calls) == 1)

    monkeypatch.setattr(range_module.urllib.request, "urlopen", urlopen)
    data, etag, total = range_module._request_range("https://example.test/archive.zip", 2, 5)
    assert data == b"cdef"
    assert etag == '"fixture-etag"'
    assert total == 10
    assert len(calls) == 2
    assert [timeout for _request, timeout in calls] == [range_module._HTTP_TIMEOUT_SECONDS] * 2
    assert [request.get_header("Range") for request, _timeout in calls] == ["bytes=2-5"] * 2


def test_range_request_fails_after_exact_bounded_timeout_attempts(monkeypatch):
    import georeliab_mve.tartanair_range as range_module

    calls = []

    def urlopen(request, *, timeout):
        calls.append((request, timeout))
        raise TimeoutError("range open stalled")

    monkeypatch.setattr(range_module.urllib.request, "urlopen", urlopen)
    with pytest.raises(PreparationError, match="bytes=2-5.*failed after 3 transport attempts"):
        range_module._request_range("https://example.test/archive.zip", 2, 5)
    assert len(calls) == range_module._HTTP_MAX_ATTEMPTS
    assert [timeout for _request, timeout in calls] == [range_module._HTTP_TIMEOUT_SECONDS] * range_module._HTTP_MAX_ATTEMPTS


def test_range_request_retries_transient_incomplete_read(monkeypatch):
    import georeliab_mve.tartanair_range as range_module

    calls = []

    def urlopen(request, *, timeout):
        calls.append((request, timeout))
        if len(calls) == 1:
            error = range_module.http.client.IncompleteRead(b"", 4)
            return _FakeHttpResponse(read_error=error)
        return _FakeHttpResponse(data=b"cdef")

    monkeypatch.setattr(range_module.urllib.request, "urlopen", urlopen)
    data, etag, total = range_module._request_range("https://example.test/archive.zip", 2, 5)
    assert data == b"cdef"
    assert etag == '"fixture-etag"'
    assert total == 10
    assert len(calls) == 2


def test_range_request_fails_after_exact_bounded_incomplete_read_attempts(monkeypatch):
    import georeliab_mve.tartanair_range as range_module

    calls = []

    def urlopen(request, *, timeout):
        calls.append((request, timeout))
        error = range_module.http.client.IncompleteRead(b"", 4)
        return _FakeHttpResponse(read_error=error)

    monkeypatch.setattr(range_module.urllib.request, "urlopen", urlopen)
    with pytest.raises(PreparationError, match="bytes=2-5.*failed after 3 transport attempts"):
        range_module._request_range("https://example.test/archive.zip", 2, 5)
    assert len(calls) == range_module._HTTP_MAX_ATTEMPTS

def test_range_materialization_is_atomic_reusable_and_tamper_closed(monkeypatch, tmp_path):
    import georeliab_mve.tartanair_range as range_module

    member = "archive/member.bin"
    raw, index = _zip_index([member], payload=b"official bytes")

    def request_range(_url: str, start: int, end: int):
        return raw[start:end + 1], index.etag, len(raw)

    monkeypatch.setattr(range_module, "_request_range", request_range)
    evidence = extract_range_members_evidence(
        "https://example.test/archive.zip", index, [member], tmp_path
    )[member]
    path = Path(evidence["path"])
    assert path.read_bytes() == b"official bytes"
    assert evidence["disposition"] == "written"
    reused = extract_range_members_evidence(
        "https://example.test/archive.zip", index, [member], tmp_path
    )[member]
    assert reused["disposition"] == "reused"
    path.write_bytes(b"tampered bytes")
    with pytest.raises(PreparationError, match="wrong (size|CRC)"):
        extract_range_members_evidence(
            "https://example.test/archive.zip", index, [member], tmp_path
        )


def _write_local_dtu_archives(root: Path) -> tuple[Path, Path]:
    sample = root / 'SampleSet.zip'
    points = root / 'Points.zip'
    with zipfile.ZipFile(sample, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for view in range(1, 50):
            projection = f'1 0 0 {-view}\n0 1 0 0\n0 0 1 0\n'
            archive.writestr(
                f'MVS Data/Calibration/cal18/pos_{view:03d}.txt', projection,
            )
        for scene in DTU_SCAN_IDS:
            archive.writestr(f'MVS Data/ObsMask/ObsMask{scene}_10.mat', f'mask-{scene}')
    with zipfile.ZipFile(points, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for scene in DTU_SCAN_IDS:
            archive.writestr(f'Points/stl/stl{scene:03d}_total.ply', f'ply-{scene}')
    return sample, points


def test_archive_inventory_reads_real_zip_members_and_official_camera_ids(tmp_path):
    sample, points = _write_local_dtu_archives(tmp_path)
    _, rectified = _zip_index(_rectified_names())
    scenes, provenance = build_dtu_archive_inventory(sample, points, rectified)
    assert len(scenes) == 124
    assert set(scenes[0].camera_centers) == set(range(1, 50))
    assert scenes[0].camera_centers[49].tolist() == pytest.approx([49.0, 0.0, 0.0])
    assert provenance['scenes'][-1]['points_member'].endswith('stl128_total.ply')
    assert provenance['camera_members']['1'].endswith('pos_001.txt')


def test_frozen_identity_verifier_checks_every_source_file_and_environment(monkeypatch, tmp_path):
    import georeliab_mve.materialization as materialization

    commits = {'vggt': 'a' * 40, 'mast3r': 'b' * 40,
               'dust3r': 'c' * 40, 'croco': 'd' * 40}
    runtime = {}
    for name in commits:
        source = tmp_path / name
        source.mkdir()
        runtime[f'{name}_source'] = str(source)
    for name, versions in {'vggt': ('3.10.20', '2.3.1+cu121'),
                           'mast3r': ('3.10.20', '2.5.1+cu121')}.items():
        env = tmp_path / f'env-{name}'
        env.mkdir()
        runtime[f'{name}_env'] = str(env)
        runtime[f'{name}_python'], runtime[f'{name}_torch'] = versions
    typing_site = tmp_path / 'typing-site'
    typing_site.mkdir()
    typing_file = typing_site / 'typing_extensions.py'
    typing_file.write_text('__version__ = "4.15.0"\n', encoding='utf-8')
    runtime['typing_extensions_site'] = str(typing_site)
    resources = {f'{name}_source_commit': commit for name, commit in commits.items()}
    resources['typing_extensions_version'] = '4.15.0'
    resources['typing_extensions_sha256'] = hashlib.sha256(typing_file.read_bytes()).hexdigest()
    for name in ('vggt_checkpoint', 'mast3r_checkpoint', 'mast3r_config'):
        path = tmp_path / name
        path.write_bytes(name.encode())
        resources[name] = str(path)
        resources[f'{name}_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    by_path = {str(tmp_path / name): commit for name, commit in commits.items()}
    monkeypatch.setattr(materialization, '_git_head', lambda path: by_path[str(path)])
    probes = []
    def fake_python_torch_versions(path, cache_root, **kwargs):
        probes.append(kwargs)
        return ('3.10.20', '2.3.1+cu121') if path.name == 'env-vggt' else ('3.10.20', '2.5.1+cu121')
    monkeypatch.setattr(materialization, '_python_torch_versions', fake_python_torch_versions)
    evidence = verify_frozen_overlay_identities(
        runtime=runtime, resources=resources, cache_root=tmp_path / 'cache',
        enforce_typing_extensions_home=False,
    )
    assert set(evidence['sources']) == set(commits)
    assert set(evidence['files']) == {'vggt_checkpoint', 'mast3r_checkpoint', 'mast3r_config'}
    assert evidence['dependencies']['typing_extensions']['path'] == str(typing_file)
    assert evidence['dependencies']['typing_extensions']['version'] == '4.15.0'
    assert probes and all(probe['typing_version'] == '4.15.0' for probe in probes)
    resources['mast3r_config_sha256'] = '0' * 64
    with pytest.raises(PreparationError, match='mast3r_config SHA-256 mismatch'):
        verify_frozen_overlay_identities(
            runtime=runtime, resources=resources, cache_root=tmp_path / 'cache',
            enforce_typing_extensions_home=False,
        )


def test_python_torch_version_probe_checks_typing_distribution_origin():
    import georeliab_mve.materialization as materialization

    source = Path(materialization.__file__).read_text(encoding='utf-8')
    assert "metadata.distribution('typing_extensions')" in source
    assert "dist.locate_file('{FROZEN_TYPING_EXTENSIONS_DIST_INFO}')" in source
    assert "dist.version == expected_version" in source


def test_frozen_identity_verifier_fails_closed_on_typing_extensions_hash_mismatch(tmp_path):
    from georeliab_mve.materialization import verify_typing_extensions_dependency

    site = tmp_path / 'typing-site-bad'
    site.mkdir()
    (site / 'typing_extensions.py').write_text('__version__ = "4.15.0"\n', encoding='utf-8')
    with pytest.raises(PreparationError, match='typing_extensions.py SHA-256 mismatch'):
        verify_typing_extensions_dependency(
            site=site,
            expected_sha256='0' * 64,
            expected_version='4.15.0',
            enforce_home_prefix=False,
        )


def test_frozen_identity_verifier_requires_exact_typing_extensions_site():
    from georeliab_mve.materialization import verify_typing_extensions_dependency

    with pytest.raises(PreparationError, match='must equal the frozen package cache'):
        verify_typing_extensions_dependency(
            site=FROZEN_TYPING_EXTENSIONS_SITE.parent / 'typing_extensions-4.15.0-pyhcf101f3_0-alias' / 'site-packages',
            expected_sha256='0' * 64,
            expected_version='4.15.0',
        )


def test_full_materialization_and_v2_provenance_contracts_fail_closed(monkeypatch, tmp_path):
    import georeliab_mve.materialization as materialization
    import georeliab_mve.tartanair_range as range_module

    sample, points = _write_local_dtu_archives(tmp_path)
    rect_raw, rectified = _zip_index(_rectified_names(), payload=None)
    image_raw, image_index = _zip_index(_tartan_names('image'), payload=None)
    depth_raw, depth_index = _zip_index(_tartan_names('depth'), payload=None)
    tartan_commit = 'e' * 40
    image_url = f'https://fixture/resolve/{tartan_commit}/image.zip'
    depth_url = f'https://fixture/resolve/{tartan_commit}/depth.zip'
    urls = {
        'https://fixture/rectified.zip': rect_raw,
        image_url: image_raw,
        depth_url: depth_raw,
    }
    indexes = {'Rectified.zip': rectified, 'tartanair-image': image_index,
               'tartanair-depth': depth_index}
    monkeypatch.setattr(
        materialization, 'validate_remote_indexes',
        lambda _resources: ({'schema_version': 'remote-zip-evidence-v1'}, indexes),
    )
    def request_range(url: str, start: int, end: int):
        raw = urls[url]
        return raw[start:end + 1], indexes[
            'Rectified.zip' if 'rectified' in url else
            ('tartanair-image' if 'image' in url else 'tartanair-depth')
        ].etag, len(raw)
    monkeypatch.setattr(range_module, '_request_range', request_range)

    scenes, provenance = build_dtu_archive_inventory(sample, points, rectified)
    split = build_split_view_manifest(scenes)
    split_path = tmp_path / 'manifests' / 'split_view_manifest.json'
    split.write(split_path)
    provenance_path = tmp_path / 'manifests' / 'dtu_inventory_provenance.json'
    provenance_path.write_text(json.dumps(provenance), encoding='utf-8')
    typing_site = tmp_path / 'typing-site-materialize'
    typing_site.mkdir()
    typing_file = typing_site / 'typing_extensions.py'
    typing_file.write_text('__version__ = "4.15.0"\n', encoding='utf-8')
    resources = {
        'typing_extensions_version': '4.15.0',
        'typing_extensions_sha256': hashlib.sha256(typing_file.read_bytes()).hexdigest(),
        'dtu_rectified_url': 'https://fixture/rectified.zip',
        'tartanair_image_url': image_url,
        'tartanair_depth_url': depth_url,
        'tartanair_hf_commit': tartan_commit,
        'dtu_sampleset_bytes': sample.stat().st_size,
        'dtu_sampleset_sha256': hashlib.sha256(sample.read_bytes()).hexdigest(),
        'dtu_points_bytes': points.stat().st_size,
        'dtu_points_sha256': hashlib.sha256(points.read_bytes()).hexdigest(),
    }
    result = materialize_frozen_selection(
        root=tmp_path, resources=resources, split_manifest_path=split_path,
        dtu_inventory_provenance_path=provenance_path,
        typing_extensions_site=typing_site,
        enforce_typing_extensions_home=False,
    )
    materialization_path = Path(result['materialization_path'])
    frozen = verify_materialization_manifest(
        materialization_path, split_manifest_path=split_path,
        enforce_typing_extensions_home=False,
    )
    assert result['dtu_rgb_count'] == 360
    assert result['tartanair_pair_count'] == 100
    assert frozen['dependencies']['typing_extensions']['path'] == str(typing_file)
    assert frozen['dependencies']['typing_extensions']['sha256'] == resources['typing_extensions_sha256']

    original_materialization = materialization_path.read_text(encoding='utf-8')
    camera_swap = json.loads(original_materialization)
    camera_keys = list(camera_swap['dtu'][0]['cameras'])
    first, second = camera_keys[:2]
    camera_swap['dtu'][0]['cameras'][first], camera_swap['dtu'][0]['cameras'][second] = (
        camera_swap['dtu'][0]['cameras'][second],
        camera_swap['dtu'][0]['cameras'][first],
    )
    materialization_path.write_text(json.dumps(camera_swap), encoding='utf-8')
    with pytest.raises(PreparationError, match='not exact pos_'):
        verify_materialization_manifest(
            materialization_path, split_manifest_path=split_path,
            enforce_typing_extensions_home=False,
        )
    materialization_path.write_text(original_materialization, encoding='utf-8')

    split_payload = json.loads(split_path.read_text(encoding='utf-8'))
    by_pair = {
        (int(scene['scene_id']), int(view)): {
            'rgb': rgb, 'camera': scene['cameras'][view],
            'points': scene['points'], 'mask': scene['mask'],
        }
        for scene in frozen['dtu'] if scene['split'] == 'dev'
        for view, rgb in scene['rgb'].items()
    }
    decoded = tmp_path / 'decoded'
    decoded.mkdir()
    prepared_records = []
    for scene in split_payload['splits']['dev']:
        for view in split_payload['views'][str(scene)]:
            expected = by_pair[(scene, view)]
            rgb_npy = decoded / f'scan{scene}_view{view}_rgb.npy'
            depth_npy = decoded / f'scan{scene}_view{view}_depth.npy'
            np.save(rgb_npy, np.full((4, 4, 3), 0.5, dtype=np.float64))
            np.save(depth_npy, np.full((4, 4), 2.0, dtype=np.float64))
            rgb_sha = hashlib.sha256(rgb_npy.read_bytes()).hexdigest()
            depth_sha = hashlib.sha256(depth_npy.read_bytes()).hexdigest()
            prepared_records.append({
                'scene_id': scene, 'view_id': view,
                'sample_key': f'dtu/dev/scan{scene}/view{view}/clean/0/0',
                'raw_rgb_path': expected['rgb']['path'],
                'raw_source_sha256': expected['rgb']['raw_sha256'],
                'linear_rgb_npy': str(rgb_npy), 'linear_rgb_npy_sha256': rgb_sha,
                'depth_npy': str(depth_npy), 'depth_npy_sha256': depth_sha,
                'gt_digest': depth_sha,
                'source_assets': {
                    name: {'member': expected[name]['member'],
                           'raw_sha256': expected[name]['raw_sha256']}
                    for name in ('rgb', 'camera', 'points', 'mask')
                },
                'depth_derivation': {
                    'algorithm': 'dtu-points-camera-projection-v1',
                    'input_sha256': {name: expected[name]['raw_sha256']
                                     for name in ('camera', 'points', 'mask')},
                    'output_sha256': depth_sha,
                },
            })
    prepared_payload = {
        'schema_version': 'prepared-input-v2', 'stage': 'smoke', 'split': 'dev',
        'split_view_manifest_path': str(split_path),
        'split_view_manifest_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
        'materialization_path': str(materialization_path),
        'materialization_sha256': hashlib.sha256(materialization_path.read_bytes()).hexdigest(),
        'records': prepared_records,
    }
    prepared_path = tmp_path / 'prepared-v2.json'
    prepared_path.write_text(json.dumps(prepared_payload), encoding='utf-8')
    # Round 4 removes the former trust gap: hand-authored decoded arrays are
    # not a production artifact, even when every self-digest is updated.
    with pytest.raises(PreparationError, match='producer/dependency recipe mismatch'):
        load_prepared_batch(prepared_path, expected_stage='smoke')
