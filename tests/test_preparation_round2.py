from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from georeliab_mve.prepare_cli import run_prepare_operation
from georeliab_mve.preparation import (
    CalibrationError,
    PreparationError,
    calibration_qa,
    calibrate_corruptions,
)


def _overlay(path: Path) -> Path:
    path.write_text(
        "[runtime]\nroot = '/srv/private/smli/GeoReliab'\n"
        "vggt_source = '/home/smli/vggt'\nmast3r_source = '/home/smli/mast3r'\n"
        "vggt_env = '/home/smli/env-vggt'\nmast3r_env = '/home/smli/env-mast3r'\nvggt_python = '3.10.20'\nvggt_torch = '2.3.1+cu121'\nmast3r_python = '3.10.20'\nmast3r_torch = '2.5.1+cu121'\n"
        "[resources]\n"
        "dtu_sampleset_url = 'https://example.test/SampleSet.zip'\n"
        "dtu_sampleset_bytes = 1\ndtu_sampleset_sha256 = '" + "0" * 64 + "'\n"
        "dtu_points_url = 'https://example.test/Points.zip'\n"
        "dtu_points_bytes = 1\ndtu_points_sha256 = '" + "1" * 64 + "'\n"
        "dtu_rectified_url = 'https://example.test/Rectified.zip'\n"
        "dtu_rectified_bytes = 2\ndtu_rectified_etag = 'etag'\n"
        "tartanair_image_url = 'https://example.test/image.zip'\n"
        "tartanair_image_bytes = 2\ntartanair_image_etag = 'image-etag'\n"
        "tartanair_depth_url = 'https://example.test/depth.zip'\n"
        "tartanair_depth_bytes = 2\ntartanair_depth_etag = 'depth-etag'\n"
        "tartanair_hf_commit = '0d2d145e973832742a2aaa04b7d2ebffc8d82817'\n"
        "vggt_checkpoint = '/home/smli/model.pt'\nvggt_checkpoint_sha256 = '" + "2" * 64 + "'\n"
        "vggt_source_commit = 'a288dd0f14786c93483e45524328726ab7b1b4ce'\n"
        "mast3r_checkpoint = '/home/smli/mast3r.safetensors'\nmast3r_checkpoint_sha256 = '" + "3" * 64 + "'\n"
        "mast3r_config = '/home/smli/config.json'\nmast3r_config_sha256 = '" + "4" * 64 + "'\n"
        "mast3r_source_commit = 'f5209afc300cec36239a7ac992263f36847bbba0'\n"
        "dust3r_source_commit = '3cc8c88c413bb9e34c41db0e0eef99c2ee010b12'\n"
        "croco_source_commit = 'd7de0705845239092414480bd829228723bf20de'\n"
        "[execution]\ndevice = 'cuda:0'\n",
        encoding='utf-8',
    )
    return path


def _prepared_contract(root: Path) -> None:
    prepared = root / 'prepared'
    prepared.mkdir(parents=True)
    image = np.linspace(0.1, 0.9, 64 * 64 * 3).reshape(64, 64, 3)
    depth = np.tile(np.linspace(1.0, 4.0, 64), (64, 1))
    records = []
    calibration_ids = [115, 107, 82, 45, 117, 61, 127, 83, 56, 92]
    for scene in calibration_ids:
        rgb = prepared / f'{scene}_rgb.npy'
        z = prepared / f'{scene}_depth.npy'
        np.save(rgb, image)
        np.save(z, depth)
        records.append({'scene_id': scene, 'view_id': 1, 'sample_key': f'scan{scene}',
                        'linear_rgb_npy': str(rgb), 'depth_npy': str(z),
                        'raw_source_sha256': hashlib.sha256(rgb.read_bytes()).hexdigest(),
                        'gt_digest': hashlib.sha256(z.read_bytes()).hexdigest()})
    (prepared / 'calibration_inputs.json').write_text(json.dumps({'schema_version': 'prepared-input-v1', 'records': records}), encoding='utf-8')
    (prepared / 'render_inputs.json').write_text(json.dumps({'schema_version': 'prepared-input-v1', 'records': records[:1]}), encoding='utf-8')
    frames = []
    for index in range(100):
        rgb = prepared / f'tartan_{index}_rgb.npy'
        z = prepared / f'tartan_{index}_depth.npy'
        np.save(rgb, 1.0 / depth[..., None] * np.ones(3))
        np.save(z, depth)
        frames.append({'rgb_npy': str(rgb), 'depth_npy': str(z)})
    (prepared / 'tartanair_p000_pairs.json').write_text(json.dumps({'schema_version': 'tartanair-p000-v1', 'pairs': frames}), encoding='utf-8')


def test_calibration_qa_rejects_each_required_nonmonotonic_failure():
    image = np.dstack([np.tile(np.linspace(0.1, 0.9, 64), (64, 1))] * 3)
    depth = np.tile(np.linspace(1.0, 4.0, 64), (64, 1))
    calibration = calibrate_corruptions([depth], [image])
    valid = calibration_qa(calibration, [(1, image, depth, '0' * 64, '1' * 64)])
    assert valid['passed'] is True
    for key in ('fog', 'low_light', 'defocus', 'gt', 'cross_view', 'synthetic_fog'):
        broken = dict(valid)
        broken['checks'] = dict(valid['checks'])
        broken['checks'][key] = False
        with pytest.raises(CalibrationError, match=key):
            calibration_qa(calibration, [(1, image, depth, '0' * 64, '1' * 64)], expected=broken)


def test_non_dry_preparation_calibration_rendering_and_sanity_have_success_paths(tmp_path):
    root = tmp_path / 'data'
    _prepared_contract(root)
    state = tmp_path / 'state.json'
    calibration = run_prepare_operation(operation='calibration', data_root=root, state_path=state, dry_run=False, overlay_path=None)
    assert calibration['resources_ready'] is True
    rendering = run_prepare_operation(operation='rendering', data_root=root, state_path=state, dry_run=False, overlay_path=None)
    assert rendering['rendered_count'] == 9
    sanity = run_prepare_operation(operation='sanity', data_root=root, state_path=state, dry_run=False, overlay_path=None)
    assert sanity['reason_code'] == 'TARTANAIR_NATIVE_FOG_SANITY'


def test_overlay_requires_every_frozen_identity(tmp_path):
    overlay = _overlay(tmp_path / 'overlay.toml')
    text = overlay.read_text(encoding='utf-8').replace("tartanair_depth_etag = 'depth-etag'\n", '')
    overlay.write_text(text, encoding='utf-8')
    with pytest.raises(PreparationError, match='missing required frozen identities'):
        run_prepare_operation(operation='download', data_root=tmp_path / 'data', state_path=tmp_path / 's.json', dry_run=False, overlay_path=overlay)


def test_non_dry_download_verify_index_and_manifests_have_success_paths(monkeypatch, tmp_path):
    import georeliab_mve.prepare_dispatch_round1 as dispatch
    from georeliab_mve.preparation import DtuScene
    from georeliab_mve.tartanair_range import RemoteZipIndex

    root = tmp_path / 'data'
    overlay = _overlay(tmp_path / 'overlay.toml')
    state = tmp_path / 'state.json'
    def fake_download(url, destination, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b'x')
        return destination
    def fake_verify(path, **kwargs):
        return {'path': str(path), 'bytes': kwargs.get('expected_bytes'), 'sha256': kwargs.get('expected_sha256', 'x'), 'entries': 1}
    def fake_index(url):
        if 'image' in url:
            return RemoteZipIndex(2, '"image-etag"', {'a': object()}, 'a' * 64)
        if 'depth' in url:
            return RemoteZipIndex(2, 'W/"depth-etag"', {'a': object()}, 'b' * 64)
        return RemoteZipIndex(2, '"etag"', {'a': object()}, 'c' * 64)
    monkeypatch.setattr(dispatch, 'download_archive', fake_download)
    monkeypatch.setattr(dispatch, 'verify_archive', fake_verify)
    monkeypatch.setattr(dispatch, 'index_remote_zip', fake_index)
    assert run_prepare_operation(operation='download', data_root=root, state_path=state, dry_run=False, overlay_path=overlay)['resources_ready'] is True
    assert run_prepare_operation(operation='verify', data_root=root, state_path=state, dry_run=False, overlay_path=overlay)['resources_ready'] is True

    ids = list(range(1, 78)) + list(range(82, 129))
    scenes = tuple(DtuScene(scene, tuple(f'rect_{view:03d}_3_r5000.png' for view in range(1, 50)),
                     {view: np.array([float(view), 0.0, 1.0]) for view in range(1, 50)},
                     f'Points/stl/stl{scene:03d}_total.ply', f'ObsMask/ObsMask{scene}_10.mat') for scene in ids)
    monkeypatch.setattr(dispatch, 'parse_dtu_inventory', lambda _: scenes)
    assert run_prepare_operation(operation='index', data_root=root, state_path=state, dry_run=False, overlay_path=None)['scene_count'] == len(scenes)
    manifest = run_prepare_operation(operation='manifests', data_root=root, state_path=state, dry_run=False, overlay_path=None)
    assert len(json.loads(Path(manifest['manifest_path']).read_text())['views']) == 45
