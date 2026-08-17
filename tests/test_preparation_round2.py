from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from georeliab_mve.cli import main
from georeliab_mve.prepare_cli import run_prepare_operation
from georeliab_mve.preparation import (
    CalibrationError,
    PreparationError,
    calibration_qa,
    calibrate_corruptions,
)
from georeliab_mve.preparation_round2 import PreparedBatch


def _patchwise_depth(values: list[float]) -> np.ndarray:
    assert len(values) == 4
    depth = np.zeros((64, 64), dtype=np.float64)
    for index, value in enumerate(values):
        y = 32 * (index // 2)
        x = 32 * (index % 2)
        depth[y:y + 32, x:x + 32] = value
    return depth


def _patchwise_contrast_image(values: list[float], *, base: float = 0.5) -> np.ndarray:
    assert len(values) == 4
    image = np.zeros((64, 64, 3), dtype=np.float64)
    pattern = ((np.indices((32, 32)).sum(axis=0) % 2) * 2 - 1).astype(np.float64)
    for index, value in enumerate(values):
        y = 32 * (index // 2)
        x = 32 * (index % 2)
        patch = np.clip(base + float(value) * pattern, 0.0, 1.0)
        image[y:y + 32, x:x + 32] = patch[..., None]
    return image


def _fake_fog_renderer_by_image(monkeypatch, mapping: dict[tuple[int, int], np.ndarray]) -> None:
    import georeliab_mve.preparation_round2 as preparation_round2

    def fake_fog(image, depth, calibration, *, severity, gt_digest, raw_source_sha256):
        rendered = mapping[(id(image), int(severity))]
        metadata = {
            'raw_source_sha256': raw_source_sha256,
            'rendered_png_sha256': hashlib.sha256(rendered.tobytes()).hexdigest(),
            'gt_digest': gt_digest,
            'parameter_manifest_sha256': calibration.manifest().sha256,
            'implementation_version': calibration.implementation_version,
            'corruption': 'fog',
            'severity': severity,
            'beta': calibration.fog_betas[severity - 1],
            'realized_transmittance': [0.8, 0.5, 0.25][severity - 1],
        }
        return rendered, metadata

    monkeypatch.setattr(preparation_round2, 'fog_render', fake_fog)


def _passing_synthetic_fog_record(monkeypatch, *, scene_id: int = 1):
    depth = _patchwise_depth([1.0, 2.0, 3.0, 4.0])
    image = _patchwise_contrast_image([0.01, 0.02, 0.03, 0.04])
    calibration = calibrate_corruptions([depth], [image])
    mapping = {
        (id(image), 1): _patchwise_contrast_image([0.01, 0.01, 0.02, 0.04]),
        (id(image), 2): _patchwise_contrast_image([0.04, 0.01, 0.02, 0.03]),
        (id(image), 3): _patchwise_contrast_image([0.04, 0.03, 0.02, 0.01]),
    }
    _fake_fog_renderer_by_image(monkeypatch, mapping)
    return calibration, (scene_id, 1, image, depth, '0' * 64, '1' * 64)


def _overlay(path: Path) -> Path:
    runtime_root = Path('/srv/private/georeliab-test')
    path.write_text(
        "[runtime]\nroot = '" + str(runtime_root) + "'\n"
        "vggt_source = '/home/smli/vggt'\nmast3r_source = '/home/smli/mast3r'\n"
        "dust3r_source = '/home/smli/mast3r/dust3r'\ncroco_source = '/home/smli/mast3r/dust3r/croco'\n"
        "vggt_env = '/home/smli/env-vggt'\nmast3r_env = '/home/smli/env-mast3r'\nvggt_python = '3.10.20'\nvggt_torch = '2.3.1+cu121'\nmast3r_python = '3.10.20'\nmast3r_torch = '2.5.1+cu121'\ntyping_extensions_site = '/home/smli/miniforge3/pkgs/typing_extensions-4.15.0-pyhcf101f3_0/site-packages'\n"
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
        "croco_source_commit = 'd7de0705845239092414480bd829228723bf20de'\ntyping_extensions_version = '4.15.0'\ntyping_extensions_sha256 = '433d11d170d3a24d2eb065ebc1bfe848cea7e3d7ce68567ab52bea2d4c2f7ed8'\n"
        "[execution]\ndevice = 'cuda:0'\n",
        encoding='utf-8',
    )
    return path


@pytest.fixture
def _validated_local_overlay_root(monkeypatch):
    """Keep the production root guard while redirecting test writes to tmp_path."""
    import georeliab_mve.prepare_dispatch_round1 as dispatch

    load_validated_overlay = dispatch._overlay

    def load_local_overlay(path: Path):
        overlay = load_validated_overlay(path)
        return type(overlay)(
            str(path.parent / 'data'),
            overlay.resources,
            overlay.execution,
            overlay.source,
        )

    monkeypatch.setattr(dispatch, '_overlay', load_local_overlay)


def test_calibration_qa_rejects_each_required_nonmonotonic_failure(monkeypatch):
    calibration, record = _passing_synthetic_fog_record(monkeypatch)
    valid = calibration_qa(calibration, [record])
    assert valid['passed'] is True
    for key in ('fog', 'low_light', 'defocus', 'gt', 'cross_view', 'synthetic_fog'):
        broken = dict(valid)
        broken['checks'] = dict(valid['checks'])
        broken['checks'][key] = False
        with pytest.raises(CalibrationError, match=key):
            calibration_qa(calibration, [record], expected=broken)


def test_calibration_qa_uses_pre_noise_brightness(monkeypatch):
    import georeliab_mve.preparation_round2 as preparation_round2

    calibration, record = _passing_synthetic_fog_record(monkeypatch)
    image = record[2]
    actual_render = preparation_round2.low_light_noise_render

    def render_with_nonmonotonic_output(*args, severity, **kwargs):
        rendered, metadata = actual_render(*args, severity=severity, **kwargs)
        expected_brightness = float(np.clip(args[0], 0.0, 1.0).mean() * metadata['exposure'])
        assert metadata['pre_noise_brightness'] == pytest.approx(expected_brightness)
        return np.full_like(rendered, float(severity)), metadata

    monkeypatch.setattr(
        preparation_round2,
        'low_light_noise_render',
        render_with_nonmonotonic_output,
    )
    result = calibration_qa(
        calibration,
        [record],
    )

    assert result['passed'] is True
    assert result['metrics'][0]['brightness'][0] > result['metrics'][0]['brightness'][1]
    assert result['metrics'][0]['brightness'][1] > result['metrics'][0]['brightness'][2]


def test_calibration_qa_rejects_rank_invariant_fog_fixture():
    '''A severity label cannot substitute for observed contrast degradation.'''
    image = np.dstack([np.tile(np.linspace(0.1, 0.9, 64), (64, 1))] * 3)
    depth = np.tile(np.linspace(1.0, 4.0, 64), (64, 1))
    calibration = calibrate_corruptions([depth], [image])
    with pytest.raises(CalibrationError, match='synthetic_fog'):
        calibration_qa(
            calibration,
            [(1, 1, image, depth, '0' * 64, '1' * 64)],
        )


def test_synthetic_fog_qa_uses_scene_level_patch_effects(monkeypatch):
    '''One view can be non-negative when the scene-level multi-view effect is negative and monotone.'''
    depth = _patchwise_depth([1.0, 2.0, 3.0, 4.0])
    image_a = _patchwise_contrast_image([0.01, 0.02, 0.03, 0.04])
    image_b = _patchwise_contrast_image([0.01, 0.02, 0.03, 0.04])
    image_c = _patchwise_contrast_image([0.01, 0.02, 0.03, 0.04])
    image_d = _patchwise_contrast_image([0.01, 0.02, 0.03, 0.04])
    calibration = calibrate_corruptions([depth], [image_a])
    mapping = {
        (id(image_a), 1): _patchwise_contrast_image([0.01, 0.02, 0.03, 0.04]),
        (id(image_a), 2): _patchwise_contrast_image([0.01, 0.02, 0.03, 0.04]),
        (id(image_a), 3): _patchwise_contrast_image([0.01, 0.02, 0.03, 0.04]),
    }
    for image in (image_b, image_c, image_d):
        mapping[(id(image), 1)] = _patchwise_contrast_image([0.04, 0.035, 0.03, 0.025])
        mapping[(id(image), 2)] = _patchwise_contrast_image([0.04, 0.03, 0.02, 0.01])
        mapping[(id(image), 3)] = _patchwise_contrast_image([0.08, 0.06, 0.04, 0.02])
    _fake_fog_renderer_by_image(monkeypatch, mapping)

    result = calibration_qa(
        calibration,
        [
            (83, 1, image_a, depth, '0' * 64, '1' * 64),
            (83, 2, image_b, depth, '0' * 64, '1' * 64),
            (83, 3, image_c, depth, '0' * 64, '1' * 64),
            (83, 4, image_d, depth, '0' * 64, '1' * 64),
        ],
    )

    scene_metric = result['synthetic_fog_metrics'][0]
    assert result['checks']['synthetic_fog'] is True
    assert scene_metric['scene_id'] == 83
    assert scene_metric['view_count'] == 4
    assert scene_metric['patch_counts']['clean'] == 16
    assert all(value < 0.0 for value in scene_metric['effects'])
    assert abs(scene_metric['effects'][0]) < abs(scene_metric['effects'][1]) < abs(scene_metric['effects'][2])


def test_synthetic_fog_qa_uses_32x32_patch_local_rms(monkeypatch):
    depth = _patchwise_depth([1.0, 2.0, 3.0, 4.0])
    image = _patchwise_contrast_image([0.01, 0.02, 0.03, 0.04])
    calibration = calibrate_corruptions([depth], [image])
    mapping = {
        (id(image), 1): _patchwise_contrast_image([0.01, 0.01, 0.02, 0.04]),
        (id(image), 2): _patchwise_contrast_image([0.04, 0.01, 0.02, 0.03]),
        (id(image), 3): _patchwise_contrast_image([0.04, 0.03, 0.02, 0.01]),
    }
    _fake_fog_renderer_by_image(monkeypatch, mapping)

    result = calibration_qa(calibration, [(56, 1, image, depth, '0' * 64, '1' * 64)])

    scene_metric = result['synthetic_fog_metrics'][0]
    assert scene_metric['patch_counts']['clean'] == 4
    assert scene_metric['clean_rho'] == pytest.approx(1.0)
    assert scene_metric['fog_rhos'][2] == pytest.approx(-1.0)


def test_synthetic_fog_qa_pools_single_patch_views_at_scene_grain(monkeypatch):
    import georeliab_mve.preparation_round2 as preparation_round2

    records = []
    mapping = {}
    clean_images = []
    for view_id, (depth_value, clean_contrast) in enumerate(
        [(1.0, 0.01), (2.0, 0.02), (3.0, 0.03), (4.0, 0.04)],
        start=1,
    ):
        depth = np.full((64, 64), np.nan, dtype=np.float64)
        depth[:32, :32] = depth_value
        image = _patchwise_contrast_image([clean_contrast, 0.01, 0.01, 0.01])
        clean_images.append(image)
        records.append((82, view_id, image, depth, '0' * 64, '1' * 64))
        fog_values = {
            1: [0.01, 0.01, 0.02, 0.04][view_id - 1],
            2: [0.04, 0.01, 0.02, 0.03][view_id - 1],
            3: [0.04, 0.03, 0.02, 0.01][view_id - 1],
        }
        for severity, contrast in fog_values.items():
            mapping[(id(image), severity)] = _patchwise_contrast_image([contrast, 0.01, 0.01, 0.01])
    calibration = calibrate_corruptions([np.nan_to_num(records[0][3], nan=1.0)], [clean_images[0]])
    _fake_fog_renderer_by_image(monkeypatch, mapping)
    monkeypatch.setattr(
        preparation_round2,
        'render_defocus',
        lambda image, depth, calibration, *, severity, gt_digest, raw_source_sha256: (
            image,
            {
                'gt_digest': gt_digest,
                'parameter_manifest_sha256': calibration.manifest().sha256,
                'focus_depth': calibration.d_ref,
                'inverse_depth_layers': 32,
                'defocus_scale': severity,
                'coc_p95': float(severity),
                'edge_energy_loss': float(severity),
            },
        ),
    )

    result = calibration_qa(calibration, records)

    scene_metric = result['synthetic_fog_metrics'][0]
    assert scene_metric['view_count'] == 4
    assert scene_metric['patch_counts']['clean'] == 4
    assert scene_metric['clean_rho'] == pytest.approx(1.0)
    assert result['checks']['synthetic_fog'] is True


def test_synthetic_fog_qa_rejects_any_nonnegative_scene_effect(monkeypatch):
    depth = _patchwise_depth([1.0, 2.0, 3.0, 4.0])
    good = _patchwise_contrast_image([0.01, 0.02, 0.03, 0.04])
    bad = _patchwise_contrast_image([0.01, 0.02, 0.03, 0.04])
    calibration = calibrate_corruptions([depth], [good])
    mapping = {
        (id(good), 1): _patchwise_contrast_image([0.04, 0.03, 0.02, 0.01]),
        (id(good), 2): _patchwise_contrast_image([0.04, 0.025, 0.015, 0.01]),
        (id(good), 3): _patchwise_contrast_image([0.04, 0.02, 0.01, 0.005]),
        (id(bad), 1): _patchwise_contrast_image([0.01, 0.02, 0.03, 0.04]),
        (id(bad), 2): _patchwise_contrast_image([0.01, 0.02, 0.03, 0.04]),
        (id(bad), 3): _patchwise_contrast_image([0.01, 0.02, 0.03, 0.04]),
    }
    _fake_fog_renderer_by_image(monkeypatch, mapping)

    with pytest.raises(CalibrationError, match='synthetic_fog'):
        calibration_qa(
            calibration,
            [
                (45, 1, good, depth, '0' * 64, '1' * 64),
                (117, 1, bad, depth, '0' * 64, '1' * 64),
            ],
        )


def test_synthetic_fog_qa_rejects_nonmonotone_scene_strength(monkeypatch):
    depth = _patchwise_depth([1.0, 2.0, 3.0, 4.0])
    image = _patchwise_contrast_image([0.01, 0.02, 0.03, 0.04])
    calibration = calibrate_corruptions([depth], [image])
    mapping = {
        (id(image), 1): _patchwise_contrast_image([0.04, 0.03, 0.02, 0.01]),
        (id(image), 2): _patchwise_contrast_image([0.04, 0.03, 0.02, 0.01]),
        (id(image), 3): _patchwise_contrast_image([0.04, 0.02, 0.01, 0.005]),
    }
    _fake_fog_renderer_by_image(monkeypatch, mapping)

    with pytest.raises(CalibrationError, match='synthetic_fog'):
        calibration_qa(calibration, [(61, 1, image, depth, '0' * 64, '1' * 64)])


def test_synthetic_fog_qa_fails_closed_for_too_few_patches_and_constant_contrast(monkeypatch):
    import georeliab_mve.preparation_round2 as preparation_round2

    depth = np.full((64, 64), np.nan, dtype=np.float64)
    depth[:32, :32] = 1.0
    image = np.full((64, 64, 3), 0.5, dtype=np.float64)
    calibration = calibrate_corruptions([np.nan_to_num(depth, nan=1.0)], [image])
    mapping = {
        (id(image), 1): image,
        (id(image), 2): image,
        (id(image), 3): image,
    }
    _fake_fog_renderer_by_image(monkeypatch, mapping)
    monkeypatch.setattr(
        preparation_round2,
        'render_defocus',
        lambda image, depth, calibration, *, severity, gt_digest, raw_source_sha256: (
            image,
            {
                'gt_digest': gt_digest,
                'parameter_manifest_sha256': calibration.manifest().sha256,
                'focus_depth': calibration.d_ref,
                'inverse_depth_layers': 32,
                'defocus_scale': severity,
                'coc_p95': float(severity),
                'edge_energy_loss': float(severity),
            },
        ),
    )

    with pytest.raises(CalibrationError, match='synthetic_fog'):
        calibration_qa(calibration, [(127, 1, image, depth, '0' * 64, '1' * 64)])


def test_synthetic_fog_qa_rejects_unchanged_contrast(monkeypatch):
    depth = _patchwise_depth([1.0, 2.0, 3.0, 4.0])
    image = _patchwise_contrast_image([0.01, 0.02, 0.03, 0.04])
    calibration = calibrate_corruptions([depth], [image])
    mapping = {
        (id(image), 1): image.copy(),
        (id(image), 2): image.copy(),
        (id(image), 3): image.copy(),
    }
    _fake_fog_renderer_by_image(monkeypatch, mapping)

    with pytest.raises(CalibrationError, match='synthetic_fog'):
        calibration_qa(calibration, [(92, 1, image, depth, '0' * 64, '1' * 64)])


def test_non_dry_preparation_calibration_rendering_and_sanity_have_success_paths(monkeypatch, tmp_path):
    import georeliab_mve.prepare_dispatch_round1 as dispatch

    root = tmp_path / 'data'
    prepared = root / 'prepared'
    prepared.mkdir(parents=True)
    for name in ('calibration_inputs.json', 'render_inputs_smoke.json', 'tartanair_p000_pairs.json'):
        (prepared / name).write_text('{}', encoding='utf-8')
    image = np.random.default_rng(9).uniform(0.1, 0.9, (64, 64, 3))
    depth = np.tile(np.linspace(1.0, 8.0, 64), (64, 1))
    calibration_records = tuple(
        (scene, 1, f'calibration/scan{scene}/view1', image, depth, '0' * 64, '1' * 64)
        for scene in (115, 107, 82, 45, 117, 61, 127, 83, 56, 92)
    )
    calibration_batch = PreparedBatch('calibration', 'calibration', '2' * 64, '3' * 64, calibration_records)
    smoke_batch = PreparedBatch('smoke', 'dev', '2' * 64, '3' * 64, (calibration_records[0],))
    monkeypatch.setattr(dispatch, 'load_prepared_batch', lambda path, expected_stage=None: calibration_batch if 'calibration' in path.name else smoke_batch)
    def passing_qa(calibration, _records):
        return {'schema_version': 'calibration-qa-v1', 'passed': True,
                'checks': {'synthetic_fog': True},
                'parameter_manifest_sha256': calibration.manifest().sha256,
                'metrics': []}
    monkeypatch.setattr(dispatch, 'calibration_qa', passing_qa)
    monkeypatch.setattr(dispatch, 'load_tartanair_prepared_pairs', lambda _path: [
        (f'{index:06d}', 1.0 / depth[..., None] * np.ones(3), depth)
        for index in range(100)
    ])
    state = tmp_path / 'state.json'
    calibration = run_prepare_operation(operation='calibration', data_root=root, state_path=state, dry_run=False, overlay_path=None)
    assert calibration['resources_ready'] is False
    rendering = run_prepare_operation(operation='rendering', stage='smoke', data_root=root, state_path=state, dry_run=False, overlay_path=None)
    assert rendering['rendered_count'] == 10
    sanity = run_prepare_operation(operation='sanity', data_root=root, state_path=state, dry_run=False, overlay_path=None)
    assert sanity['reason_code'] == 'TARTANAIR_NATIVE_FOG_SANITY'


def test_test_render_lock_and_existing_artifacts_are_immutable(monkeypatch, tmp_path):
    import georeliab_mve.prepare_dispatch_round1 as dispatch

    root = tmp_path / 'data'
    manifests = root / 'manifests'
    prepared = root / 'prepared'
    manifests.mkdir(parents=True)
    prepared.mkdir()
    image = np.random.default_rng(12).uniform(0.1, 0.9, (16, 16, 3))
    depth = np.tile(np.linspace(1.0, 4.0, 16), (16, 1))
    calibration = calibrate_corruptions([depth], [image])
    calibration.manifest().write(manifests / 'corruption_calibration.json')
    (manifests / 'corruption_calibration_qa.json').write_text(json.dumps({
        'passed': True,
        'parameter_manifest_sha256': calibration.manifest().sha256,
        'split_view_manifest_sha256': '2' * 64,
        'materialization_sha256': '3' * 64,
    }), encoding='utf-8')
    render_input = prepared / 'render_inputs_test.json'
    render_input.write_text('{"frozen":true}', encoding='utf-8')
    record = (1, 1, 'dtu/test/scan1/view1/clean/0/0', image, depth, '0' * 64, '1' * 64)
    batch = PreparedBatch('test', 'test', '2' * 64, '3' * 64, (record,))
    monkeypatch.setattr(dispatch, 'load_prepared_batch', lambda _path, expected_stage=None: batch)
    first = dispatch._run_rendering(root, stage='test')
    assert first['written'] == 10 and first['reused'] == 0
    second = dispatch._run_rendering(root, stage='test')
    assert second['written'] == 0 and second['reused'] == 10
    original_input = render_input.read_text(encoding='utf-8')
    render_input.write_text('{"frozen":false}', encoding='utf-8')
    with pytest.raises(PreparationError, match='changed after freeze'):
        dispatch._run_rendering(root, stage='test')
    render_input.write_text(original_input, encoding='utf-8')
    rendered = next((root / 'rendered' / 'test').glob('*.png'))
    rendered.write_bytes(b'tampered')
    with pytest.raises(PreparationError, match='tampered'):
        dispatch._run_rendering(root, stage='test')


def test_overlay_requires_every_frozen_identity(
    tmp_path, _validated_local_overlay_root
):
    overlay = _overlay(tmp_path / 'overlay.toml')
    text = overlay.read_text(encoding='utf-8').replace("tartanair_depth_etag = 'depth-etag'\n", '')
    overlay.write_text(text, encoding='utf-8')
    with pytest.raises(PreparationError, match='missing required frozen identities'):
        run_prepare_operation(operation='download', data_root=tmp_path / 'data', state_path=tmp_path / 's.json', dry_run=False, overlay_path=overlay)


def test_overlay_requires_frozen_typing_extensions_identity(
    tmp_path, _validated_local_overlay_root
):
    overlay = _overlay(tmp_path / 'overlay.toml')
    text = overlay.read_text(encoding='utf-8').replace("typing_extensions_sha256 = '433d11d170d3a24d2eb065ebc1bfe848cea7e3d7ce68567ab52bea2d4c2f7ed8'\n", '')
    overlay.write_text(text, encoding='utf-8')
    with pytest.raises(PreparationError, match='typing_extensions_sha256'):
        run_prepare_operation(operation='verify', data_root=tmp_path / 'data', state_path=tmp_path / 's.json', dry_run=False, overlay_path=overlay)


def test_non_dry_download_verify_index_and_manifests_have_success_paths(
    monkeypatch, tmp_path, _validated_local_overlay_root
):
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
    empty_indexes = {
        name: RemoteZipIndex(2, 'etag', {}, 'a' * 64)
        for name in ('Rectified.zip', 'tartanair-image', 'tartanair-depth')
    }
    remote_evidence = {'schema_version': 'remote-zip-evidence-v1',
                       'remote_indexes': [], 'tartanair_selected_frame_ids': []}
    monkeypatch.setattr(dispatch, 'download_archive', fake_download)
    monkeypatch.setattr(dispatch, 'verify_archive', fake_verify)
    monkeypatch.setattr(dispatch, 'validate_remote_indexes', lambda _resources: (remote_evidence, empty_indexes))
    monkeypatch.setattr(dispatch, 'verify_frozen_overlay_identities', lambda **_kwargs: {'schema_version': 'frozen-runtime-identity-v1'})
    download = run_prepare_operation(operation='download', data_root=root, state_path=state, dry_run=False, overlay_path=overlay)
    assert download['resources_ready'] is False
    assert download['schema_version'] == 'preparation-state-v5'
    verify = run_prepare_operation(operation='verify', data_root=root, state_path=state, dry_run=False, overlay_path=overlay)
    assert verify['resources_ready'] is True
    assert verify['schema_version'] == 'preparation-state-v5'

    ids = list(range(1, 78)) + list(range(82, 129))
    scenes = tuple(DtuScene(scene, tuple(f'rect_{view:03d}_3_r5000.png' for view in range(1, 50)),
                     {view: np.array([float(view), 0.0, 1.0]) for view in range(1, 50)},
                     f'Points/stl/stl{scene:03d}_total.ply', f'ObsMask/ObsMask{scene}_10.mat') for scene in ids)
    provenance = {'schema_version': 'dtu-archive-inventory-provenance-v1', 'camera_members': {}, 'scenes': []}
    monkeypatch.setattr(dispatch, 'build_dtu_archive_inventory', lambda *_args: (scenes, provenance))
    assert run_prepare_operation(operation='index', data_root=root, state_path=state, dry_run=False, overlay_path=overlay)['scene_count'] == len(scenes)
    def fake_materialize(**kwargs):
        path = root / 'manifests' / 'frozen_materialization.json'
        path.write_text('{}', encoding='utf-8')
        return {'materialization_path': str(path), 'materialization_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'dtu_scene_count': 45, 'dtu_rgb_count': 360, 'tartanair_pair_count': 100}
    monkeypatch.setattr(dispatch, 'materialize_frozen_selection', fake_materialize)
    monkeypatch.setattr(dispatch, 'verify_materialization_manifest', lambda *_args, **_kwargs: {})
    manifest = run_prepare_operation(operation='manifests', data_root=root, state_path=state, dry_run=False, overlay_path=overlay)
    assert len(json.loads(Path(manifest['manifest_path']).read_text())['views']) == 45


@pytest.mark.parametrize('operation', ('download', 'verify'))
def test_prepare_writer_keeps_state_contract_authoritative(
    monkeypatch, tmp_path, operation, _validated_local_overlay_root
):
    import georeliab_mve.prepare_dispatch_round1 as dispatch

    root = tmp_path / 'data'
    overlay = _overlay(tmp_path / 'overlay.toml')
    state = tmp_path / f'{operation}.json'
    result = {
        'schema_version': 'remote-zip-evidence-v1',
        'operation': 'wrong-operation',
        'stage': 'test',
        'data_root': '/wrong/root',
        'dry_run': True,
        'scientific_ready': True,
        'resources_ready': 'wrong',
        'state_transition': 'wrong:completed',
        'remote_indexes': [
            {
                'name': 'Rectified.zip',
                'url': 'https://example.test/Rectified.zip',
                'bytes': 2,
                'etag': 'etag',
                'central_directory_sha256': 'a' * 64,
                'member_count': 360,
                'required_member_count': 360,
            }
        ],
        'tartanair_selected_frame_ids': ['000000'],
    }
    monkeypatch.setattr(
        dispatch,
        f'_run_{operation}',
        lambda *_args, **_kwargs: dict(result),
    )

    payload = run_prepare_operation(
        operation=operation,
        data_root=root,
        state_path=state,
        dry_run=False,
        overlay_path=overlay,
    )

    assert json.loads(state.read_text(encoding='utf-8')) == payload
    assert payload['schema_version'] == 'preparation-state-v5'
    assert payload['operation'] == operation
    assert payload['stage'] is None
    assert payload['data_root'] == str(root)
    assert payload['dry_run'] is False
    assert payload['scientific_ready'] is False
    assert payload['resources_ready'] is (operation == 'verify')
    assert payload['state_transition'] == f'{operation}:completed'
    assert payload['remote_indexes'] == result['remote_indexes']


def test_prepared_operation_invokes_production_writer(
    monkeypatch, tmp_path, _validated_local_overlay_root
):
    import georeliab_mve.prepare_dispatch_round1 as dispatch

    root = tmp_path / 'data'
    overlay = _overlay(tmp_path / 'overlay.toml')
    state = tmp_path / 'state.json'
    expected = {
        'calibration_record_count': 80,
        'smoke_record_count': 80,
        'test_record_count': 160,
        'tartanair_record_count': 100,
    }
    calls = []

    def fake_writer(actual_root):
        calls.append(actual_root)
        return expected

    monkeypatch.setattr(dispatch, 'write_prepared_inputs', fake_writer)
    result = run_prepare_operation(
        operation='prepared', data_root=root, state_path=state,
        dry_run=False, overlay_path=overlay,
    )
    assert calls == [root]
    assert {key: result[key] for key in expected} == expected
    assert result['state_transition'] == 'prepared:completed'
    assert json.loads(state.read_text(encoding='utf-8')) == result


def test_prepare_cli_requires_stage_only_for_rendering(tmp_path):
    root = tmp_path / 'data'
    state = tmp_path / 'state.json'
    base = ['prepare-georeliab', '--data-root', str(root), '--state', str(state), '--dry-run']

    assert main([*base, '--operation', 'rendering']) == 2
    assert main([*base, '--operation', 'index', '--stage', 'smoke']) == 2
    assert main([*base, '--operation', 'rendering', '--stage', 'smoke']) == 0
    payload = json.loads(state.read_text(encoding='utf-8'))
    assert payload['operation'] == 'rendering'
    assert payload['stage'] == 'smoke'
