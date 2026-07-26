from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from georeliab_mve.cli import main
from georeliab_mve.preparation import (
    A100Overlay,
    CalibrationError,
    DtuScene,
    PreparationError,
    TARTANAIR_NATIVE_FOG_SANITY,
    build_split_view_manifest,
    calibrate_corruptions,
    deterministic_support_splits,
    deterministic_png,
    fog_render,
    low_light_noise_render,
    render_defocus,
    seed_for_sample,
    select_fps_views,
    tartanair_native_fog_sanity,
    verify_dtu_scene,
)


TEST_SCENES = (1, 9, 10, 11, 12, 13, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118)


def _scene(scene_id: int, centers: dict[int, np.ndarray] | None = None) -> DtuScene:
    centers = centers or {view: np.array([float(view), 0.0, 1.0]) for view in range(1, 50)}
    return DtuScene(
        scene_id=scene_id,
        rgb_files=tuple(f"rect_{view:03d}_3_r5000.png" for view in range(1, 50)),
        camera_centers=centers,
        points_path=f"Points/stl{scene_id:03d}_total.ply",
        mask_path=f"ObsMask/ObsMask{scene_id:03d}_10.mat",
    )


def test_frozen_test_split_exclusions_and_deterministic_support_assignment():
    candidates = [_scene(scene_id) for scene_id in range(1, 90) if scene_id not in TEST_SCENES]
    candidates.extend(_scene(scene_id) for scene_id in (100, 101, 102))
    splits = deterministic_support_splits(candidates)
    assert splits['test'] == TEST_SCENES
    assert 4 not in splits['dev'] + splits['calibration'] + splits['reference-token']
    assert 15 not in splits['dev'] + splits['calibration'] + splits['reference-token']
    assert {name: len(splits[name]) for name in ('dev', 'calibration', 'reference-token')} == {
        'dev': 10,
        'calibration': 10,
        'reference-token': 5,
    }
    assert splits == deterministic_support_splits(reversed(candidates))


def test_incomplete_dtu_scene_fails_closed():
    incomplete = _scene(2)
    incomplete = DtuScene(**{**incomplete.__dict__, 'rgb_files': incomplete.rgb_files[:-1]})
    with pytest.raises(PreparationError, match='49'):
        verify_dtu_scene(incomplete)


def test_camera_center_fps_is_translation_scale_invariant_and_breaks_ties():
    centers = {
        3: np.array([0.0, 0.0, 0.0]),
        7: np.array([1.0, 0.0, 0.0]),
        9: np.array([-1.0, 0.0, 0.0]),
        12: np.array([0.0, 1.0, 0.0]),
        15: np.array([0.0, -1.0, 0.0]),
        18: np.array([1.0, 1.0, 0.0]),
        21: np.array([-1.0, -1.0, 0.0]),
        24: np.array([2.0, 0.0, 0.0]),
    }
    selected = select_fps_views(centers, count=4)
    transformed = {key: value * 13.0 + np.array([100.0, -7.0, 2.0]) for key, value in centers.items()}
    assert selected == select_fps_views(transformed, count=4)
    assert selected[0] == 3
    assert selected[1] == 24


def test_manifest_is_byte_identical_and_hashed():
    scenes = [_scene(scene_id) for scene_id in list(range(1, 78)) + list(range(82, 129))]
    first = build_split_view_manifest(scenes)
    second = build_split_view_manifest(reversed(scenes))
    assert first.json_bytes == second.json_bytes
    assert first.sha256 == hashlib.sha256(first.json_bytes).hexdigest()
    assert json.loads(first.json_bytes)['schema_version'] == 'dtu-preparation-v1'


def test_seed_and_corruptions_are_deterministic_and_preserve_gt_digest():
    image = np.full((12, 16, 3), 0.5, dtype=np.float64)
    depth = np.linspace(1.0, 3.0, image.shape[0] * image.shape[1]).reshape(image.shape[:2])
    params = calibrate_corruptions([depth], [image])
    assert seed_for_sample('calibration/scan2', 4) == seed_for_sample('calibration/scan2', 4)
    fog1, fog_meta = fog_render(image, depth, params, severity=2, gt_digest='gt')
    fog2, _ = fog_render(image, depth, params, severity=2, gt_digest='gt')
    low1, low_meta = low_light_noise_render(image, 'calibration/scan2', 4, severity=2, gt_digest='gt')
    low2, _ = low_light_noise_render(image, 'calibration/scan2', 4, severity=2, gt_digest='gt')
    defocus1, defocus_meta = render_defocus(image, depth, params, severity=2, gt_digest='gt')
    defocus2, _ = render_defocus(image, depth, params, severity=2, gt_digest='gt')
    assert np.array_equal(fog1, fog2)
    assert np.array_equal(low1, low2)
    assert np.array_equal(defocus1, defocus2)
    assert {fog_meta['gt_digest'], low_meta['gt_digest'], defocus_meta['gt_digest']} == {'gt'}
    assert deterministic_png(fog1) == deterministic_png(fog2)


def test_calibration_monotonicity_and_cross_view_parameters():
    image = np.full((48, 48, 3), 0.7, dtype=np.float64)
    image[:, 24:, :] = 0.2
    depth = np.linspace(0.5, 4.0, image.shape[0] * image.shape[1]).reshape(image.shape[:2])
    params = calibrate_corruptions([depth, depth * 1.1], [image, image * 0.9])
    fog_trans = []
    brightness = []
    noise = []
    coc = []
    edge = []
    for severity in (1, 2, 3):
        rendered_fog, fog_meta = fog_render(image, depth, params, severity=severity, gt_digest='fixed')
        rendered_low, low_meta = low_light_noise_render(image, 'key', 0, severity=severity, gt_digest='fixed')
        rendered_defocus, defocus_meta = render_defocus(image, depth, params, severity=severity, gt_digest='fixed')
        fog_trans.append(fog_meta['realized_transmittance'])
        brightness.append(rendered_low.mean())
        noise.append(low_meta['measured_noise'])
        coc.append(defocus_meta['coc_p95'])
        edge.append(defocus_meta['edge_energy_loss'])
        assert fog_meta['beta'] == params.fog_betas[severity - 1]
        assert rendered_fog.shape == image.shape == rendered_defocus.shape
    assert fog_trans[0] > fog_trans[1] > fog_trans[2]
    assert brightness[0] > brightness[1] > brightness[2]
    assert noise[0] < noise[1] < noise[2]
    assert coc[0] < coc[1] < coc[2]
    assert all(np.isfinite(edge))


def test_tartanair_sanity_enforces_80_of_100_and_fails_closed_for_misalignment():
    depth = np.tile(np.linspace(1.0, 4.0, 64), (64, 1))
    contrast = 1.0 / depth
    rgb = np.repeat(contrast[..., None], 3, axis=-1)
    passing = tartanair_native_fog_sanity([(rgb, depth)] * 100)
    assert passing.reason_code == TARTANAIR_NATIVE_FOG_SANITY
    assert passing.passed is True
    failing = tartanair_native_fog_sanity([(rgb, depth)] * 79)
    assert failing.passed is False
    with pytest.raises(CalibrationError, match='aligned'):
        tartanair_native_fog_sanity([(rgb[:-1], depth)] * 100)


def test_overlay_rejects_scientific_threshold_overrides(tmp_path):
    valid = tmp_path / 'a100.toml'
    valid.write_text("[runtime]\nroot = '/srv/private/smli/GeoReliab'\n[resources]\ndtu_sampleset_url = 'https://example.test/SampleSet.zip'\n", encoding='utf-8')
    assert A100Overlay.load(valid).runtime_root == '/srv/private/smli/GeoReliab'
    bad = tmp_path / 'bad.toml'
    bad.write_text("[georeliab_gate]\nrho_decline = 0.1\n", encoding='utf-8')
    with pytest.raises(PreparationError, match='scientific threshold'):
        A100Overlay.load(bad)


def test_prepare_cli_dry_run_and_state_transitions(tmp_path):
    root = tmp_path / 'data'
    state = tmp_path / 'state.json'
    assert main(['prepare-georeliab', '--operation', 'index', '--data-root', str(root), '--state', str(state), '--dry-run']) == 0
    payload = json.loads(state.read_text(encoding='utf-8'))
    assert payload['operation'] == 'index'
    assert payload['resources_ready'] is False
    assert payload['scientific_ready'] is False
