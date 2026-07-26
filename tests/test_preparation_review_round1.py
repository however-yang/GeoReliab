from __future__ import annotations

import hashlib

import numpy as np
import pytest

from georeliab_mve.prepare_cli import run_prepare_operation
from georeliab_mve.preparation import (
    A100Overlay,
    CalibrationError,
    DtuScene,
    PreparationError,
    build_split_view_manifest,
    calibrate_corruptions,
    low_light_noise_render,
    render_defocus,
    select_fps_views,
)


def _scene(scene_id: int) -> DtuScene:
    return DtuScene(
        scene_id=scene_id,
        rgb_files=tuple(f'rect_{view:03d}_3_r5000.png' for view in range(1, 50)),
        camera_centers={view: np.array([float(view), 0.0, 1.0]) for view in range(1, 50)},
        points_path=f'Points/stl{scene_id:03d}_total.ply',
        mask_path=f'ObsMask/ObsMask{scene_id:03d}_10.mat',
    )


def test_manifest_refuses_missing_any_frozen_scene_inventory():
    with pytest.raises(PreparationError, match='missing verified inventory'):
        build_split_view_manifest([_scene(2 + index) for index in range(25)])


def test_low_light_metadata_contains_raw_and_png_digests():
    image = np.full((8, 8, 3), 0.5)
    _, metadata = low_light_noise_render(image, 's', 1, severity=1, gt_digest='g')
    assert metadata['raw_source_sha256'] == hashlib.sha256(image.tobytes()).hexdigest()
    assert len(metadata['rendered_png_sha256']) == 64
    assert len(metadata['parameter_manifest_sha256']) == 64


def test_defocus_reports_measured_not_manufactured_edge_energy():
    image = np.zeros((32, 64, 3))
    image[:, 55:] = 1.0
    depth = np.tile(np.linspace(0.5, 4.0, 64), (32, 1))
    calibration = calibrate_corruptions([depth], [image])
    rendered, metadata = render_defocus(image, depth, calibration, severity=3, gt_digest='g')
    measured = float(np.mean(np.abs(np.diff(rendered, axis=0))) + np.mean(np.abs(np.diff(rendered, axis=1))))
    assert metadata['edge_energy'] == pytest.approx(measured)
    assert metadata['edge_energy_loss'] > 0


def test_overlay_rejects_nested_threshold_override(tmp_path):
    path = tmp_path / 'nested.toml'
    path.write_text("[execution.tuning.georeliab_gate]\nrho_decline = 0.1\n", encoding='utf-8')
    with pytest.raises(PreparationError, match='scientific threshold'):
        A100Overlay.load(path)


def test_prepare_non_dry_index_fails_closed_without_verified_inventory(tmp_path):
    with pytest.raises(PreparationError, match='inventory'):
        run_prepare_operation(
            operation='index', data_root=tmp_path / 'missing',
            state_path=tmp_path / 'state.json', dry_run=False, overlay_path=None,
        )

def test_official_inventory_hash_order_matches_frozen_support_splits():
    from georeliab_mve.preparation import deterministic_support_splits
    scenes = [_scene(scene_id) for scene_id in list(range(1, 78)) + list(range(82, 129))]
    splits = deterministic_support_splits(scenes)
    assert splits['dev'] == (39, 125, 71, 55, 2, 43, 54, 94, 84, 122)
    assert splits['calibration'] == (115, 107, 82, 45, 117, 61, 127, 83, 56, 92)
    assert splits['reference-token'] == (66, 87, 20, 104, 7)


def test_official_camera_source_ids_are_preserved_in_fps():
    centers = {view: np.array([float(view), 0.0, 1.0]) for view in range(1, 50)}
