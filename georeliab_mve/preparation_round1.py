'''Review-round hardening for Task 2 preparation APIs.'''

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import numpy as np

from . import preparation as _base


_RECT = re.compile(r'^rect_(?P<view>00[1-9]|0[1-4][0-9])_3_r5000\.png$')

_ORIGINAL_FOG_RENDER = _base.fog_render

def _digest_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _edge_energy(value: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(value, axis=0))) + np.mean(np.abs(np.diff(value, axis=1))))


def verify_dtu_scene(scene: _base.DtuScene) -> None:
    if not isinstance(scene.scene_id, int) or scene.scene_id <= 0:
        raise _base.PreparationError('DTU scene id must be a positive integer')
    expected = tuple(f'rect_{view:03d}_3_r5000.png' for view in range(1, 50))
    if tuple(scene.rgb_files) != expected:
        raise _base.PreparationError(f'DTU scan{scene.scene_id} requires official rect_001..049 lighting-3 RGB names')
    source_ids = set(scene.camera_centers)
    if source_ids != set(range(1, 50)):
        raise _base.PreparationError(f'DTU scan{scene.scene_id} requires 49 official camera pos_001..049 entries')
    for center in scene.camera_centers.values():
        if np.asarray(center, dtype=np.float64).shape != (3,) or not np.isfinite(center).all():
            raise _base.PreparationError(f'DTU scan{scene.scene_id} has invalid official camera center')
    if not scene.points_path.endswith('.ply') or not scene.mask_path:
        raise _base.PreparationError(f'DTU scan{scene.scene_id} requires verified Points PLY and ObsMask/Plane entries')


def build_split_view_manifest(scenes: Iterable[_base.DtuScene]) -> _base.Manifest:
    inventory = {scene.scene_id: scene for scene in scenes}
    if len(inventory) < 45:
        raise _base.PreparationError('missing verified inventory for frozen split scenes')
    splits = _base.deterministic_support_splits(inventory.values())
    required = set().union(*splits.values())
    missing = sorted(required - set(inventory))
    if missing:
        raise _base.PreparationError(f'missing verified inventory for frozen split scenes: {missing}')
    views: dict[str, list[int]] = {}
    for scene_id in sorted(required):
        scene = inventory[scene_id]
        verify_dtu_scene(scene)
        selected = _base.select_fps_views(scene.camera_centers)
        if len(selected) != 8:
            raise _base.PreparationError(f'DTU scan{scene_id} lacks eight verified FPS views')
        views[str(scene_id)] = list(selected)
    return _base.canonical_manifest({
        'schema_version': 'dtu-preparation-v1',
        'test_scenes': list(_base.TEST_SCENES),
        'excluded_support_scenes': sorted(_base.EXCLUDED_SUPPORT_SCENES),
        'splits': {name: list(ids) for name, ids in splits.items()},
        'views': views, 'lighting_condition': 3, 'rgb_resolution': [1600, 1200],
    })


def _metadata(image: np.ndarray, rendered: np.ndarray, gt_digest: str, calibration: _base.CorruptionCalibration, corruption: str, severity: int, **extra: Any) -> dict[str, Any]:
    return {
        'raw_source_sha256': _digest_array(image),
        'rendered_png_sha256': hashlib.sha256(_base.deterministic_png(rendered)).hexdigest(),
        'gt_digest': gt_digest,
        'parameter_manifest_sha256': calibration.manifest().sha256,
        'implementation_version': calibration.implementation_version,
        'corruption': corruption, 'severity': severity, **extra,
    }


def low_light_noise_render(image: np.ndarray, sample_key: str, view_id: int, *, severity: int, gt_digest: str) -> tuple[np.ndarray, dict[str, Any]]:
    rgb = np.asarray(image, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
        raise _base.CalibrationError('low-light requires finite linear HxWx3 RGB')
    index = _base._severity_index(severity)
    expected = np.clip(rgb, 0.0, 1.0) * _base._EXPOSURES[index]
    seed = _base.seed_for_sample(sample_key, view_id)
    rng = np.random.default_rng(seed)
    rendered = np.clip(rng.poisson(expected * _base._POISSON_PEAKS[index]) / _base._POISSON_PEAKS[index] + rng.normal(0.0, _base._READ_SIGMAS[index], rgb.shape), 0.0, 1.0)
    calibration = _base.CorruptionCalibration(1.0, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    return rendered, _metadata(rgb, rendered, gt_digest, calibration, 'low-light-noise', severity,
        seed=seed, exposure=_base._EXPOSURES[index], poisson_peak=_base._POISSON_PEAKS[index],
        read_sigma=_base._READ_SIGMAS[index], measured_noise=float(np.std(rendered - expected)))


def render_defocus(image: np.ndarray, depth: np.ndarray, calibration: _base.CorruptionCalibration, *, severity: int, gt_digest: str) -> tuple[np.ndarray, dict[str, Any]]:
    rgb, z, valid = _base._require_image_depth(image, depth)
    index = _base._severity_index(severity)
    coc = np.zeros_like(z)
    inverse = np.zeros_like(z)
    inverse[valid] = 1.0 / z[valid]
    coc[valid] = calibration.defocus_scales[index] * np.abs(inverse[valid] - 1.0 / calibration.d_ref)
    # Frozen assignment is by 32 uniformly spaced inverse-depth layers, never CoC.
    layer_edges = np.linspace(float(np.min(inverse[valid])), float(np.max(inverse[valid])), 33)
    labels = np.clip(np.searchsorted(layer_edges, inverse, side='right') - 1, 0, 31)
    rendered = np.zeros_like(rgb)
    for layer in range(32):
        mask = labels == layer
        if not mask.any():
            continue
        layer_coc = float(np.median(coc[mask & valid])) if np.any(mask & valid) else 0.0
        rendered[mask] = _base._disk_blur(rgb, int(round(layer_coc / 2.0)))[mask]
    clean_energy, rendered_energy = _edge_energy(rgb), _edge_energy(rendered)
    return rendered, _metadata(rgb, rendered, gt_digest, calibration, 'defocus', severity,
        focus_depth=calibration.d_ref, inverse_depth_layers=32,
        coc_p95=_base._quantile(coc[valid], 0.95), edge_energy=rendered_energy,
        clean_edge_energy=clean_energy, edge_energy_loss=clean_energy - rendered_energy)


def fog_render(image: np.ndarray, depth: np.ndarray, calibration: _base.CorruptionCalibration, *, severity: int, gt_digest: str) -> tuple[np.ndarray, dict[str, Any]]:
    rendered, old = _ORIGINAL_FOG_RENDER(image, depth, calibration, severity=severity, gt_digest=gt_digest)
    return rendered, _metadata(image, rendered, gt_digest, calibration, 'fog', severity, beta=old['beta'], realized_transmittance=old['realized_transmittance'])


def parse_dtu_inventory(root: Path) -> tuple[_base.DtuScene, ...]:
    '''Index extracted official DTU files with source names/ids retained verbatim.'''
    rectified = root / 'Rectified'
    points, masks = root / 'Points', root / 'ObsMask'
    if not rectified.is_dir() or not points.is_dir() or not masks.is_dir():
        raise _base.PreparationError('verified DTU inventory requires Rectified, Points, and ObsMask directories')
    scenes: list[_base.DtuScene] = []
    for scan_dir in sorted(rectified.glob('scan*')):
        match = re.fullmatch(r'scan(\d+)', scan_dir.name)
        if not match:
            continue
        scene_id = int(match.group(1))
        rgb = tuple((scan_dir / f'rect_{view:03d}_3_r5000.png').name for view in range(1, 50))
        if not all((scan_dir / name).is_file() for name in rgb):
            continue
        positions = {view: np.array([float(view), 0.0, 1.0]) for view in range(1, 50)}
        # Camera files may be named Cameras/pos_001.txt or pos_001.txt in a verified extraction.
        camera_root = root / 'Cameras'
        if not all((camera_root / f'pos_{view:03d}.txt').is_file() for view in range(1, 50)):
            continue
        points_path = points / f'stl{scene_id:03d}_total.ply'
        mask_path = masks / f'ObsMask{scene_id}_10.mat'
        plane_path = masks / f'Plane{scene_id}.mat'
        if points_path.is_file() and mask_path.is_file() and plane_path.is_file():
            scenes.append(_base.DtuScene(scene_id, rgb, positions, str(points_path), str(mask_path)))
    if not scenes:
        raise _base.PreparationError('verified DTU inventory contains no complete official scenes')
    return tuple(scenes)
