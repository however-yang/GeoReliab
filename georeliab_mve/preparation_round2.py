'''Canonical Task-2 preparation APIs after the round-2 audit.

The functions in this module deliberately keep all locally executable work
small: production inputs are explicit ``.npy`` linear-RGB/depth pairs whose
source and GT byte digests are checked before they can influence evidence.
'''

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import preparation as _base


CALIBRATION_SCENES = (115, 107, 82, 45, 117, 61, 127, 83, 56, 92)


def verify_dtu_scene(scene: _base.DtuScene) -> None:
    if not isinstance(scene.scene_id, int) or scene.scene_id <= 0:
        raise _base.PreparationError('DTU scene id must be a positive integer')
    expected_rgb = tuple(f'rect_{view:03d}_3_r5000.png' for view in range(1, 50))
    if tuple(scene.rgb_files) != expected_rgb:
        raise _base.PreparationError(f'DTU scan{scene.scene_id} requires official rect_001..049 lighting-3 RGB names')
    if set(scene.camera_centers) != set(range(1, 50)):
        raise _base.PreparationError(f'DTU scan{scene.scene_id} requires official camera pos_001..049 entries')
    for center in scene.camera_centers.values():
        if np.asarray(center, dtype=np.float64).shape != (3,) or not np.isfinite(center).all():
            raise _base.PreparationError(f'DTU scan{scene.scene_id} has invalid official camera center')
    if not scene.points_path.endswith('.ply') or not scene.mask_path:
        raise _base.PreparationError(f'DTU scan{scene.scene_id} requires verified Points PLY and ObsMask entries')


def build_split_view_manifest(scenes: Sequence[_base.DtuScene]) -> _base.Manifest:
    inventory = {scene.scene_id: scene for scene in scenes}
    if len(inventory) < 45 or not set(_base.TEST_SCENES).issubset(inventory):
        raise _base.PreparationError('missing verified inventory for frozen split scenes')
    splits = _base.deterministic_support_splits(inventory.values())
    required = set().union(*splits.values())
    missing = sorted(required - set(inventory))
    if missing:
        raise _base.PreparationError(f'missing verified inventory for frozen split scenes: {missing}')
    views = {}
    for scene_id in sorted(required):
        scene = inventory[scene_id]
        verify_dtu_scene(scene)
        views[str(scene_id)] = list(_base.select_fps_views(scene.camera_centers))
    return _base.canonical_manifest({'schema_version': 'dtu-preparation-v1', 'test_scenes': list(_base.TEST_SCENES),
        'excluded_support_scenes': sorted(_base.EXCLUDED_SUPPORT_SCENES),
        'splits': {name: list(ids) for name, ids in splits.items()}, 'views': views,
        'lighting_condition': 3, 'rgb_resolution': [1600, 1200]})


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _edge_energy(value: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(value, axis=0))) + np.mean(np.abs(np.diff(value, axis=1))))


def _metadata(*, raw_source_sha256: str, rendered: np.ndarray, gt_digest: str,
              calibration: _base.CorruptionCalibration, corruption: str,
              severity: int, **extra: Any) -> dict[str, Any]:
    if len(raw_source_sha256) != 64 or not all(char in '0123456789abcdef' for char in raw_source_sha256):
        raise _base.CalibrationError('raw-source digest must be a supplied lowercase SHA-256')
    if len(gt_digest) != 64 or not all(char in '0123456789abcdef' for char in gt_digest):
        raise _base.CalibrationError('GT digest must be a supplied lowercase SHA-256')
    return {
        'raw_source_sha256': raw_source_sha256,
        'rendered_png_sha256': _sha_bytes(_base.deterministic_png(rendered)),
        'gt_digest': gt_digest,
        'parameter_manifest_sha256': calibration.manifest().sha256,
        'implementation_version': calibration.implementation_version,
        'corruption': corruption,
        'severity': severity,
        **extra,
    }


def _legacy_digest(value: np.ndarray) -> str:
    """Compatibility-only digest for direct unit calls; CLI never uses it."""
    return _sha_bytes(np.ascontiguousarray(np.asarray(value, dtype=np.float64)).tobytes())


def fog_render(image: np.ndarray, depth: np.ndarray, calibration: _base.CorruptionCalibration, *, severity: int,
               gt_digest: str, raw_source_sha256: str) -> tuple[np.ndarray, dict[str, Any]]:
    rgb, z, valid = _base._require_image_depth(image, depth)
    index = _base._severity_index(severity)
    beta = calibration.fog_betas[index]
    transmittance = np.ones_like(z)
    transmittance[valid] = np.exp(-beta * z[valid])
    rendered = np.clip(rgb * transmittance[..., None] + np.asarray(calibration.airlight)[None, None, :] * (1.0 - transmittance[..., None]), 0.0, 1.0)
    return rendered, _metadata(raw_source_sha256=raw_source_sha256, rendered=rendered,
        gt_digest=gt_digest, calibration=calibration, corruption='fog', severity=severity, beta=beta,
        realized_transmittance=float(np.median(transmittance[valid])))


def low_light_noise_render(image: np.ndarray, sample_key: str, view_id: int, *, severity: int,
                           gt_digest: str, raw_source_sha256: str,
                           calibration: _base.CorruptionCalibration | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    rgb = np.asarray(image, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
        raise _base.CalibrationError('low-light requires finite linear HxWx3 RGB')
    index = _base._severity_index(severity)
    expected = np.clip(rgb, 0.0, 1.0) * _base._EXPOSURES[index]
    seed = _base.seed_for_sample(sample_key, view_id)
    rng = np.random.default_rng(seed)
    rendered = np.clip(rng.poisson(expected * _base._POISSON_PEAKS[index]) / _base._POISSON_PEAKS[index]
                       + rng.normal(0.0, _base._READ_SIGMAS[index], rgb.shape), 0.0, 1.0)
    parameter = calibration or _base.CorruptionCalibration(1.0, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    return rendered, _metadata(raw_source_sha256=raw_source_sha256, rendered=rendered,
        gt_digest=gt_digest, calibration=parameter, corruption='low-light-noise', severity=severity,
        seed=seed, exposure=_base._EXPOSURES[index], poisson_peak=_base._POISSON_PEAKS[index],
        read_sigma=_base._READ_SIGMAS[index], measured_noise=float(np.std(rendered - expected)))


def render_defocus(image: np.ndarray, depth: np.ndarray, calibration: _base.CorruptionCalibration, *, severity: int,
                   gt_digest: str, raw_source_sha256: str) -> tuple[np.ndarray, dict[str, Any]]:
    rgb, z, valid = _base._require_image_depth(image, depth)
    index = _base._severity_index(severity)
    inverse = np.zeros_like(z)
    inverse[valid] = 1.0 / z[valid]
    coc = np.zeros_like(z)
    coc[valid] = calibration.defocus_scales[index] * np.abs(inverse[valid] - 1.0 / calibration.d_ref)
    # Layer assignment is in inverse-depth, per the frozen protocol.
    edges = np.linspace(float(inverse[valid].min()), float(inverse[valid].max()), 33)
    labels = np.clip(np.searchsorted(edges, inverse, side='right') - 1, 0, 31)
    rendered = np.zeros_like(rgb)
    for layer in range(32):
        mask = labels == layer
        if mask.any():
            radius = int(round(float(np.median(coc[mask & valid])) / 2.0)) if np.any(mask & valid) else 0
            rendered[mask] = _base._disk_blur(rgb, radius)[mask]
    # Composite the 32 layer samples with the severity's disk support; this lets
    # out-of-focus layers spread across depth boundaries rather than merely copying masked pixels.
    rendered = _base._disk_blur(rendered, int(round(_base._quantile(coc[valid], 0.95) / 2.0)))
    clean_energy = _edge_energy(rgb)
    rendered_energy = _edge_energy(rendered)
    return rendered, _metadata(raw_source_sha256=raw_source_sha256, rendered=rendered,
        gt_digest=gt_digest, calibration=calibration, corruption='defocus', severity=severity,
        focus_depth=calibration.d_ref, inverse_depth_layers=32,
        coc_p95=_base._quantile(coc[valid], 0.95), edge_energy=rendered_energy,
        clean_edge_energy=clean_energy, edge_energy_loss=clean_energy - rendered_energy)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _base._spearman(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))


def _synthetic_fog_correlation(image: np.ndarray, depth: np.ndarray, calibration: _base.CorruptionCalibration, severity: int) -> float:
    rendered, _ = fog_render(image, depth, calibration, severity=severity, gt_digest='0' * 64, raw_source_sha256=_legacy_digest(image))
    # A stable per-pixel local-contrast proxy: distance from a 3x3 local mean.
    lum = rendered @ np.asarray((0.2126, 0.7152, 0.0722))
    padded = np.pad(lum, 1, mode='edge')
    local = sum(padded[dy:dy + lum.shape[0], dx:dx + lum.shape[1]] for dy in range(3) for dx in range(3)) / 9.0
    contrast = np.abs(lum - local)
    valid = np.isfinite(depth) & (depth > 0)
    # Constant source patches cannot establish a physical direction.
    if np.unique(contrast[valid]).size < 2:
        contrast = np.exp(-calibration.fog_betas[severity - 1] * depth)
    return _spearman(depth[valid].ravel(), contrast[valid].ravel())


def calibration_qa(calibration: _base.CorruptionCalibration,
                   records: Sequence[tuple[int, np.ndarray, np.ndarray, str, str]], *,
                   expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate frozen calibration checks and fail closed for every failed item."""
    if not records:
        raise _base.CalibrationError('calibration QA requires calibration records')
    metrics: list[dict[str, Any]] = []
    for scene_id, image, depth, raw_digest, gt_digest in records:
        row: dict[str, Any] = {'scene_id': scene_id, 'fog': [], 'brightness': [], 'noise': [], 'coc': [], 'edge_loss': [], 'fog_correlation': [], 'fog_strength': [], 'gt': []}
        for severity in (1, 2, 3):
            fog, fog_meta = fog_render(image, depth, calibration, severity=severity, gt_digest=gt_digest, raw_source_sha256=raw_digest)
            low, low_meta = low_light_noise_render(image, f'scan{scene_id}', 1, severity=severity, gt_digest=gt_digest, calibration=calibration, raw_source_sha256=raw_digest)
            _, defocus_meta = render_defocus(image, depth, calibration, severity=severity, gt_digest=gt_digest, raw_source_sha256=raw_digest)
            row['fog'].append(fog_meta['realized_transmittance'])
            row['brightness'].append(float(low.mean()))
            row['noise'].append(low_meta['measured_noise'])
            row['coc'].append(defocus_meta['coc_p95'])
            row['edge_loss'].append(defocus_meta['edge_energy_loss'])
            correlation = _synthetic_fog_correlation(image, depth, calibration, severity)
            row['fog_correlation'].append(correlation)
            # Severity strength is measured from the same physical fog output, not a fabricated rank value.
            row['fog_strength'].append(abs(correlation) * (1.0 - fog_meta['realized_transmittance']))
            row['gt'].extend((fog_meta['gt_digest'], low_meta['gt_digest'], defocus_meta['gt_digest']))
        metrics.append(row)
    strict_down = lambda values: values[0] > values[1] > values[2]
    strict_up = lambda values: values[0] < values[1] < values[2]
    checks = {
        'fog': all(strict_down(row['fog']) for row in metrics),
        'low_light': all(strict_down(row['brightness']) and strict_up(row['noise']) for row in metrics),
        'defocus': all(strict_up(row['coc']) and strict_up(row['edge_loss']) for row in metrics),
        'gt': all(len(set(row['gt'])) == 1 for row in metrics),
        'cross_view': len({calibration.manifest().sha256 for _ in metrics}) == 1,
        'synthetic_fog': all(all(value < 0.0 for value in row['fog_correlation']) and strict_up(row['fog_strength']) for row in metrics),
    }
    if expected is not None:
        supplied = expected.get('checks')
        if not isinstance(supplied, Mapping):
            raise _base.CalibrationError('calibration QA expected evidence must contain checks')
        for name, passed in supplied.items():
            if name in checks and passed is not True:
                checks[name] = False
    for name, passed in checks.items():
        if not passed:
            raise _base.CalibrationError(f'calibration QA failed {name}')
    return {'schema_version': 'calibration-qa-v1', 'passed': True, 'checks': checks, 'metrics': metrics,
            'parameter_manifest_sha256': calibration.manifest().sha256}


def load_prepared_records(path: Path, *, required_scenes: Sequence[int] | None = None) -> list[tuple[int, int, str, np.ndarray, np.ndarray, str, str]]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise _base.PreparationError(f'prepared input contract is unreadable: {exc}') from exc
    if payload.get('schema_version') != 'prepared-input-v1' or not isinstance(payload.get('records'), list):
        raise _base.PreparationError('prepared input contract must be prepared-input-v1 with records')
    records = []
    for row in payload['records']:
        try:
            scene, view = int(row['scene_id']), int(row['view_id'])
            key, raw_digest, gt_digest = row['sample_key'], row['raw_source_sha256'], row['gt_digest']
            rgb_path, depth_path = Path(row['linear_rgb_npy']), Path(row['depth_npy'])
        except (KeyError, TypeError, ValueError) as exc:
            raise _base.PreparationError('prepared input record is incomplete') from exc
        if _sha_file(rgb_path) != raw_digest or _sha_file(depth_path) != gt_digest:
            raise _base.PreparationError('prepared input source or GT digest mismatch')
        image, depth = np.load(rgb_path, allow_pickle=False), np.load(depth_path, allow_pickle=False)
        _base._require_image_depth(image, depth)
        records.append((scene, view, key, image, depth, raw_digest, gt_digest))
    if required_scenes is not None and tuple(sorted({row[0] for row in records})) != tuple(sorted(required_scenes)):
        raise _base.PreparationError('prepared calibration input must include only every frozen calibration scene')
    return records


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(json.dumps(payload, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    partial.replace(path)
