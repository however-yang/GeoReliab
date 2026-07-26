'''Canonical Task-2 preparation APIs after the round-2 audit.

The functions in this module deliberately keep all locally executable work
small: production inputs are explicit ``.npy`` linear-RGB/depth pairs whose
source and GT byte digests are checked before they can influence evidence.
'''

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import preparation as _base


CALIBRATION_SCENES = (115, 107, 82, 45, 117, 61, 127, 83, 56, 92)


@dataclass(frozen=True)
class PreparedBatch:
    stage: str
    split: str
    split_view_manifest_sha256: str
    materialization_sha256: str
    records: tuple[tuple[int, int, str, np.ndarray, np.ndarray, str, str], ...]


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
    try:
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
    except OSError as exc:
        raise _base.PreparationError(f'provenance file is unreadable: {path}') from exc
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
                           calibration: _base.CorruptionCalibration) -> tuple[np.ndarray, dict[str, Any]]:
    rgb = np.asarray(image, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
        raise _base.CalibrationError('low-light requires finite linear HxWx3 RGB')
    index = _base._severity_index(severity)
    expected = np.clip(rgb, 0.0, 1.0) * _base._EXPOSURES[index]
    seed = _base.seed_for_sample(sample_key, view_id)
    rng = np.random.default_rng(seed)
    rendered = np.clip(rng.poisson(expected * _base._POISSON_PEAKS[index]) / _base._POISSON_PEAKS[index]
                       + rng.normal(0.0, _base._READ_SIGMAS[index], rgb.shape), 0.0, 1.0)
    if not isinstance(calibration, _base.CorruptionCalibration):
        raise _base.CalibrationError('low-light requires the shared CorruptionCalibration')
    parameter = calibration
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
        defocus_scale=calibration.defocus_scales[index],
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
    # Non-informative observed contrast has no physical-direction evidence.
    if np.unique(contrast[valid]).size < 2 or float(np.std(contrast[valid])) == 0.0:
        raise _base.CalibrationError('synthetic fog has non-informative observed contrast')
    return _spearman(depth[valid].ravel(), contrast[valid].ravel())


def calibration_qa(calibration: _base.CorruptionCalibration,
                   records: Sequence[tuple[int, int, np.ndarray, np.ndarray, str, str]], *,
                   expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate frozen calibration checks and fail closed for every failed item."""
    if not records:
        raise _base.CalibrationError('calibration QA requires calibration records')
    metrics: list[dict[str, Any]] = []
    parameter_vectors: dict[tuple[str, int], set[tuple[Any, ...]]] = {}
    for scene_id, view_id, image, depth, raw_digest, gt_digest in records:
        row: dict[str, Any] = {'scene_id': scene_id, 'view_id': view_id, 'fog': [], 'brightness': [], 'noise': [], 'coc': [], 'edge_loss': [], 'fog_correlation': [], 'fog_strength': [], 'gt': []}
        for severity in (1, 2, 3):
            fog, fog_meta = fog_render(image, depth, calibration, severity=severity, gt_digest=gt_digest, raw_source_sha256=raw_digest)
            low, low_meta = low_light_noise_render(image, f'scan{scene_id}', view_id, severity=severity, gt_digest=gt_digest, calibration=calibration, raw_source_sha256=raw_digest)
            _, defocus_meta = render_defocus(image, depth, calibration, severity=severity, gt_digest=gt_digest, raw_source_sha256=raw_digest)
            row['fog'].append(fog_meta['realized_transmittance'])
            row['brightness'].append(float(low.mean()))
            row['noise'].append(low_meta['measured_noise'])
            row['coc'].append(defocus_meta['coc_p95'])
            row['edge_loss'].append(defocus_meta['edge_energy_loss'])
            correlation = _synthetic_fog_correlation(image, depth, calibration, severity)
            row['fog_correlation'].append(correlation)
            # This is the observed depth/contrast association itself; no
            # transmittance-derived severity multiplier is permitted.
            row['fog_strength'].append(abs(correlation))
            row['gt'].extend((fog_meta['gt_digest'], low_meta['gt_digest'], defocus_meta['gt_digest']))
            parameter_vectors.setdefault(('fog', severity), set()).add((
                fog_meta['parameter_manifest_sha256'], fog_meta['beta'],
            ))
            parameter_vectors.setdefault(('low-light-noise', severity), set()).add((
                low_meta['parameter_manifest_sha256'], low_meta['exposure'],
                low_meta['poisson_peak'], low_meta['read_sigma'],
            ))
            parameter_vectors.setdefault(('defocus', severity), set()).add((
                defocus_meta['parameter_manifest_sha256'], defocus_meta['focus_depth'],
                defocus_meta['inverse_depth_layers'], defocus_meta['defocus_scale'],
            ))
        metrics.append(row)
    strict_down = lambda values: values[0] > values[1] > values[2]
    strict_up = lambda values: values[0] < values[1] < values[2]
    checks = {
        'fog': all(strict_down(row['fog']) for row in metrics),
        'low_light': all(strict_down(row['brightness']) and strict_up(row['noise']) for row in metrics),
        'defocus': all(strict_up(row['coc']) and strict_up(row['edge_loss']) for row in metrics),
        'gt': all(len(set(row['gt'])) == 1 for row in metrics),
        'cross_view': all(len(values) == 1 for values in parameter_vectors.values()),
        'synthetic_fog': all(
            all(value < 0.0 for value in row['fog_correlation'])
            and strict_up(row['fog_strength'])
            for row in metrics
        ),
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


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise _base.PreparationError(f'{label} is unreadable: {exc}') from exc
    if not isinstance(payload, dict):
        raise _base.PreparationError(f'{label} must be a JSON object')
    return payload


def _materialized_dtu_map(payload: Mapping[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for scene_row in payload['dtu']:
        scene = int(scene_row['scene_id'])
        for view, rgb in scene_row['rgb'].items():
            key = (scene, int(view))
            if key in result:
                raise _base.PreparationError('materialization contains duplicate DTU scene/view')
            result[key] = {
                'rgb': rgb,
                'camera': scene_row['cameras'][view],
                'points': scene_row['points'],
                'mask': scene_row['mask'],
                'split': scene_row['split'],
            }
    return result


def _verify_asset_reference(
    row: Mapping[str, Any], expected: Mapping[str, Any], name: str
) -> None:
    try:
        supplied = row['source_assets'][name]
        member = supplied['member']
        raw_sha = supplied['raw_sha256']
    except (KeyError, TypeError) as exc:
        raise _base.PreparationError(f'prepared input is missing {name} provenance') from exc
    if member != expected['member'] or raw_sha != expected['raw_sha256']:
        raise _base.PreparationError(
            f'prepared input {name} provenance is a cross-resource swap'
        )


_STAGE_SPLIT = {
    'calibration': 'calibration',
    'smoke': 'dev',
    'test': 'test',
}


def load_prepared_batch(
    path: Path, *, expected_stage: str | None = None,
) -> PreparedBatch:
    '''Load decoded arrays only after proving their official-byte provenance.'''

    payload = _load_json(path, 'prepared input')
    if payload.get('schema_version') != 'prepared-input-v2':
        raise _base.PreparationError('prepared input must use prepared-input-v2')
    stage = payload.get('stage')
    split = payload.get('split')
    if stage not in _STAGE_SPLIT or split != _STAGE_SPLIT[stage]:
        raise _base.PreparationError('prepared input stage/split binding is invalid')
    if expected_stage is not None and stage != expected_stage:
        raise _base.PreparationError(
            f'prepared input stage {stage} does not match {expected_stage}'
        )
    try:
        split_path = Path(str(payload['split_view_manifest_path']))
        materialization_path = Path(str(payload['materialization_path']))
        supplied_split_sha = payload['split_view_manifest_sha256']
        supplied_materialization_sha = payload['materialization_sha256']
    except (KeyError, TypeError) as exc:
        raise _base.PreparationError(
            'prepared input is missing frozen manifest provenance'
        ) from exc
    split_sha = _sha_file(split_path)
    materialization_sha = _sha_file(materialization_path)
    if supplied_split_sha != split_sha or supplied_materialization_sha != materialization_sha:
        raise _base.PreparationError('prepared input manifest digest mismatch')
    split_payload = _load_json(split_path, 'split/view manifest')
    if split_payload.get('schema_version') != 'dtu-preparation-v1':
        raise _base.PreparationError('split/view manifest schema mismatch')
    from .materialization import verify_materialization_manifest
    materialization = verify_materialization_manifest(
        materialization_path, split_manifest_path=split_path,
    )
    materialized = _materialized_dtu_map(materialization)
    try:
        scenes = tuple(int(value) for value in split_payload['splits'][split])
        required = {
            (scene, int(view))
            for scene in scenes
            for view in split_payload['views'][str(scene)]
        }
        rows = payload['records']
    except (KeyError, TypeError, ValueError) as exc:
        raise _base.PreparationError('prepared input schedule is incomplete') from exc
    if not isinstance(rows, list) or not required:
        raise _base.PreparationError('prepared input records must be a non-empty list')

    records: list[tuple[int, int, str, np.ndarray, np.ndarray, str, str]] = []
    seen: set[tuple[int, int]] = set()
    sample_keys: set[str] = set()
    for row in rows:
        try:
            scene = int(row['scene_id'])
            view = int(row['view_id'])
            sample_key = row['sample_key']
            raw_path = Path(str(row['raw_rgb_path']))
            raw_sha = row['raw_source_sha256']
            rgb_path = Path(str(row['linear_rgb_npy']))
            rgb_sha = row['linear_rgb_npy_sha256']
            depth_path = Path(str(row['depth_npy']))
            depth_sha = row['depth_npy_sha256']
            gt_digest = row['gt_digest']
            derivation = row['depth_derivation']
        except (KeyError, TypeError, ValueError) as exc:
            raise _base.PreparationError('prepared input record is incomplete') from exc
        pair = (scene, view)
        if pair in seen or not isinstance(sample_key, str) or not sample_key or sample_key in sample_keys:
            raise _base.PreparationError('prepared input has duplicate scene/view or sample_key')
        if pair not in required:
            raise _base.PreparationError('prepared input contains a record outside its frozen split')
        expected = materialized.get(pair)
        if expected is None or expected['split'] != split:
            raise _base.PreparationError('prepared input is not present in its materialized split')
        for name in ('rgb', 'camera', 'points', 'mask'):
            _verify_asset_reference(row, expected[name], name)
        if raw_path != Path(str(expected['rgb']['path'])) or raw_sha != expected['rgb']['raw_sha256']:
            raise _base.PreparationError('prepared RGB is not bound to the official materialized bytes')
        if not raw_path.is_file() or _sha_file(raw_path) != raw_sha:
            raise _base.PreparationError('prepared raw RGB digest mismatch')
        for file_path, digest, label in (
            (rgb_path, rgb_sha, 'linear RGB'),
            (depth_path, depth_sha, 'depth'),
        ):
            if not file_path.is_file() or _sha_file(file_path) != digest:
                raise _base.PreparationError(f'prepared {label} array digest mismatch')
        if gt_digest != depth_sha:
            raise _base.PreparationError('prepared GT digest is not bound to the depth array')
        expected_inputs = {
            name: expected[name]['raw_sha256']
            for name in ('camera', 'points', 'mask')
        }
        if (
            not isinstance(derivation, Mapping)
            or derivation.get('algorithm') != 'dtu-points-camera-projection-v1'
            or derivation.get('input_sha256') != expected_inputs
            or derivation.get('output_sha256') != depth_sha
        ):
            raise _base.PreparationError('prepared depth derivation provenance mismatch')
        try:
            image = np.load(rgb_path, allow_pickle=False)
            depth = np.load(depth_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise _base.PreparationError('prepared arrays are unreadable') from exc
        image, depth, _ = _base._require_image_depth(image, depth)
        records.append((scene, view, sample_key, image, depth, raw_sha, gt_digest))
        seen.add(pair)
        sample_keys.add(sample_key)
    if seen != required:
        missing = sorted(required - seen)
        extra = sorted(seen - required)
        raise _base.PreparationError(
            f'prepared input does not exactly cover its frozen schedule: missing={missing}, extra={extra}'
        )
    return PreparedBatch(
        stage=stage,
        split=split,
        split_view_manifest_sha256=split_sha,
        materialization_sha256=materialization_sha,
        records=tuple(records),
    )


def load_prepared_records(
    path: Path, *, required_scenes: Sequence[int] | None = None,
    expected_stage: str | None = None,
) -> list[tuple[int, int, str, np.ndarray, np.ndarray, str, str]]:
    '''Compatibility return shape backed exclusively by the v2 contract.'''

    batch = load_prepared_batch(path, expected_stage=expected_stage)
    if required_scenes is not None:
        actual = {record[0] for record in batch.records}
        expected = set(map(int, required_scenes))
        if actual != expected:
            raise _base.PreparationError(
                f'prepared input scenes do not match required scenes: {sorted(actual)}'
            )
    return list(batch.records)


def load_tartanair_prepared_pairs(
    path: Path,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    '''Verify 100 decoded P000 pairs against exact materialized official bytes.'''

    payload = _load_json(path, 'TartanAir prepared input')
    if payload.get('schema_version') != 'tartanair-prepared-v2':
        raise _base.PreparationError('TartanAir prepared input must use tartanair-prepared-v2')
    try:
        materialization_path = Path(str(payload['materialization_path']))
        materialization_sha = payload['materialization_sha256']
        records = payload['records']
    except (KeyError, TypeError) as exc:
        raise _base.PreparationError('TartanAir prepared input is missing provenance') from exc
    if not materialization_path.is_file() or _sha_file(materialization_path) != materialization_sha:
        raise _base.PreparationError('TartanAir materialization digest mismatch')
    materialization = _load_json(materialization_path, 'frozen materialization')
    if materialization.get('schema_version') != 'frozen-materialization-v1':
        raise _base.PreparationError('TartanAir materialization schema mismatch')
    try:
        split_path = Path(str(materialization['split_view_manifest_path']))
    except (KeyError, TypeError) as exc:
        raise _base.PreparationError('TartanAir materialization is not split-bound') from exc
    from .materialization import verify_materialization_manifest
    materialization = verify_materialization_manifest(
        materialization_path, split_manifest_path=split_path,
    )
    official = {
        str(pair['frame_id']): pair
        for pair in materialization.get('tartanair', {}).get('pairs', [])
    }
    if len(official) != 100 or not isinstance(records, list) or len(records) != 100:
        raise _base.PreparationError('TartanAir prepared input requires exactly 100 frozen pairs')
    result: list[tuple[str, np.ndarray, np.ndarray]] = []
    seen: set[str] = set()
    for row in records:
        try:
            frame = str(row['frame_id'])
            raw_rgb_path = Path(str(row['raw_rgb_path']))
            raw_depth_path = Path(str(row['raw_depth_path']))
            rgb_path = Path(str(row['rgb_npy']))
            depth_path = Path(str(row['depth_npy']))
            rgb_sha = row['rgb_npy_sha256']
            depth_sha = row['depth_npy_sha256']
            sources = row['source_assets']
            rgb_decode = row['rgb_decode']
            depth_decode = row['depth_decode']
        except (KeyError, TypeError) as exc:
            raise _base.PreparationError('TartanAir prepared record is incomplete') from exc
        expected = official.get(frame)
        if frame in seen or expected is None:
            raise _base.PreparationError('TartanAir prepared input has duplicate or unknown frame')
        for name, raw_path in (('rgb', raw_rgb_path), ('depth', raw_depth_path)):
            expected_asset = expected[name]
            supplied = sources.get(name) if isinstance(sources, Mapping) else None
            if (
                not isinstance(supplied, Mapping)
                or supplied.get('member') != expected_asset['member']
                or supplied.get('raw_sha256') != expected_asset['raw_sha256']
                or raw_path != Path(str(expected_asset['path']))
                or not raw_path.is_file()
                or _sha_file(raw_path) != expected_asset['raw_sha256']
            ):
                raise _base.PreparationError(
                    f'TartanAir {name} provenance is tampered or cross-frame swapped'
                )
        for decoded_path, decoded_sha, label in (
            (rgb_path, rgb_sha, 'RGB'), (depth_path, depth_sha, 'depth'),
        ):
            if not decoded_path.is_file() or _sha_file(decoded_path) != decoded_sha:
                raise _base.PreparationError(f'TartanAir decoded {label} digest mismatch')
        if (
            not isinstance(rgb_decode, Mapping)
            or rgb_decode.get('algorithm') != 'png-linear-rgb-v1'
            or rgb_decode.get('input_sha256') != expected['rgb']['raw_sha256']
            or rgb_decode.get('output_sha256') != rgb_sha
            or not isinstance(depth_decode, Mapping)
            or depth_decode.get('algorithm') != 'tartanair-depth-png-v1'
            or depth_decode.get('input_sha256') != expected['depth']['raw_sha256']
            or depth_decode.get('output_sha256') != depth_sha
        ):
            raise _base.PreparationError('TartanAir decode provenance mismatch')
        try:
            image = np.load(rgb_path, allow_pickle=False)
            depth = np.load(depth_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise _base.PreparationError('TartanAir decoded arrays are unreadable') from exc
        image, depth, _ = _base._require_image_depth(image, depth)
        result.append((frame, image, depth))
        seen.add(frame)
    if seen != set(official):
        raise _base.PreparationError('TartanAir prepared input does not exactly cover frozen frames')
    return result


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(json.dumps(payload, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    partial.replace(path)
