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


def _producer_recipe(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        'writer_version': evidence.get('writer_version'),
        'source_module': evidence.get('source_module'),
        'source_sha256': evidence.get('source_sha256'),
        'algorithms': evidence.get('algorithms'),
    }

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
        read_sigma=_base._READ_SIGMAS[index], pre_noise_brightness=float(expected.mean()),
        measured_noise=float(np.std(rendered - expected)))


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


def _patch_depth_contrast(image: np.ndarray, depth: np.ndarray) -> tuple[list[float], list[float]]:
    rgb, z, valid = _base._require_image_depth(image, depth)
    luminance = rgb @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)
    depth_values: list[float] = []
    contrast_values: list[float] = []
    height, width = z.shape
    for y in range(0, height - 31, 32):
        for x in range(0, width - 31, 32):
            patch_valid = valid[y:y + 32, x:x + 32]
            if not patch_valid.all():
                continue
            patch_luminance = luminance[y:y + 32, x:x + 32]
            depth_values.append(float(np.median(z[y:y + 32, x:x + 32])))
            mean = float(patch_luminance.mean())
            contrast_values.append(float(np.sqrt(np.mean((patch_luminance - mean) ** 2))))
    return depth_values, contrast_values


def _scene_synthetic_fog_metrics(scene_patch_data: Mapping[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    scene_metrics: list[dict[str, Any]] = []
    passed = True
    for scene_id in sorted(scene_patch_data):
        data = scene_patch_data[scene_id]
        errors = list(data['errors'])
        clean_depth = np.asarray(data['clean_depth'], dtype=np.float64)
        clean_contrast = np.asarray(data['clean_contrast'], dtype=np.float64)
        fog_rhos: list[float] = []
        effects: list[float] = []
        patch_counts = {'clean': int(clean_depth.size)}
        clean_rho = float('nan')
        scene_passed = not errors
        try:
            if clean_depth.size < 2 or np.unique(clean_contrast).size < 2 or float(np.std(clean_contrast)) == 0.0:
                raise _base.CalibrationError('synthetic_fog clean patch evidence is non-informative')
            clean_rho = _spearman(clean_depth, clean_contrast)
            if not np.isfinite(clean_rho):
                raise _base.CalibrationError('synthetic_fog clean rho is undefined')
            for severity in (1, 2, 3):
                fog_depth = np.asarray(data['fog_depth'][severity], dtype=np.float64)
                fog_contrast = np.asarray(data['fog_contrast'][severity], dtype=np.float64)
                patch_counts[f's{severity}'] = int(fog_depth.size)
                if fog_depth.size < 2 or np.unique(fog_contrast).size < 2 or float(np.std(fog_contrast)) == 0.0:
                    raise _base.CalibrationError(f'synthetic_fog severity-{severity} patch evidence is non-informative')
                rho = _spearman(fog_depth, fog_contrast)
                if not np.isfinite(rho):
                    raise _base.CalibrationError(f'synthetic_fog severity-{severity} rho is undefined')
                fog_rhos.append(float(rho))
                effects.append(float(rho - clean_rho))
        except _base.CalibrationError as exc:
            errors.append(str(exc))
            scene_passed = False
        if not (len(fog_rhos) == 3 and all(effect < 0.0 for effect in effects)
                and abs(effects[0]) < abs(effects[1]) < abs(effects[2])):
            scene_passed = False
        if not scene_passed:
            passed = False
        scene_metrics.append({
            'scene_id': int(scene_id),
            'view_count': len(data['views']),
            'patch_counts': patch_counts,
            'clean_rho': clean_rho,
            'fog_rhos': fog_rhos,
            'effects': effects,
            'passed': scene_passed,
            'errors': errors,
        })
    return scene_metrics, passed


def calibration_qa(calibration: _base.CorruptionCalibration,
                   records: Sequence[tuple[int, int, np.ndarray, np.ndarray, str, str]], *,
                   expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate frozen calibration checks and fail closed for every failed item."""
    if not records:
        raise _base.CalibrationError('calibration QA requires calibration records')
    metrics: list[dict[str, Any]] = []
    parameter_vectors: dict[tuple[str, int], set[tuple[Any, ...]]] = {}
    scene_patch_data: dict[int, dict[str, Any]] = {}
    for scene_id, view_id, image, depth, raw_digest, gt_digest in records:
        scene_data = scene_patch_data.setdefault(scene_id, {
            'views': set(),
            'clean_depth': [],
            'clean_contrast': [],
            'fog_depth': {1: [], 2: [], 3: []},
            'fog_contrast': {1: [], 2: [], 3: []},
            'errors': [],
        })
        scene_data['views'].add(view_id)
        try:
            clean_depth, clean_contrast = _patch_depth_contrast(image, depth)
            scene_data['clean_depth'].extend(clean_depth)
            scene_data['clean_contrast'].extend(clean_contrast)
        except _base.CalibrationError as exc:
            scene_data['errors'].append(f'view {view_id} clean: {exc}')
        row: dict[str, Any] = {'scene_id': scene_id, 'view_id': view_id, 'fog': [], 'brightness': [], 'noise': [], 'coc': [], 'edge_loss': [], 'gt': []}
        for severity in (1, 2, 3):
            fog, fog_meta = fog_render(image, depth, calibration, severity=severity, gt_digest=gt_digest, raw_source_sha256=raw_digest)
            _, low_meta = low_light_noise_render(image, f'scan{scene_id}', view_id, severity=severity, gt_digest=gt_digest, calibration=calibration, raw_source_sha256=raw_digest)
            _, defocus_meta = render_defocus(image, depth, calibration, severity=severity, gt_digest=gt_digest, raw_source_sha256=raw_digest)
            row['fog'].append(fog_meta['realized_transmittance'])
            row['brightness'].append(low_meta['pre_noise_brightness'])
            row['noise'].append(low_meta['measured_noise'])
            row['coc'].append(defocus_meta['coc_p95'])
            row['edge_loss'].append(defocus_meta['edge_energy_loss'])
            try:
                fog_depth, fog_contrast = _patch_depth_contrast(fog, depth)
                scene_data['fog_depth'][severity].extend(fog_depth)
                scene_data['fog_contrast'][severity].extend(fog_contrast)
            except _base.CalibrationError as exc:
                scene_data['errors'].append(f'view {view_id} severity {severity}: {exc}')
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
    synthetic_fog_metrics, synthetic_fog_passed = _scene_synthetic_fog_metrics(scene_patch_data)
    strict_down = lambda values: values[0] > values[1] > values[2]
    strict_up = lambda values: values[0] < values[1] < values[2]
    checks = {
        'fog': all(strict_down(row['fog']) for row in metrics),
        'low_light': all(strict_down(row['brightness']) and strict_up(row['noise']) for row in metrics),
        'defocus': all(strict_up(row['coc']) and strict_up(row['edge_loss']) for row in metrics),
        'gt': all(len(set(row['gt'])) == 1 for row in metrics),
        'cross_view': all(len(values) == 1 for values in parameter_vectors.values()),
        'synthetic_fog': synthetic_fog_passed,
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
            'synthetic_fog_metrics': synthetic_fog_metrics,
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


def _require_prepared_output(path: Path, manifest_path: Path, label: str) -> None:
    """Decoded outputs must live below the manifest's prepared directory."""
    try:
        path.resolve().relative_to(manifest_path.parent.resolve())
    except (OSError, ValueError) as exc:
        raise _base.PreparationError(f'{label} output is out-of-band: {path}') from exc


def _verify_camera_view(evidence: Mapping[str, Any], view_id: int) -> None:
    expected = f'MVS Data/Calibration/cal18/pos_{view_id:03d}.txt'
    member = str(evidence.get('member', '')).replace('\\', '/')
    if not member.endswith(expected):
        raise _base.PreparationError(
            f'prepared camera for view {view_id} is not exact pos_{view_id:03d}.txt'
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
    from .prepared_inputs import (
        DTU_IMAGE_SIZE, DtuPlyCache, decode_dtu_assets,
        implementation_evidence, npy_bytes,
    )
    producer = payload.get('producer')
    if (
        not isinstance(producer, Mapping)
        or not isinstance(producer.get('dependencies'), Mapping)
        or not producer['dependencies']
        or _producer_recipe(producer) != _producer_recipe(implementation_evidence())
    ):
        raise _base.PreparationError('prepared input producer/dependency recipe mismatch')
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
    if (not isinstance(rows, list) or not required
            or payload.get('record_count') != len(required)):
        raise _base.PreparationError('prepared input records must be a non-empty list')

    records: list[tuple[int, int, str, np.ndarray, np.ndarray, str, str]] = []
    seen: set[tuple[int, int]] = set()
    sample_keys: set[str] = set()
    ply_cache = DtuPlyCache()
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
            rgb_decode = row['rgb_decode']
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
        _verify_camera_view(expected['camera'], view)
        if raw_path != Path(str(expected['rgb']['path'])) or raw_sha != expected['rgb']['raw_sha256']:
            raise _base.PreparationError('prepared RGB is not bound to the official materialized bytes')
        if not raw_path.is_file() or _sha_file(raw_path) != raw_sha:
            raise _base.PreparationError('prepared raw RGB digest mismatch')
        _require_prepared_output(rgb_path, path, 'linear RGB')
        _require_prepared_output(depth_path, path, 'depth')
        image, depth, expected_rgb_decode, expected_derivation = decode_dtu_assets(
            expected, view_id=view, ply_cache=ply_cache, image_size=DTU_IMAGE_SIZE,
        )
        expected_rgb_sha = _sha_bytes(npy_bytes(image))
        expected_depth_sha = _sha_bytes(npy_bytes(depth))
        expected_rgb_decode['output_sha256'] = expected_rgb_sha
        expected_derivation['output_sha256'] = expected_depth_sha
        if (not rgb_path.is_file() or _sha_file(rgb_path) != expected_rgb_sha
                or rgb_sha != expected_rgb_sha):
            raise _base.PreparationError(
                'prepared linear RGB does not match deterministic decode of official bytes'
            )
        if (not depth_path.is_file() or _sha_file(depth_path) != expected_depth_sha
                or depth_sha != expected_depth_sha or gt_digest != expected_depth_sha):
            raise _base.PreparationError(
                'prepared depth does not match deterministic projection of official bytes'
            )
        if rgb_decode != expected_rgb_decode:
            raise _base.PreparationError('prepared RGB decode recipe/provenance mismatch')
        if derivation != expected_derivation:
            raise _base.PreparationError('prepared depth derivation recipe/provenance mismatch')
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
    from .prepared_inputs import (
        TARTAN_IMAGE_SIZE, decode_tartanair_assets,
        implementation_evidence, npy_bytes,
    )
    producer = payload.get('producer')
    if (
        not isinstance(producer, Mapping)
        or not isinstance(producer.get('dependencies'), Mapping)
        or not producer['dependencies']
        or _producer_recipe(producer) != _producer_recipe(implementation_evidence())
    ):
        raise _base.PreparationError('TartanAir producer/dependency recipe mismatch')
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
    if (len(official) != 100 or not isinstance(records, list)
            or len(records) != 100 or payload.get('record_count') != 100):
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
        _require_prepared_output(rgb_path, path, 'TartanAir RGB')
        _require_prepared_output(depth_path, path, 'TartanAir depth')
        image, depth, expected_rgb_decode, expected_depth_decode = decode_tartanair_assets(
            expected, image_size=TARTAN_IMAGE_SIZE,
        )
        expected_rgb_sha = _sha_bytes(npy_bytes(image))
        expected_depth_sha = _sha_bytes(npy_bytes(depth))
        expected_rgb_decode['output_sha256'] = expected_rgb_sha
        expected_depth_decode['output_sha256'] = expected_depth_sha
        if (not rgb_path.is_file() or _sha_file(rgb_path) != expected_rgb_sha
                or rgb_sha != expected_rgb_sha):
            raise _base.PreparationError(
                'TartanAir RGB does not match deterministic decode of official bytes'
            )
        if (not depth_path.is_file() or _sha_file(depth_path) != expected_depth_sha
                or depth_sha != expected_depth_sha):
            raise _base.PreparationError(
                'TartanAir depth does not match official BGRA little-endian decode'
            )
        if rgb_decode != expected_rgb_decode or depth_decode != expected_depth_decode:
            raise _base.PreparationError('TartanAir decode recipe/provenance mismatch')
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
