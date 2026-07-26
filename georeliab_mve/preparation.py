'''Deterministic DTU/TartanAir preparation and shared image corruptions.

This module implements Task 2 of ``docs/plans/2026-07-26-georeliab-real-mve.md``.
It deliberately keeps network and TartanAir imports optional so local protocol
tests cannot accidentally download data or require the remote runtime.
'''

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import struct
import tomllib
from typing import Any
import urllib.request
import zipfile
import zlib

import numpy as np


TEST_SCENES = (
    1, 9, 10, 11, 12, 13, 23, 24, 29, 32, 33, 34, 48, 49,
    62, 75, 77, 110, 114, 118,
)
EXCLUDED_SUPPORT_SCENES = frozenset((4, 15))
CORRUPTION_IMPLEMENTATION_VERSION = 'georeliab-corruptions-v1'
TARTANAIR_NATIVE_FOG_SANITY = 'TARTANAIR_NATIVE_FOG_SANITY'
_SEVERITIES = (1, 2, 3)
_FOG_TARGETS = (0.80, 0.50, 0.25)
_EXPOSURES = (0.5, 0.25, 0.125)
_POISSON_PEAKS = (2048, 512, 128)
_READ_SIGMAS = (0.002, 0.005, 0.010)
_COC_TARGETS = (4.0, 10.0, 20.0)


class PreparationError(ValueError):
    '''Raised for missing or invalid frozen preparation inputs.'''


class CalibrationError(PreparationError):
    '''Raised when calibration or physical-direction sanity fails.'''


@dataclass(frozen=True)
class DtuScene:
    '''The minimum verified DTU scene inventory needed by the protocol.'''

    scene_id: int
    rgb_files: tuple[str, ...]
    camera_centers: Mapping[int, np.ndarray]
    points_path: str
    mask_path: str


@dataclass(frozen=True)
class Manifest:
    json_bytes: bytes
    sha256: str

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.json_bytes)


@dataclass(frozen=True)
class CorruptionCalibration:
    d_ref: float
    airlight: tuple[float, float, float]
    fog_betas: tuple[float, float, float]
    defocus_scales: tuple[float, float, float]
    implementation_version: str = CORRUPTION_IMPLEMENTATION_VERSION

    def manifest(self) -> Manifest:
        return canonical_manifest(
            {
                'schema_version': 'corruption-calibration-v1',
                'implementation_version': self.implementation_version,
                'd_ref': self.d_ref,
                'airlight': list(self.airlight),
                'fog_betas': list(self.fog_betas),
                'defocus_scales': list(self.defocus_scales),
                'exposure': list(_EXPOSURES),
                'poisson_peaks': list(_POISSON_PEAKS),
                'read_sigma': list(_READ_SIGMAS),
                'inverse_depth_layers': 32,
                'coc_p95_targets': list(_COC_TARGETS),
            }
        )


@dataclass(frozen=True)
class TartanAirSanityResult:
    passed: bool
    negative_frames: int
    evaluated_frames: int
    correlations: tuple[float, ...]
    reason_code: str = TARTANAIR_NATIVE_FOG_SANITY


@dataclass(frozen=True)
class A100Overlay:
    runtime_root: str
    resources: Mapping[str, Any]
    execution: Mapping[str, Any]
    source: Path

    @classmethod
    def load(cls, path: Path) -> 'A100Overlay':
        try:
            payload = tomllib.loads(path.read_text(encoding='utf-8'))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise PreparationError(f'cannot load A100 overlay: {exc}') from exc
        forbidden = {
            'geometry_gate', 'georeliab_gate', 'budgets', 'splits', 'geometry',
            'georeliab', 'bootstrap_resamples', 'multiple_testing',
        }
        overridden = sorted(forbidden & set(payload))
        if overridden:
            raise PreparationError(
                'A100 overlay must not override scientific threshold config: '
                + ', '.join(overridden)
            )
        runtime = payload.get('runtime')
        resources = payload.get('resources', {})
        execution = payload.get('execution', {})
        if not isinstance(runtime, dict) or not isinstance(runtime.get('root'), str):
            raise PreparationError('A100 overlay requires [runtime].root')
        root = runtime['root']
        if not root.startswith('/srv/private/') or root.startswith('/home/'):
            raise PreparationError('A100 runtime root must be below /srv/private and never /home')
        if not isinstance(resources, dict) or not isinstance(execution, dict):
            raise PreparationError('A100 overlay resources/execution must be tables')
        return cls(root, resources, execution, path)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_manifest(payload: Mapping[str, Any]) -> Manifest:
    raw = json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
    ).encode('utf-8')
    return Manifest(json_bytes=raw + b'\n', sha256=_sha256(raw + b'\n'))


def _scene_hash(scene_id: int) -> str:
    return _sha256(f'GeoReliab-DTU-v1:scan{scene_id}'.encode('utf-8'))


def verify_dtu_scene(scene: DtuScene) -> None:
    '''Fail closed unless a support scene has all frozen official inputs.'''

    if not isinstance(scene.scene_id, int) or scene.scene_id <= 0:
        raise PreparationError('DTU scene id must be a positive integer')
    if len(scene.rgb_files) != 49:
        raise PreparationError(f'DTU scan{scene.scene_id} requires exactly 49 RGB views')
    if len(set(scene.rgb_files)) != 49 or any(not value for value in scene.rgb_files):
        raise PreparationError(f'DTU scan{scene.scene_id} has invalid RGB inventory')
    if len(scene.camera_centers) != 49 or set(scene.camera_centers) != set(range(49)):
        raise PreparationError(f'DTU scan{scene.scene_id} requires cameras 0..48')
    for view_id, center in scene.camera_centers.items():
        array = np.asarray(center, dtype=np.float64)
        if array.shape != (3,) or not np.isfinite(array).all():
            raise PreparationError(f'DTU scan{scene.scene_id} view {view_id} has invalid camera center')
    if not scene.points_path or not scene.mask_path:
        raise PreparationError(f'DTU scan{scene.scene_id} requires reference points and observability mask')


def deterministic_support_splits(scenes: Iterable[DtuScene]) -> dict[str, tuple[int, ...]]:
    '''Assign verified non-test scenes using the approved hash ordering.'''

    inventory: dict[int, DtuScene] = {}
    for scene in scenes:
        if scene.scene_id in inventory:
            raise PreparationError(f'duplicate DTU scan{scene.scene_id} inventory entry')
        inventory[scene.scene_id] = scene
    eligible: list[int] = []
    for scene_id, scene in inventory.items():
        if scene_id in TEST_SCENES or scene_id in EXCLUDED_SUPPORT_SCENES:
            continue
        verify_dtu_scene(scene)
        eligible.append(scene_id)
    ordered = sorted(eligible, key=lambda scene_id: (_scene_hash(scene_id), scene_id))
    if len(ordered) < 25:
        raise PreparationError('need at least 25 complete eligible DTU support scenes')
    return {
        'dev': tuple(ordered[:10]),
        'calibration': tuple(ordered[10:20]),
        'reference-token': tuple(ordered[20:25]),
        'test': TEST_SCENES,
    }


def select_fps_views(camera_centers: Mapping[int, np.ndarray], *, count: int = 8) -> tuple[int, ...]:
    '''Camera-center FPS normalized by mean/RMS radius with stable id ties.'''

    if count <= 0:
        raise PreparationError('FPS count must be positive')
    ids = sorted(camera_centers)
    if len(ids) < count:
        raise PreparationError(f'FPS requires at least {count} camera centers')
    values = np.asarray([camera_centers[view_id] for view_id in ids], dtype=np.float64)
    if values.shape != (len(ids), 3) or not np.isfinite(values).all():
        raise PreparationError('camera centers must be finite xyz triples')
    normalized = values - values.mean(axis=0, keepdims=True)
    rms = float(np.sqrt(np.mean(np.sum(normalized * normalized, axis=1))))
    if not math.isfinite(rms) or rms <= 0:
        raise PreparationError('camera centers require non-zero RMS radius')
    normalized /= rms
    selected = [0]
    while len(selected) < count:
        candidates = [index for index in range(len(ids)) if index not in selected]
        nearest = np.min(
            np.sum(
                (normalized[candidates, None, :] - normalized[np.asarray(selected), :][None, :, :]) ** 2,
                axis=2,
            ),
            axis=1,
        )
        farthest = float(np.max(nearest))
        options = [candidate for candidate, distance in zip(candidates, nearest) if np.isclose(distance, farthest, rtol=0.0, atol=1e-12)]
        selected.append(min(options, key=lambda index: ids[index]))
    return tuple(ids[index] for index in selected)


def build_split_view_manifest(scenes: Iterable[DtuScene]) -> Manifest:
    inventory = {scene.scene_id: scene for scene in scenes}
    splits = deterministic_support_splits(inventory.values())
    views: dict[str, list[int]] = {}
    for scene_id in sorted(set().union(*splits.values())):
        scene = inventory.get(scene_id)
        if scene is None:
            # Test inventory may be intentionally deferred until the full archive
            # is extracted; no inferred test view list is permitted.
            continue
        verify_dtu_scene(scene)
        views[str(scene_id)] = list(select_fps_views(scene.camera_centers))
    return canonical_manifest(
        {
            'schema_version': 'dtu-preparation-v1',
            'test_scenes': list(TEST_SCENES),
            'excluded_support_scenes': sorted(EXCLUDED_SUPPORT_SCENES),
            'splits': {name: list(ids) for name, ids in splits.items()},
            'views': views,
            'lighting_condition': 3,
            'rgb_resolution': [1600, 1200],
        }
    )


def seed_for_sample(sample_key: str, view_id: int) -> int:
    if not isinstance(sample_key, str) or not sample_key or not isinstance(view_id, int):
        raise PreparationError('sample key and integer view id are required for deterministic seed')
    digest = hashlib.sha256(f'{sample_key}{view_id}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], byteorder='big', signed=False)


def _valid_depth(depth: np.ndarray) -> np.ndarray:
    result = np.asarray(depth, dtype=np.float64)
    return np.isfinite(result) & (result > 0)


def _require_image_depth(image: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.asarray(image, dtype=np.float64)
    z = np.asarray(depth, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or z.shape != rgb.shape[:2]:
        raise CalibrationError('RGB and depth must be aligned HxWx3/HxW arrays')
    if not np.isfinite(rgb).all():
        raise CalibrationError('linear RGB must be finite')
    valid = _valid_depth(z)
    if not valid.any():
        raise CalibrationError('depth has no valid positive pixels')
    return np.clip(rgb, 0.0, 1.0), z, valid


def _brightest_colors(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float64)
    luminance = rgb @ np.asarray((0.2126, 0.7152, 0.0722))
    count = max(1, int(math.ceil(luminance.size * 0.001)))
    selected = np.argpartition(luminance.ravel(), -count)[-count:]
    return rgb.reshape(-1, 3)[selected]


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method='linear'))


def calibrate_corruptions(depths: Sequence[np.ndarray], images: Sequence[np.ndarray]) -> CorruptionCalibration:
    if not depths or len(depths) != len(images):
        raise CalibrationError('calibration needs aligned non-empty depth/image collections')
    valid_depths: list[np.ndarray] = []
    airlights: list[np.ndarray] = []
    inverse_offsets: list[np.ndarray] = []
    for image, depth in zip(images, depths):
        rgb, z, valid = _require_image_depth(image, depth)
        valid_depths.append(z[valid])
        airlights.append(np.median(_brightest_colors(rgb), axis=0))
    d_ref = float(np.median(np.concatenate(valid_depths)))
    if d_ref <= 0 or not math.isfinite(d_ref):
        raise CalibrationError('calibration d_ref must be finite and positive')
    for depth in depths:
        z = np.asarray(depth, dtype=np.float64)
        valid = _valid_depth(z)
        inverse_offsets.append(np.abs(1.0 / z[valid] - 1.0 / d_ref))
    base_p95 = _quantile(np.concatenate(inverse_offsets), 0.95)
    if base_p95 <= 0:
        # Flat-depth calibration remains renderable; its severity has no optical
        # spread and QA will reject it rather than invent test-dependent values.
        base_p95 = 1.0
    return CorruptionCalibration(
        d_ref=d_ref,
        airlight=tuple(float(value) for value in np.median(np.asarray(airlights), axis=0)),
        fog_betas=tuple(-math.log(target) / d_ref for target in _FOG_TARGETS),
        defocus_scales=tuple(target / base_p95 for target in _COC_TARGETS),
    )


def fog_render(
    image: np.ndarray,
    depth: np.ndarray,
    calibration: CorruptionCalibration,
    *,
    severity: int,
    gt_digest: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    rgb, z, valid = _require_image_depth(image, depth)
    index = _severity_index(severity)
    beta = calibration.fog_betas[index]
    transmittance = np.ones_like(z)
    transmittance[valid] = np.exp(-beta * z[valid])
    airlight = np.asarray(calibration.airlight, dtype=np.float64)
    rendered = rgb * transmittance[..., None] + airlight[None, None, :] * (1.0 - transmittance[..., None])
    return np.clip(rendered, 0.0, 1.0), _render_metadata(
        gt_digest, calibration, 'fog', severity,
        beta=beta,
        realized_transmittance=float(np.median(transmittance[valid])),
    )


def low_light_noise_render(
    image: np.ndarray,
    sample_key: str,
    view_id: int,
    *,
    severity: int,
    gt_digest: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    rgb = np.asarray(image, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
        raise CalibrationError('low-light requires finite linear HxWx3 RGB')
    index = _severity_index(severity)
    exposure = _EXPOSURES[index]
    peak = _POISSON_PEAKS[index]
    sigma = _READ_SIGMAS[index]
    generator = np.random.default_rng(seed_for_sample(sample_key, view_id))
    expected = np.clip(rgb, 0.0, 1.0) * exposure
    poisson = generator.poisson(expected * peak) / peak
    rendered = np.clip(poisson + generator.normal(0.0, sigma, size=rgb.shape), 0.0, 1.0)
    metadata = {
        'gt_digest': gt_digest,
        'corruption': 'low-light-noise',
        'severity': severity,
        'seed': seed_for_sample(sample_key, view_id),
        'exposure': exposure,
        'poisson_peak': peak,
        'read_sigma': sigma,
        'measured_noise': float(np.std(rendered - expected)),
        'implementation_version': CORRUPTION_IMPLEMENTATION_VERSION,
    }
    return rendered, metadata


def _disk_blur(image: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return image.copy()
    coordinates = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing='ij')
    kernel = ((xx * xx + yy * yy) <= radius * radius).astype(np.float64)
    kernel /= kernel.sum()
    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode='edge')
    output = np.zeros_like(image, dtype=np.float64)
    for dy in range(kernel.shape[0]):
        for dx in range(kernel.shape[1]):
            if kernel[dy, dx]:
                output += kernel[dy, dx] * padded[dy:dy + image.shape[0], dx:dx + image.shape[1], :]
    return output


def render_defocus(
    image: np.ndarray,
    depth: np.ndarray,
    calibration: CorruptionCalibration,
    *,
    severity: int,
    gt_digest: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    rgb, z, valid = _require_image_depth(image, depth)
    index = _severity_index(severity)
    coc = np.zeros_like(z)
    coc[valid] = calibration.defocus_scales[index] * np.abs(1.0 / z[valid] - 1.0 / calibration.d_ref)
    # Quantize CoC to the frozen 32 inverse-depth layers and composite disk PSF
    # outputs. The construction is deterministic and keeps focus depth fixed.
    layers = np.linspace(0.0, float(np.max(coc)), 32)
    rendered = np.zeros_like(rgb)
    labels = np.minimum(np.searchsorted(layers, coc, side='right') - 1, 31)
    for layer in range(32):
        mask = labels == layer
        if not mask.any():
            continue
        radius = int(round(float(layers[layer]) / 2.0))
        blurred = _disk_blur(rgb, radius)
        rendered[mask] = blurred[mask]
    coc_p95 = _quantile(coc[valid], 0.95)
    raw_edge_energy = float(np.mean(np.abs(np.diff(rendered, axis=0))) + np.mean(np.abs(np.diff(rendered, axis=1))))
    edge_energy = raw_edge_energy / (1.0 + coc_p95)
    return np.clip(rendered, 0.0, 1.0), _render_metadata(
        gt_digest, calibration, 'defocus', severity,
        focus_depth=calibration.d_ref,
        inverse_depth_layers=32,
        coc_p95=_quantile(coc[valid], 0.95),
        edge_energy=edge_energy,
    )


def _severity_index(severity: int) -> int:
    if severity not in _SEVERITIES:
        raise CalibrationError('severity must be one of 1, 2, 3')
    return severity - 1


def _render_metadata(
    gt_digest: str,
    calibration: CorruptionCalibration,
    corruption: str,
    severity: int,
    **values: Any,
) -> dict[str, Any]:
    return {
        'gt_digest': gt_digest,
        'corruption': corruption,
        'severity': severity,
        'parameter_manifest_sha256': calibration.manifest().sha256,
        'implementation_version': calibration.implementation_version,
        **values,
    }


def deterministic_png(image: np.ndarray) -> bytes:
    '''Encode clipped linear RGB as deterministic 8-bit PNG with no metadata.'''

    rgb = np.asarray(image, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise PreparationError('PNG requires HxWx3 RGB')
    # Standard sRGB transfer, performed only after all linear-domain operations.
    encoded = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(np.maximum(rgb, 0), 1 / 2.4) - 0.055)
    pixels = np.rint(np.clip(encoded, 0.0, 1.0) * 255.0).astype(np.uint8)
    height, width, _ = pixels.shape
    rows = b''.join(b'\x00' + pixels[row].tobytes() for row in range(height))
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(rows, level=9)) + chunk(b'IEND', b'')


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0
        index = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        raise CalibrationError('sanity frame has fewer than two valid 32x32 patches')
    a, b = _rank(left), _rank(right)
    denominator = float(np.sqrt(np.sum((a - a.mean()) ** 2) * np.sum((b - b.mean()) ** 2)))
    if denominator == 0:
        raise CalibrationError('sanity frame has undefined contrast/depth Spearman')
    return float(np.sum((a - a.mean()) * (b - b.mean())) / denominator)


def tartanair_native_fog_sanity(frames: Sequence[tuple[np.ndarray, np.ndarray]]) -> TartanAirSanityResult:
    '''Apply the frozen 100-frame physical direction sanity check.'''

    correlations: list[float] = []
    for image, depth in frames[:100]:
        rgb, z, valid = _require_image_depth(image, depth)
        depth_values: list[float] = []
        contrast_values: list[float] = []
        luminance = rgb @ np.asarray((0.2126, 0.7152, 0.0722))
        for y in range(0, z.shape[0] - 31, 32):
            for x in range(0, z.shape[1] - 31, 32):
                patch_valid = valid[y:y + 32, x:x + 32]
                if not patch_valid.all():
                    continue
                patch_luminance = luminance[y:y + 32, x:x + 32]
                mean = float(patch_luminance.mean())
                if mean <= 0:
                    continue
                depth_values.append(float(np.median(z[y:y + 32, x:x + 32])))
                contrast_values.append(float(np.sqrt(np.mean((patch_luminance - mean) ** 2))))
        correlations.append(_spearman(np.asarray(depth_values), np.asarray(contrast_values)))
    negative = sum(value < 0.0 for value in correlations)
    return TartanAirSanityResult(
        passed=len(correlations) == 100 and negative >= 80,
        negative_frames=negative,
        evaluated_frames=len(correlations),
        correlations=tuple(correlations),
    )


def verify_archive(path: Path, *, required_entries: Sequence[str] = ()) -> dict[str, Any]:
    '''Verify a complete official zip archive; partial files are never valid.'''

    if path.suffix == '.partial' or not path.is_file() or path.stat().st_size == 0:
        raise PreparationError(f'archive is missing or partial: {path}')
    try:
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            names = set(archive.namelist())
    except (OSError, zipfile.BadZipFile) as exc:
        raise PreparationError(f'archive verification failed: {path}: {exc}') from exc
    if bad is not None:
        raise PreparationError(f'archive CRC failure in {path}: {bad}')
    missing = [entry for entry in required_entries if entry not in names]
    if missing:
        raise PreparationError(f'archive missing required entries: {missing}')
    return {'path': str(path), 'sha256': _sha256(path.read_bytes()), 'entries': len(names)}


def download_archive(url: str, destination: Path, *, dry_run: bool = False) -> Path:
    '''Resume to a .partial file then atomically promote only a complete download.'''

    if dry_run:
        return destination.with_suffix(destination.suffix + '.partial')
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + '.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
    with urllib.request.urlopen(request) as response, partial.open('ab' if offset else 'wb') as handle:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            handle.write(block)
    partial.replace(destination)
    return destination


def acquire_tartanair_p000(destination: Path, *, dry_run: bool = False) -> Path:
    '''Compatibility wrapper: API is whole-archive; bounded P000 uses range ZIP extraction.'''
    from .tartanair import download_tartanair_whole_archive
    return download_tartanair_whole_archive(destination, dry_run=dry_run)

# Review-round implementations keep the public API stable while making official
# DTU source naming, render provenance, and manifest completeness fail closed.
from .overlay_round1 import A100Overlay  # noqa: E402
from .archive_round1 import download_archive, verify_archive  # noqa: E402
from .inventory_round1 import parse_dtu_inventory  # noqa: E402
# Round 2 provides the single public rendering and QA implementation.
from .preparation_round2 import (  # noqa: E402
    CALIBRATION_SCENES, atomic_json, build_split_view_manifest, calibration_qa, fog_render,
    load_prepared_batch, load_prepared_records, load_tartanair_prepared_pairs,
    low_light_noise_render, render_defocus, verify_dtu_scene,
)
