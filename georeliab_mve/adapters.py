'''External-model boundaries for frozen GeoReliab real-model inference.'''

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import io
from dataclasses import asdict, dataclass, field
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .contracts import PredictionArtifact, RunManifest, SampleKey, write_json_artifact
from .materialization import (
    FROZEN_TYPING_EXTENSIONS_DIST_INFO,
    FROZEN_TYPING_EXTENSIONS_SHA256,
    FROZEN_TYPING_EXTENSIONS_SITE,
    FROZEN_TYPING_EXTENSIONS_VERSION,
    require_git_commit,
    require_sha256,
    sha256_file,
)


class AdapterError(RuntimeError):
    '''Raised when frozen adapter validation or inference fails closed.'''


@dataclass(frozen=True, slots=True)
class GeometryIntervention:
    name: str
    source_scene: str | None = None
    hook_location: str | None = None


@dataclass(frozen=True, slots=True)
class CorruptionCondition:
    name: str
    severity: int


@dataclass(frozen=True, slots=True)
class RenderedView:
    view_id: int
    png_path: Path
    png_sha256: str
    width: int | None = None
    height: int | None = None
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.view_id, bool) or self.view_id < 0:
            raise AdapterError('view_id must be a non-negative integer')
        require_sha256(self.png_sha256, 'rendered PNG digest')
        if self.source_sha256 is not None:
            require_sha256(self.source_sha256, 'source PNG digest')
        if (self.width is None) != (self.height is None):
            raise AdapterError('rendered width and height must be provided together')
        if self.width is not None and (self.width <= 0 or self.height is None or self.height <= 0):
            raise AdapterError('rendered width/height must be positive')


@dataclass(frozen=True, slots=True)
class FrozenRuntime:
    source: Path
    source_commit: str
    environment: Path
    python_version: str
    torch_version: str
    checkpoint: Path
    checkpoint_sha256: str
    config: Path | None = None
    config_sha256: str | None = None
    dust3r_source: Path | None = None
    dust3r_source_commit: str | None = None
    croco_source: Path | None = None
    croco_source_commit: str | None = None
    typing_extensions_site: Path = FROZEN_TYPING_EXTENSIONS_SITE
    typing_extensions_sha256: str = FROZEN_TYPING_EXTENSIONS_SHA256
    typing_extensions_version: str = FROZEN_TYPING_EXTENSIONS_VERSION

    def __post_init__(self) -> None:
        require_git_commit(self.source_commit, 'source commit')
        require_sha256(self.checkpoint_sha256, 'checkpoint digest')
        require_sha256(self.typing_extensions_sha256, 'typing_extensions.py digest')
        if self.typing_extensions_version != FROZEN_TYPING_EXTENSIONS_VERSION:
            raise AdapterError(
                f'typing_extensions version mismatch: {self.typing_extensions_version} != {FROZEN_TYPING_EXTENSIONS_VERSION}'
            )
        if self.config is not None:
            if self.config_sha256 is None:
                raise AdapterError('config_sha256 is required when config is set')
            require_sha256(self.config_sha256, 'config digest')
        for path_label, commit_label, path_value, commit_value in ((
            ('dust3r_source', 'dust3r_source_commit', self.dust3r_source, self.dust3r_source_commit),
            ('croco_source', 'croco_source_commit', self.croco_source, self.croco_source_commit),
        )):
            if (path_value is None) != (commit_value is None):
                raise AdapterError(f'{path_label} and {commit_label} must be provided together')
            if commit_value is not None:
                require_git_commit(commit_value, commit_label)


@dataclass(frozen=True, slots=True)
class AdapterPreflight:
    model: str
    source_commit: str
    checkpoint_sha256: str
    environment: Mapping[str, str]
    config_sha256: str | None = None
    dust3r_source_commit: str | None = None
    croco_source_commit: str | None = None


@dataclass(frozen=True, slots=True)
class RenderedPngBatch:
    views: tuple[RenderedView, ...]
    png_bytes: tuple[bytes, ...]
    ordered_digest: str
    source_digests: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_views(cls, views: Sequence[RenderedView]) -> 'RenderedPngBatch':
        if len(views) != 8:
            raise AdapterError('GeoReliab adapters require exactly eight rendered PNG views')
        digest = hashlib.sha256()
        payloads: list[bytes] = []
        source_digests: dict[str, str] = {}
        seen: set[int] = set()
        for view in views:
            if view.view_id in seen:
                raise AdapterError(f'duplicate rendered view id: {view.view_id}')
            seen.add(view.view_id)
            if view.png_path.suffix.lower() != '.png':
                raise AdapterError(f'rendered input is not a PNG: {view.png_path}')
            data = view.png_path.read_bytes()
            actual = hashlib.sha256(data).hexdigest()
            if actual != view.png_sha256:
                raise AdapterError(f'rendered PNG digest mismatch for view {view.view_id}: {actual} != {view.png_sha256}')
            digest.update(str(view.view_id).encode('ascii'))
            digest.update(b'\0')
            digest.update(actual.encode('ascii'))
            digest.update(b'\0')
            payloads.append(data)
            if view.source_sha256 is not None:
                source_digests[str(view.view_id)] = view.source_sha256
        return cls(tuple(views), tuple(payloads), digest.hexdigest(), source_digests)


@dataclass(frozen=True, slots=True)
class AdapterOutput:
    points_world: np.ndarray
    camera_c2w: np.ndarray
    intrinsics: np.ndarray
    pixel_xy: np.ndarray
    view_id: np.ndarray
    raw_confidence: np.ndarray
    valid_mask: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


EnvProbe = Callable[[Path, Path, str], tuple[str, str, str, str, str]]
GitProbe = Callable[[Path], str]


def vggt_risk_from_depth_conf(depth_conf: np.ndarray) -> np.ndarray:
    return -np.log(np.maximum(np.asarray(depth_conf, dtype=np.float64) - 1.0, 1e-12))


def mast3r_risk_from_confidence(confidence: np.ndarray) -> np.ndarray:
    return -np.log(np.maximum(np.asarray(confidence, dtype=np.float64), 1e-12))


def _git_head(path: Path) -> str:
    try:
        result = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise AdapterError(f'cannot verify upstream git identity: {path}') from exc
    return result.stdout.strip().lower()


def _env_python(env_path: Path) -> Path:
    for candidate in (env_path / 'bin' / 'python', env_path / 'Scripts' / 'python.exe'):
        if candidate.is_file():
            return candidate
    raise AdapterError(f'frozen environment has no Python executable: {env_path}')


def _probe_python_torch(env_path: Path, typing_site: Path, typing_sha256: str) -> tuple[str, str, str, str, str]:
    env = dict(os.environ)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONNOUSERSITE'] = '1'
    code = (
        'import hashlib, platform, sys\n'
        'from pathlib import Path\n'
        'site = Path(sys.argv[1])\n'
        'expected_sha = sys.argv[2]\n'
        'sys.path.insert(0, str(site))\n'
        'import typing_extensions\n'
        'from importlib import metadata\n'
        f'expected_dist_info = (site / "{FROZEN_TYPING_EXTENSIONS_DIST_INFO}").resolve()\n'
        'actual_file = Path(typing_extensions.__file__).resolve()\n'
        'expected_file = (site / "typing_extensions.py").resolve()\n'
        'assert actual_file == expected_file, f"typing_extensions import escaped frozen site: {actual_file}"\n'
        'assert hashlib.sha256(actual_file.read_bytes()).hexdigest() == expected_sha\n'
        'dist = metadata.distribution("typing_extensions")\n'
        'dist_root = Path(dist.locate_file("")).resolve()\n'
        f'dist_info = Path(dist.locate_file("{FROZEN_TYPING_EXTENSIONS_DIST_INFO}")).resolve()\n'
        'assert dist_root == site.resolve(), f"typing_extensions distribution root escaped frozen site: {dist_root}"\n'
        'assert dist_info == expected_dist_info and dist_info.is_dir(), f"typing_extensions dist-info escaped frozen site: {dist_info}"\n'
        'assert dist.version == metadata.version("typing_extensions")\n'
        'import torch\n'
        'print(platform.python_version())\n'
        'print(torch.__version__)\n'
        'print(metadata.version("typing_extensions"))\n'
        'print(str(actual_file))\n'
        'print(str(dist_info))\n'
    )
    try:
        result = subprocess.run([str(_env_python(env_path)), '-I', '-B', '-c', code, str(typing_site), typing_sha256], check=True, capture_output=True, text=True, timeout=60, env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        raise AdapterError(f'cannot verify Python/Torch/typing_extensions versions in {env_path}') from exc
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 5:
        raise AdapterError(f'frozen environment emitted invalid version evidence: {env_path}')
    return lines[0], lines[1], lines[2], lines[3], lines[4]


def verify_frozen_runtime(model: str, runtime: FrozenRuntime, *, git_probe: GitProbe = _git_head, env_probe: EnvProbe = _probe_python_torch) -> AdapterPreflight:
    actual_source = git_probe(runtime.source).lower()
    if actual_source != runtime.source_commit:
        raise AdapterError(f'{model} source commit mismatch: {actual_source} != {runtime.source_commit}')
    if not runtime.checkpoint.is_file():
        raise AdapterError(f'{model} checkpoint is missing: {runtime.checkpoint}')
    actual_checkpoint = sha256_file(runtime.checkpoint)
    if actual_checkpoint != runtime.checkpoint_sha256:
        raise AdapterError(f'{model} checkpoint SHA-256 mismatch: {actual_checkpoint} != {runtime.checkpoint_sha256}')
    actual_config: str | None = None
    if runtime.config is not None:
        if not runtime.config.is_file():
            raise AdapterError(f'{model} config is missing: {runtime.config}')
        actual_config = sha256_file(runtime.config)
        if actual_config != runtime.config_sha256:
            raise AdapterError(f'{model} config SHA-256 mismatch: {actual_config} != {runtime.config_sha256}')
    typing_path = runtime.typing_extensions_site / 'typing_extensions.py'
    if not typing_path.is_file():
        raise AdapterError(f'{model} frozen typing_extensions.py is missing: {typing_path}')
    actual_typing_sha = sha256_file(typing_path)
    if actual_typing_sha != runtime.typing_extensions_sha256:
        raise AdapterError(
            f'{model} typing_extensions.py SHA-256 mismatch: {actual_typing_sha} != {runtime.typing_extensions_sha256}'
        )
    python, torch, typing_version, typing_file, typing_dist_info = env_probe(
        runtime.environment, runtime.typing_extensions_site, runtime.typing_extensions_sha256
    )
    if python != runtime.python_version or torch != runtime.torch_version:
        raise AdapterError(f'{model} environment mismatch: Python {python}/Torch {torch} != Python {runtime.python_version}/Torch {runtime.torch_version}')
    if typing_version != runtime.typing_extensions_version:
        raise AdapterError(
            f'{model} typing_extensions mismatch: {typing_version} != {runtime.typing_extensions_version}'
        )
    try:
        actual_typing_file = Path(typing_file).resolve()
    except OSError as exc:
        raise AdapterError(f'{model} typing_extensions module origin is unreadable: {typing_file}') from exc
    if actual_typing_file != typing_path.resolve():
        raise AdapterError(
            f'{model} typing_extensions module origin mismatch: {actual_typing_file} != {typing_path.resolve()}'
        )
    expected_dist_info = (runtime.typing_extensions_site / FROZEN_TYPING_EXTENSIONS_DIST_INFO).resolve()
    try:
        actual_dist_info = Path(typing_dist_info).resolve()
    except OSError as exc:
        raise AdapterError(f'{model} typing_extensions dist-info origin is unreadable: {typing_dist_info}') from exc
    if actual_dist_info != expected_dist_info or not actual_dist_info.is_dir():
        raise AdapterError(
            f'{model} typing_extensions dist-info origin mismatch: {actual_dist_info} != {expected_dist_info}'
        )
    dust3r_commit = None
    croco_commit = None
    if model == 'MASt3R' and (runtime.dust3r_source is None or runtime.dust3r_source_commit is None or runtime.croco_source is None or runtime.croco_source_commit is None):
        raise AdapterError('MASt3R preflight requires frozen DUSt3R and CroCo source paths and commits')
    if runtime.dust3r_source is not None:
        dust3r_commit = git_probe(runtime.dust3r_source).lower()
        if dust3r_commit != runtime.dust3r_source_commit:
            raise AdapterError('MASt3R DUSt3R source commit mismatch')
    if runtime.croco_source is not None:
        croco_commit = git_probe(runtime.croco_source).lower()
        if croco_commit != runtime.croco_source_commit:
            raise AdapterError('MASt3R CroCo source commit mismatch')
    return AdapterPreflight(model, actual_source, actual_checkpoint, {
        'path': str(runtime.environment),
        'python': python,
        'torch': torch,
        'typing_extensions_site': str(runtime.typing_extensions_site),
        'typing_extensions_file': typing_file,
        'typing_extensions_dist_info': typing_dist_info,
        'typing_extensions_version': typing_version,
        'typing_extensions_sha256': runtime.typing_extensions_sha256,
    }, actual_config, dust3r_commit, croco_commit)


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode('utf-8')).hexdigest()


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        if not value:
            return np.asarray(value)
        converted = [_to_numpy(item) for item in value]
        try:
            return np.stack(converted, axis=0)
        except ValueError:
            return np.asarray(converted)
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value)


def _digest_arraylike(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, Mapping):
            for key in sorted(item):
                digest.update(str(key).encode('utf-8'))
                update(item[key])
            return
        if isinstance(item, (list, tuple)):
            digest.update(f'len={len(item)}'.encode('ascii'))
            for child in item:
                update(child)
            return
        array = _to_numpy(item)
        digest.update(str(array.shape).encode('ascii'))
        digest.update(str(array.dtype).encode('ascii'))
        digest.update(np.ascontiguousarray(array).tobytes())

    update(value)
    return digest.hexdigest()


def _squeeze_model_volume(value: Any, name: str) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim == 5 and array.shape[0] == 1 and array.shape[-1] == 1:
        array = array[0, ..., 0]
    elif array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    elif array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise AdapterError(f'{name} must have shape (V,H,W), optionally with batch/trailing singleton')
    return array


def _normalize_mast3r_dense_output(
    points: Any,
    confidence: Any,
) -> tuple[np.ndarray, np.ndarray]:
    points_by_view = _to_numpy(points).astype(np.float64, copy=False)
    confidence_by_view = _to_numpy(confidence).astype(np.float64, copy=False)
    if confidence_by_view.ndim != 3:
        raise AdapterError('MASt3R confidence must have shape (V,H,W)')
    view_count, height, width = confidence_by_view.shape
    dense_shape = (view_count, height, width, 3)
    flattened_shape = (view_count, height * width, 3)
    if points_by_view.shape == flattened_shape:
        points_by_view = points_by_view.reshape(dense_shape)
    elif points_by_view.shape != dense_shape:
        raise AdapterError(
            'MASt3R pts3d must match confidence as (V,H,W,3) or (V,H*W,3)'
        )
    return points_by_view, confidence_by_view


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_camera_stack(camera_c2w: Any, view_count: int) -> np.ndarray:
    array = _to_numpy(camera_c2w).astype(np.float64, copy=False)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.shape == (view_count, 3, 4):
        bottom = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), (view_count, 1, 4))
        array = np.concatenate([array, bottom], axis=1)
    if array.shape != (view_count, 4, 4):
        raise AdapterError(f'camera_c2w must have shape ({view_count},4,4)')
    return array


def _normalize_intrinsics(intrinsics: Any, view_count: int) -> np.ndarray:
    array = _to_numpy(intrinsics).astype(np.float64, copy=False)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (view_count, 3, 3):
        raise AdapterError(f'intrinsics must have shape ({view_count},3,3)')
    return array


def source_pixel_grid(height: int, width: int, source_height: int | None, source_width: int | None) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    if source_height is None or source_width is None:
        return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float64)
    # Pixel-center resize inverse: model pixel center -> source pixel center.
    sx = source_width / width
    sy = source_height / height
    src_x = (xx.reshape(-1) + 0.5) * sx - 0.5
    src_y = (yy.reshape(-1) + 0.5) * sy - 0.5
    return np.stack([src_x, src_y], axis=1).astype(np.float64)


def _depth_to_points_world(depth: np.ndarray, camera_c2w: np.ndarray, intrinsics: np.ndarray, views: Sequence[RenderedView]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 3:
        raise AdapterError('depth must have shape (V,H,W)')
    camera_c2w = _normalize_camera_stack(camera_c2w, depth.shape[0])
    intrinsics = _normalize_intrinsics(intrinsics, depth.shape[0])
    points_world: list[np.ndarray] = []
    pixels: list[np.ndarray] = []
    view_ids: list[np.ndarray] = []
    for local_view, depth_map in enumerate(depth):
        height, width = depth_map.shape
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        pixel_h = np.stack([xx, yy, np.ones_like(xx)], axis=-1).reshape(-1, 3).astype(np.float64)
        try:
            rays = pixel_h @ np.linalg.inv(intrinsics[local_view]).T
        except np.linalg.LinAlgError:
            rays = np.full((height * width, 3), np.nan, dtype=np.float64)
        camera_points = rays * depth_map.reshape(-1, 1)
        homogeneous = np.concatenate([camera_points, np.ones((len(camera_points), 1))], axis=1)
        world = homogeneous @ camera_c2w[local_view].T
        points_world.append(world[:, :3])
        pixels.append(source_pixel_grid(height, width, views[local_view].height, views[local_view].width))
        view_ids.append(np.full((height * width,), views[local_view].view_id, dtype=np.int64))
    return np.concatenate(points_world), np.concatenate(pixels), np.concatenate(view_ids)


def _valid_output_mask(points: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=1) & np.isfinite(confidence)


def _has_singular_intrinsics(intrinsics: np.ndarray) -> bool:
    if intrinsics.size == 0 or not np.isfinite(intrinsics).all():
        return True
    for matrix in intrinsics:
        try:
            if np.linalg.matrix_rank(matrix) < 3 or abs(float(np.linalg.det(matrix))) <= 1e-12:
                return True
        except np.linalg.LinAlgError:
            return True
    return False


def _has_degenerate_camera(camera_c2w: np.ndarray) -> bool:
    if camera_c2w.size == 0 or not np.isfinite(camera_c2w).all():
        return True
    for matrix in camera_c2w:
        linear = matrix[:3, :3]
        try:
            if np.linalg.matrix_rank(linear) < 3 or abs(float(np.linalg.det(linear))) <= 1e-12:
                return True
        except np.linalg.LinAlgError:
            return True
    return False


def _output_validation_failures(model: str, output: AdapterOutput) -> list[str]:
    failures: list[str] = []
    if output.points_world.size == 0:
        failures.append(f'{model} produced empty geometry')
    if output.raw_confidence.size == 0:
        failures.append(f'{model} produced empty native confidence')
    if output.points_world.shape[0] != output.raw_confidence.shape[0] or output.points_world.shape[0] != output.valid_mask.shape[0]:
        failures.append(f'{model} output point/confidence/mask lengths do not match')
    if output.points_world.size and not np.isfinite(output.points_world).all():
        failures.append(f'{model} produced non-finite geometry')
    if output.raw_confidence.size and not np.isfinite(output.raw_confidence).all():
        failures.append(f'{model} produced non-finite native confidence')
    if _has_singular_intrinsics(output.intrinsics):
        failures.append(f'{model} produced singular or non-finite intrinsics')
    if _has_degenerate_camera(output.camera_c2w):
        failures.append(f'{model} produced degenerate or non-finite cameras')
    return failures

def _empty_output(metadata: Mapping[str, Any]) -> AdapterOutput:
    return AdapterOutput(np.empty((0, 3)), np.empty((0, 4, 4)), np.empty((0, 3, 3)), np.empty((0, 2)), np.empty((0,), dtype=np.int64), np.empty((0,)), np.empty((0,), dtype=bool), dict(metadata))


def _lazy_import(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise AdapterError(f'missing upstream dependency {module_name!r}; install/use the frozen upstream environment') from exc


def _prepend_source_path(source: Path) -> None:
    text = str(source)
    if text not in sys.path:
        sys.path.insert(0, text)


class RealVGGTUpstream:
    def __init__(self, runtime: FrozenRuntime, device: str = 'cuda:0') -> None:
        self.runtime = runtime
        self.device = device
        self._model: Any | None = None

    def preprocess(self, png_paths: Sequence[Path]) -> tuple[Any, str]:
        _prepend_source_path(self.runtime.source)
        load_fn = _lazy_import('vggt.utils.load_fn')
        images = load_fn.load_and_preprocess_images([str(path) for path in png_paths])
        return images, _stable_digest({'shape': tuple(getattr(images, 'shape', ())), 'dtype': str(getattr(images, 'dtype', ''))})

    def load_model(self) -> Any:
        if self._model is None:
            _prepend_source_path(self.runtime.source)
            torch = _lazy_import('torch')
            model_module = _lazy_import('vggt.models.vggt')
            model = model_module.VGGT()
            state = torch.load(str(self.runtime.checkpoint), map_location='cpu', weights_only=True)
            model.load_state_dict(state)
            model.eval()
            model.to(self.device)
            self._model = model
        return self._model

    def infer(self, images: Any) -> Mapping[str, Any]:
        torch = _lazy_import('torch')
        model = self.load_model()
        with torch.no_grad():
            if str(self.device).startswith('cuda'):
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    return model(images.to(self.device) if hasattr(images, 'to') else images)
            return model(images)

    def camera_depth_to_world(self, predictions: Mapping[str, Any], images_shape: Sequence[int]) -> tuple[Any, Any, Any]:
        _prepend_source_path(self.runtime.source)
        pose = _lazy_import('vggt.utils.pose_enc')
        geometry = _lazy_import('vggt.utils.geometry')
        extrinsic, intrinsic = pose.pose_encoding_to_extri_intri(predictions['pose_enc'], images_shape[-2:])
        points = geometry.unproject_depth_map_to_point_map(predictions['depth'].squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0))
        return points, extrinsic, intrinsic


class RealMASt3RUpstream:
    MATCHING_CONFIDENCE_DISABLED = float('-inf')

    def __init__(self, runtime: FrozenRuntime, device: str = 'cuda:0', cache_dir: Path | None = None) -> None:
        self.runtime = runtime
        self.device = device
        if cache_dir is None:
            raise AdapterError('MASt3R upstream requires an explicit writable cache_dir outside frozen /home resources')
        self.cache_dir = cache_dir
        self._model: Any | None = None

    def preprocess(self, png_paths: Sequence[Path]) -> tuple[Any, str]:
        _prepend_source_path(self.runtime.source)
        if self.runtime.dust3r_source is not None:
            _prepend_source_path(self.runtime.dust3r_source)
        if self.runtime.croco_source is not None:
            _prepend_source_path(self.runtime.croco_source)
        image_module = _lazy_import('dust3r.utils.image')
        images = image_module.load_images([str(path) for path in png_paths], size=512)
        return images, _stable_digest({'count': len(images), 'size': 512})

    def load_model(self) -> Any:
        if self._model is None:
            _prepend_source_path(self.runtime.source)
            model_module = _lazy_import('mast3r.model')
            model = model_module.AsymmetricMASt3R.from_pretrained(str(self.runtime.checkpoint.parent), local_files_only=True)
            model.eval()
            model.to(self.device)
            self._model = model
        return self._model

    def infer(self, image_paths: Sequence[Path], images: Any) -> Mapping[str, Any]:
        _prepend_source_path(self.runtime.source)
        if self.runtime.dust3r_source is not None:
            _prepend_source_path(self.runtime.dust3r_source)
        if self.runtime.croco_source is not None:
            _prepend_source_path(self.runtime.croco_source)
        pairs_module = _lazy_import('mast3r.image_pairs')
        cloud_module = _lazy_import('mast3r.cloud_opt.sparse_ga')
        pairs = pairs_module.make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        scene = cloud_module.sparse_global_alignment(
            [str(path) for path in image_paths],
            pairs,
            cache_path=str(self.cache_dir),
            model=self.load_model(),
            lr1=0.07,
            niter1=300,
            lr2=0.01,
            niter2=300,
            device=self.device,
            opt_depth=True,
            shared_intrinsics=False,
            matching_conf_thr=self.MATCHING_CONFIDENCE_DISABLED,
        )
        pts3d, _depthmaps, confs = scene.get_dense_pts3d(clean_depth=False)
        return {'pts3d': pts3d, 'conf': confs, 'camera_c2w': scene.get_im_poses(), 'intrinsics': scene.intrinsics, 'raw_pairwise_cache_files': collect_mast3r_cache_trace(self.cache_dir), 'matching_conf_thr': self.MATCHING_CONFIDENCE_DISABLED}





def _validate_manifest_matches_runtime(model: str, manifest: RunManifest, runtime: FrozenRuntime, preflight: AdapterPreflight) -> None:
    if manifest.model != model:
        raise AdapterError(f'manifest model mismatch: {manifest.model} != {model}')
    if manifest.checkpoint_hash != preflight.checkpoint_sha256:
        raise AdapterError('manifest checkpoint_hash does not match frozen checkpoint')
    if manifest.provenance is None:
        raise AdapterError('real adapter inference requires RunManifest provenance')
    if manifest.provenance.model_source_commit != preflight.source_commit:
        raise AdapterError('manifest model_source_commit does not match frozen source')
    if model == 'MASt3R':
        if manifest.provenance.dust3r_source_commit != preflight.dust3r_source_commit:
            raise AdapterError('manifest DUSt3R source commit does not match frozen source')
        if manifest.provenance.croco_source_commit != preflight.croco_source_commit:
            raise AdapterError('manifest CroCo source commit does not match frozen source')

def collect_mast3r_cache_trace(cache_dir: Path) -> tuple[dict[str, str], ...]:
    if not cache_dir.exists():
        return ()
    rows: list[dict[str, str]] = []
    for child in sorted(cache_dir.rglob('*.pth')):
        try:
            relative = child.relative_to(cache_dir).as_posix()
        except ValueError:
            relative = child.name
        if relative.startswith('forward/') or relative.startswith('corres_'):
            rows.append({'relative_path': relative, 'uri': child.as_uri(), 'path': str(child), 'sha256': _file_digest(child), 'format': 'mast3r-sparse-ga-torch-pth'})
    return tuple(rows)

def _write_pairwise_trace(result: Mapping[str, Any], output_dir: Path, prefix: str) -> tuple[str, str, list[dict[str, str]]] | None:
    arrays: dict[str, np.ndarray] = {}
    skipped: list[dict[str, str]] = []
    if 'pairwise_pts3d' in result:
        arrays['pairwise_pts3d'] = _to_numpy(result['pairwise_pts3d'])
    if 'pairwise_conf' in result:
        arrays['pairwise_conf'] = _to_numpy(result['pairwise_conf'])
    raw = result.get('raw_pairwise')
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            try:
                array = _to_numpy(value)
            except Exception as exc:
                skipped.append({'key': str(key), 'exception_type': type(exc).__name__})
                continue
            if array.dtype == object:
                skipped.append({'key': str(key), 'exception_type': 'ObjectDTypeUnsupported'})
                continue
            arrays[f'raw_pairwise_{key}'] = array
    if not arrays:
        if skipped:
            return '', '', skipped
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'{prefix}_raw_pairwise_trace.npz'
    _write_deterministic_npz(path, arrays)
    return path.as_uri(), _file_digest(path), skipped
def _write_deterministic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(path), mode='w', compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f'{name}.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())


def _write_output_payloads(output: AdapterOutput, output_dir: Path, prefix: str) -> tuple[Path, Path, Path, Mapping[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = output_dir / f'{prefix}_geometry.npz'
    confidence_path = output_dir / f'{prefix}_confidence.npz'
    valid_path = output_dir / f'{prefix}_valid_mask.npz'
    _write_deterministic_npz(geometry_path, {'camera_c2w': output.camera_c2w, 'intrinsics': output.intrinsics, 'metadata': json.dumps(dict(output.metadata), sort_keys=True), 'pixel_xy': output.pixel_xy, 'points_world': output.points_world, 'view_id': output.view_id})
    _write_deterministic_npz(confidence_path, {'raw_confidence': output.raw_confidence})
    _write_deterministic_npz(valid_path, {'valid_mask': output.valid_mask.astype(bool, copy=False)})
    return geometry_path, confidence_path, valid_path, {'geometry_prediction_uri': _file_digest(geometry_path), 'native_confidence_uri': _file_digest(confidence_path), 'valid_mask_uri': _file_digest(valid_path)}

def serialize_prediction_output(*, manifest: RunManifest, sample_key: SampleKey, output: AdapterOutput, output_dir: Path, prefix: str, runtime_seconds: float, peak_memory_mb: float, invalid_prediction: bool, write_prediction_json: bool = True) -> PredictionArtifact:
    geometry_path, confidence_path, valid_path, digests = _write_output_payloads(output, output_dir, prefix)
    prediction = PredictionArtifact(manifest.run_id, str(sample_key), geometry_path.as_uri(), confidence_path.as_uri(), valid_path.as_uri(), None, runtime_seconds, peak_memory_mb, invalid_prediction, digests)
    if write_prediction_json:
        write_json_artifact(output_dir / f'{prefix}_prediction.json', prediction)
    return prediction


def _peak_cuda_memory_mb(device: str) -> float:
    if not str(device).startswith('cuda'):
        return 0.0
    try:
        torch = importlib.import_module('torch')
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    except ImportError:
        pass
    return 0.0


class VGGTAdapter:
    model_name = 'VGGT'

    def __init__(self, runtime: FrozenRuntime, *, output_root: Path, device: str = 'cuda:0', upstream: Any | None = None, git_probe: GitProbe = _git_head, env_probe: EnvProbe = _probe_python_torch) -> None:
        self.runtime = runtime
        self.output_root = output_root
        self.device = device
        self.upstream = upstream if upstream is not None else RealVGGTUpstream(runtime, device=device)
        self._git_probe = git_probe
        self._env_probe = env_probe

    def reproducible(self) -> bool:
        return False

    def preflight(self) -> AdapterPreflight:
        return verify_frozen_runtime(self.model_name, self.runtime, git_probe=self._git_probe, env_probe=self._env_probe)

    def predict_sample(self, manifest: RunManifest, sample_key: SampleKey, rendered_views: Sequence[RenderedView]) -> PredictionArtifact:
        start = time.perf_counter()
        batch = RenderedPngBatch.from_views(rendered_views)
        prefix = f'{manifest.run_id}_{self.model_name.lower()}'
        metadata: dict[str, Any] = {'model': self.model_name, 'ordered_png_digest': batch.ordered_digest, 'rendered_png_sha256': {str(view.view_id): view.png_sha256 for view in batch.views}, 'source_png_sha256': dict(batch.source_digests), 'view_ids': [view.view_id for view in batch.views], 'risk_definition': '-log(max(depth_conf - 1, 1e-12))', 'primary_geometry': 'depth+predicted-camera-unprojection', 'deterministic': False, 'nondeterministic_reasons': ['cuda kernels', 'upstream model inference']}
        invalid = False
        try:
            preflight = self.preflight()
            _validate_manifest_matches_runtime(self.model_name, manifest, self.runtime, preflight)
            metadata['preflight'] = asdict(preflight)
            images, preprocessing_digest = self.upstream.preprocess([view.png_path for view in batch.views])
            metadata['upstream_preprocessing_digest'] = preprocessing_digest
            metadata['preprocessing_digest'] = _digest_arraylike(images)
            predictions = self.upstream.infer(images)
            if hasattr(self.upstream, 'camera_depth_to_world'):
                point_map, extrinsic, intrinsic = self.upstream.camera_depth_to_world(predictions, getattr(images, 'shape', np.asarray(images).shape))
                points_by_view = _to_numpy(point_map).astype(np.float64, copy=False)
                if points_by_view.ndim == 5 and points_by_view.shape[0] == 1:
                    points_by_view = points_by_view[0]
                depth = _squeeze_model_volume(predictions['depth'], 'VGGT depth')
                view_count, height, width = depth.shape
                camera_w2c = _normalize_camera_stack(_to_numpy(extrinsic), view_count)
                camera_c2w = np.linalg.inv(camera_w2c)
                intrinsics = _normalize_intrinsics(_to_numpy(intrinsic), view_count)
                if points_by_view.shape == (view_count, height, width, 3):
                    points = points_by_view.reshape(-1, 3)
                    pixels = np.concatenate([source_pixel_grid(height, width, view.height, view.width) for view in batch.views])
                    view_id = np.concatenate([np.full((height * width,), batch.views[index].view_id, dtype=np.int64) for index in range(view_count)])
                else:
                    points, pixels, view_id = _depth_to_points_world(depth, camera_c2w, intrinsics, batch.views)
            else:
                depth = _squeeze_model_volume(predictions['depth'], 'VGGT depth')
                camera_c2w = _normalize_camera_stack(_to_numpy(predictions['camera_c2w']), depth.shape[0])
                intrinsics = _normalize_intrinsics(_to_numpy(predictions['intrinsics']), depth.shape[0])
                points, pixels, view_id = _depth_to_points_world(depth, camera_c2w, intrinsics, batch.views)
            confidence = _squeeze_model_volume(predictions['depth_conf'], 'VGGT depth_conf').astype(np.float64, copy=False)
            raw_conf = confidence.reshape(-1)
            valid = _valid_output_mask(points, raw_conf)
            output = AdapterOutput(points, camera_c2w, intrinsics, pixels, view_id, raw_conf, valid, metadata)
            failures = _output_validation_failures(self.model_name, output)
            if failures:
                invalid = True
                metadata['failure_type'] = 'AdapterOutputValidationError'
                metadata['failure_message'] = '; '.join(failures)
                metadata['validation_failures'] = failures
        except Exception as exc:
            invalid = True
            metadata['failure_type'] = type(exc).__name__
            metadata['failure_message'] = str(exc)
            output = _empty_output(metadata)
        return serialize_prediction_output(manifest=manifest, sample_key=sample_key, output=output, output_dir=self.output_root, prefix=prefix, runtime_seconds=time.perf_counter() - start, peak_memory_mb=_peak_cuda_memory_mb(self.device), invalid_prediction=invalid)

    def predict(self, manifest: RunManifest, sample_key: SampleKey, condition: CorruptionCondition) -> PredictionArtifact:
        raise AdapterError('VGGTAdapter.predict requires rendered PNGs; use predict_sample')


class MASt3RAdapter:
    model_name = 'MASt3R'

    def __init__(self, runtime: FrozenRuntime, *, output_root: Path, device: str = 'cuda:0', cache_dir: Path | None = None, upstream: Any | None = None, git_probe: GitProbe = _git_head, env_probe: EnvProbe = _probe_python_torch) -> None:
        self.runtime = runtime
        self.output_root = output_root
        self.device = device
        self.cache_dir = cache_dir or (output_root / 'mast3r_cache')
        self.upstream = upstream if upstream is not None else RealMASt3RUpstream(runtime, device=device, cache_dir=self.cache_dir)
        self._git_probe = git_probe
        self._env_probe = env_probe

    def reproducible(self) -> bool:
        return False

    def preflight(self) -> AdapterPreflight:
        return verify_frozen_runtime(self.model_name, self.runtime, git_probe=self._git_probe, env_probe=self._env_probe)

    def predict_sample(self, manifest: RunManifest, sample_key: SampleKey, rendered_views: Sequence[RenderedView]) -> PredictionArtifact:
        start = time.perf_counter()
        batch = RenderedPngBatch.from_views(rendered_views)
        prefix = f'{manifest.run_id}_{self.model_name.lower()}'
        metadata: dict[str, Any] = {'model': self.model_name, 'ordered_png_digest': batch.ordered_digest, 'rendered_png_sha256': {str(view.view_id): view.png_sha256 for view in batch.views}, 'source_png_sha256': dict(batch.source_digests), 'view_ids': [view.view_id for view in batch.views], 'risk_definition': '-log(max(conf, 1e-12))', 'primary_geometry': 'sparse-global-alignment-dense-pts3d', 'matching_conf_thr': 'disabled:-inf', 'prohibited_postprocessing': ['clean_pointcloud', 'TSDF', 'confidence_threshold'], 'deterministic': False, 'nondeterministic_reasons': ['cuda kernels', 'upstream global alignment']}
        invalid = False
        try:
            preflight = self.preflight()
            _validate_manifest_matches_runtime(self.model_name, manifest, self.runtime, preflight)
            metadata['preflight'] = asdict(preflight)
            images, preprocessing_digest = self.upstream.preprocess([view.png_path for view in batch.views])
            metadata['upstream_preprocessing_digest'] = preprocessing_digest
            metadata['preprocessing_digest'] = _digest_arraylike(images)
            result = self.upstream.infer([view.png_path for view in batch.views], images)
            forbidden = result.get('forbidden_operations', ()) if isinstance(result, Mapping) else ()
            if any(str(item).lower() in {'clean_pointcloud', 'tsdf', 'confidence_threshold'} for item in forbidden):
                raise AdapterError('MASt3R adapter forbids clean_pointcloud, TSDF, and confidence thresholds')
            pts_by_view, conf_by_view = _normalize_mast3r_dense_output(
                result['pts3d'], result['conf']
            )
            view_count, height, width, _ = pts_by_view.shape
            camera_c2w = _normalize_camera_stack(_to_numpy(result['camera_c2w']), view_count)
            intrinsics = _normalize_intrinsics(_to_numpy(result['intrinsics']), view_count)
            pixels = np.concatenate([source_pixel_grid(height, width, view.height, view.width) for view in batch.views])
            view_id = np.concatenate([np.full((height * width,), batch.views[index].view_id, dtype=np.int64) for index in range(view_count)])
            points = pts_by_view.reshape(-1, 3)
            raw_conf = conf_by_view.reshape(-1)
            valid = _valid_output_mask(points, raw_conf)
            output = AdapterOutput(points, camera_c2w, intrinsics, pixels, view_id, raw_conf, valid, metadata)
            failures = _output_validation_failures(self.model_name, output)
            if failures:
                invalid = True
                metadata['failure_type'] = 'AdapterOutputValidationError'
                metadata['failure_message'] = '; '.join(failures)
                metadata['validation_failures'] = failures
            cache_trace = result.get('raw_pairwise_cache_files', ())
            if cache_trace:
                metadata['raw_pairwise_cache_files'] = list(cache_trace)
            try:
                trace = _write_pairwise_trace(result, self.output_root, prefix)
            except Exception as exc:
                invalid = True
                metadata['failure_type'] = 'AdapterPairwiseTraceError'
                metadata['failure_message'] = f'failed to serialize numeric MASt3R pairwise trace: {type(exc).__name__}: {exc}'
                metadata['raw_pairwise_trace_status'] = 'numeric-trace-conversion-failed'
            else:
                if trace is not None:
                    if trace[0]:
                        metadata['raw_pairwise_trace_uri'] = trace[0]
                        metadata['raw_pairwise_trace_sha256'] = trace[1]
                    if trace[2]:
                        metadata['raw_pairwise_skipped'] = trace[2]
                elif result.get('raw_pairwise') is not None:
                    metadata['raw_pairwise_present'] = True
                    metadata['raw_pairwise_trace_status'] = 'metadata-only-no-numeric-pts3d-conf-exposed'
        except Exception as exc:
            invalid = True
            metadata['failure_type'] = type(exc).__name__
            metadata['failure_message'] = str(exc)
            output = _empty_output(metadata)
        return serialize_prediction_output(manifest=manifest, sample_key=sample_key, output=output, output_dir=self.output_root, prefix=prefix, runtime_seconds=time.perf_counter() - start, peak_memory_mb=_peak_cuda_memory_mb(self.device), invalid_prediction=invalid)

    def predict(self, manifest: RunManifest, sample_key: SampleKey, condition: CorruptionCondition) -> PredictionArtifact:
        raise AdapterError('MASt3RAdapter.predict requires rendered PNGs; use predict_sample')


@runtime_checkable
class GeometryModelAdapter(Protocol):
    '''Adapter must keep RGB, prompt, decoder, and seed fixed across arms.'''

    @property
    def model_name(self) -> str: ...

    def reproducible(self) -> bool: ...

    def hook_locations(self) -> tuple[str, ...]: ...

    def predict(self, manifest: RunManifest, sample_key: SampleKey, intervention: GeometryIntervention) -> PredictionArtifact: ...


@runtime_checkable
class GeoReliabModelAdapter(Protocol):
    '''Frozen GFM adapter exposing native confidence and geometry output.'''

    @property
    def model_name(self) -> str: ...

    def reproducible(self) -> bool: ...

    def predict(self, manifest: RunManifest, sample_key: SampleKey, condition: CorruptionCondition) -> PredictionArtifact: ...
