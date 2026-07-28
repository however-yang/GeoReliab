from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from georeliab_mve.adapters import (
    AdapterError,
    AdapterOutput,
    FrozenRuntime,
    MASt3RAdapter,
    RealMASt3RUpstream,
    RealVGGTUpstream,
    RenderedPngBatch,
    RenderedView,
    VGGTAdapter,
    mast3r_risk_from_confidence,
    serialize_prediction_output,
    source_pixel_grid,
    verify_frozen_runtime,
    vggt_risk_from_depth_conf,
)
from georeliab_mve.contracts import RunManifest, RunMode, SampleKey, ScientificProvenance, ScientificValidity
from georeliab_mve.contracts import _file_uri_path

GIT = 'a' * 40
SHA = 'b' * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime(tmp_path: Path, *, mast3r: bool = False) -> FrozenRuntime:
    ckpt = tmp_path / 'model.pt'
    ckpt.write_bytes(b'checkpoint')
    cfg = tmp_path / 'config.json'
    if mast3r:
        cfg.write_text('{}', encoding='utf-8')
    typing_site = tmp_path / 'typing-site'
    typing_site.mkdir(exist_ok=True)
    typing_file = typing_site / 'typing_extensions.py'
    typing_file.write_text('# no module __version__ on the frozen source\n', encoding='utf-8')
    dist = typing_site / 'typing_extensions-4.15.0.dist-info'
    dist.mkdir(exist_ok=True)
    (dist / 'METADATA').write_text('Name: typing_extensions\nVersion: 4.15.0\n', encoding='utf-8')
    return FrozenRuntime(
        source=tmp_path / ('mast3r' if mast3r else 'vggt'),
        source_commit='1' * 40,
        environment=tmp_path / ('mast3r-env' if mast3r else 'vggt-env'),
        python_version='3.10.20',
        torch_version='2.3.1+cu121' if not mast3r else '2.5.1+cu121',
        checkpoint=ckpt,
        checkpoint_sha256=_sha(ckpt),
        config=cfg if mast3r else None,
        config_sha256=_sha(cfg) if mast3r else None,
        dust3r_source=tmp_path / 'dust3r' if mast3r else None,
        dust3r_source_commit='2' * 40 if mast3r else None,
        croco_source=tmp_path / 'croco' if mast3r else None,
        croco_source_commit='3' * 40 if mast3r else None,
        typing_extensions_site=typing_site,
        typing_extensions_sha256=_sha(typing_file),
        typing_extensions_version='4.15.0',
    )


def _manifest(model: str, runtime: FrozenRuntime, *, mast3r: bool = False) -> RunManifest:
    return RunManifest(
        run_id=f'RUN{model.replace("3", "")}',
        mode=RunMode.REAL,
        scientific_validity=ScientificValidity.SCIENTIFIC,
        model=model,
        checkpoint_hash=runtime.checkpoint_sha256,
        dataset='dtu',
        split='test',
        seed=0,
        intervention_version='none',
        corruption_version='clean',
        environment={'python': '3.10.20'},
        rgb_digest='rgb',
        prompt_digest='prompt',
        decoder_digest='decoder',
        provenance=ScientificProvenance(
            project_commit='4' * 40,
            project_tree='5' * 40,
            model_source_commit=runtime.source_commit,
            environment_lock_sha256='6' * 64,
            corruption_manifest_sha256='7' * 64,
            split_view_manifest_sha256='8' * 64,
            dust3r_source_commit=runtime.dust3r_source_commit if mast3r else None,
            croco_source_commit=runtime.croco_source_commit if mast3r else None,
        ),
    )


def _views(tmp_path: Path) -> list[RenderedView]:
    out = []
    for i in range(8):
        p = tmp_path / f'v{i}.png'
        p.write_bytes(b'PNG' + bytes([i]))
        out.append(RenderedView(i, p, _sha(p), width=1600, height=1200, source_sha256='c' * 64))
    return out


def _git(path: Path) -> str:
    text = str(path)
    if text.endswith('dust3r'):
        return '2' * 40
    if text.endswith('croco'):
        return '3' * 40
    return '1' * 40


def _env(path: Path, typing_site: Path, typing_sha: str) -> tuple[str, str, str, str, str]:
    return (
        '3.10.20',
        '2.5.1+cu121' if 'mast3r' in str(path).lower() else '2.3.1+cu121',
        '4.15.0',
        str(typing_site / 'typing_extensions.py'),
        str(typing_site / 'typing_extensions-4.15.0.dist-info'),
    )


def test_lazy_import_reports_missing_upstream_without_importing_at_module_import(tmp_path):
    runtime = _runtime(tmp_path)
    upstream = RealVGGTUpstream(runtime, device='cpu')
    with pytest.raises(AdapterError, match='missing upstream dependency'):
        upstream.preprocess([tmp_path / 'missing.png'])


def test_frozen_runtime_verifies_commit_checkpoint_config_environment(tmp_path):
    runtime = _runtime(tmp_path, mast3r=True)
    evidence = verify_frozen_runtime('MASt3R', runtime, git_probe=_git, env_probe=_env)
    assert evidence.source_commit == '1' * 40
    assert evidence.config_sha256 == _sha(runtime.config)  # type: ignore[arg-type]
    assert evidence.dust3r_source_commit == '2' * 40
    from dataclasses import replace
    bad = replace(runtime, source_commit='9' * 40)
    with pytest.raises(AdapterError, match='source commit mismatch'):
        verify_frozen_runtime('MASt3R', bad, git_probe=_git, env_probe=_env)


def test_frozen_runtime_probe_carries_typing_extensions_identity(tmp_path):
    runtime = _runtime(tmp_path)
    seen = {}

    def probe(env_path: Path, typing_site: Path, typing_sha: str) -> tuple[str, str, str, str, str]:
        seen['typing_site'] = typing_site
        seen['typing_sha'] = typing_sha
        return '3.10.20', '2.3.1+cu121', '4.15.0', str(typing_site / 'typing_extensions.py'), str(typing_site / 'typing_extensions-4.15.0.dist-info')

    evidence = verify_frozen_runtime('VGGT', runtime, git_probe=_git, env_probe=probe)
    assert seen == {'typing_site': runtime.typing_extensions_site, 'typing_sha': runtime.typing_extensions_sha256}
    assert evidence.environment['typing_extensions_version'] == '4.15.0'
    assert evidence.environment['typing_extensions_sha256'] == runtime.typing_extensions_sha256
    assert evidence.environment['typing_extensions_dist_info'].endswith('typing_extensions-4.15.0.dist-info')


@pytest.mark.parametrize('dist_info', ['', 'outside/typing_extensions-4.15.0.dist-info'])
def test_frozen_runtime_rejects_wrong_typing_extensions_distribution_origin(tmp_path, dist_info):
    runtime = _runtime(tmp_path)

    def probe(env_path: Path, typing_site: Path, typing_sha: str) -> tuple[str, str, str, str, str]:
        return '3.10.20', '2.3.1+cu121', '4.15.0', str(typing_site / 'typing_extensions.py'), str(tmp_path / dist_info)

    with pytest.raises(AdapterError, match='dist-info origin'):
        verify_frozen_runtime('VGGT', runtime, git_probe=_git, env_probe=probe)


def test_frozen_runtime_rejects_typing_extensions_hash_mismatch_before_probe(tmp_path):
    runtime = _runtime(tmp_path)

    def forbidden_probe(*_args):
        raise AssertionError('env probe should not run after dependency hash mismatch')

    bad = replace(runtime, typing_extensions_sha256='0' * 64)
    with pytest.raises(AdapterError, match='typing_extensions.py SHA-256 mismatch'):
        verify_frozen_runtime('VGGT', bad, git_probe=_git, env_probe=forbidden_probe)


def test_rendered_png_digest_is_ordered_and_verifies_exact_bytes(tmp_path):
    views = _views(tmp_path)
    batch = RenderedPngBatch.from_views(views)
    swapped = RenderedPngBatch.from_views(list(reversed(views)))
    assert batch.ordered_digest != swapped.ordered_digest
    views[0].png_path.write_bytes(b'tamper')
    with pytest.raises(AdapterError, match='digest mismatch'):
        RenderedPngBatch.from_views(views)


def test_source_pixel_grid_maps_model_pixels_back_to_non_square_source():
    pixels = source_pixel_grid(3, 4, 1200, 1600)
    assert pixels.shape == (12, 2)
    np.testing.assert_allclose(pixels[0], [199.5, 199.5])
    np.testing.assert_allclose(pixels[-1], [1399.5, 999.5])


class FakeVGGT:
    def __init__(self):
        self.preprocess_paths = None

    def preprocess(self, paths):
        self.preprocess_paths = tuple(paths)
        return np.zeros((1, 8, 3, 2, 2)), 'pre-vggt'

    def infer(self, images):
        depth = np.ones((1, 8, 2, 2), dtype=float)
        conf = np.full((1, 8, 2, 2), 2.0, dtype=float)
        return {'depth': depth, 'depth_conf': conf, 'camera_c2w': np.repeat(np.eye(4)[None], 8, axis=0), 'intrinsics': np.repeat(np.eye(3)[None], 8, axis=0)}


def test_vggt_unprojects_depth_preserves_view_pixel_and_confidence(tmp_path):
    runtime = _runtime(tmp_path)
    adapter = VGGTAdapter(runtime, output_root=tmp_path / 'out', device='cpu', upstream=FakeVGGT(), git_probe=_git, env_probe=_env)
    pred = adapter.predict_sample(_manifest('VGGT', runtime), __import__('georeliab_mve.contracts').contracts.SampleKey.parse('dtu/test/scan001/views-0001/clean/0/0'), _views(tmp_path))
    assert not pred.invalid_prediction
    geom = np.load(_file_uri_path(pred.geometry_prediction_uri, 'geometry'))
    conf = np.load(_file_uri_path(pred.native_confidence_uri, 'confidence'))
    mask = np.load(_file_uri_path(pred.valid_mask_uri, 'mask'))
    assert geom['points_world'].shape == (32, 3)
    assert geom['pixel_xy'].shape == (32, 2)
    assert set(geom['view_id'].tolist()) == set(range(8))
    assert np.all(conf['raw_confidence'] == 2.0)
    assert mask['valid_mask'].all()
    assert vggt_risk_from_depth_conf(np.array([2.0, 1.0]))[0] < vggt_risk_from_depth_conf(np.array([2.0, 1.0]))[1]



class TensorLike:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.detached = False
        self.on_cpu = False

    def detach(self):
        self.detached = True
        return self

    def cpu(self):
        self.on_cpu = True
        return self

    def numpy(self):
        if not self.detached or not self.on_cpu:
            raise RuntimeError('tensor-like value was not detached and moved to cpu')
        return self.value

class FakeMASt3R:
    def __init__(self, *, forbidden=()):
        self.calls = []
        self.forbidden = forbidden

    def preprocess(self, paths):
        self.calls.append(('preprocess', tuple(paths)))
        return [{'img': i} for i in range(8)], 'pre-mast3r'

    def infer(self, paths, images):
        self.calls.append(('infer', {'n_paths': len(paths), 'n_images': len(images), 'clean_depth': False, 'matching_conf_thr': float('-inf'), 'symmetrize': True, 'lr1': 0.07, 'niter1': 300, 'lr2': 0.01, 'niter2': 300, 'shared_intrinsics': False, 'opt_depth': True}))
        pts = np.zeros((8, 2, 2, 3), dtype=float)
        pts[..., 2] = 1.0
        return {'pts3d': [TensorLike(item) for item in pts], 'conf': [TensorLike(item) for item in np.ones((8, 2, 2))], 'camera_c2w': np.repeat(np.eye(4)[None], 8, axis=0), 'intrinsics': np.repeat(np.eye(3)[None], 8, axis=0), 'raw_pairwise': {'kept': True}, 'forbidden_operations': self.forbidden}


class FakeMASt3RFlattenedDenseOutput(FakeMASt3R):
    def infer(self, paths, images):
        result = dict(super().infer(paths, images))
        result['pts3d'] = [
            TensorLike(item.value.reshape(-1, 3))
            for item in result['pts3d']
        ]
        return result


def test_mast3r_exact_alignment_contract_and_clean_depth_false(tmp_path):
    fake = FakeMASt3R()
    runtime = _runtime(tmp_path, mast3r=True)
    adapter = MASt3RAdapter(runtime, output_root=tmp_path / 'out', device='cpu', upstream=fake, git_probe=_git, env_probe=_env)
    views = _views(tmp_path)
    pred = adapter.predict_sample(_manifest('MASt3R', runtime, mast3r=True), __import__('georeliab_mve.contracts').contracts.SampleKey.parse('dtu/test/scan001/views-0001/clean/0/0'), views)
    assert not pred.invalid_prediction
    call = fake.calls[-1][1]
    assert call == {'n_paths': 8, 'n_images': 8, 'clean_depth': False, 'matching_conf_thr': float('-inf'), 'symmetrize': True, 'lr1': 0.07, 'niter1': 300, 'lr2': 0.01, 'niter2': 300, 'shared_intrinsics': False, 'opt_depth': True}
    geom = np.load(_file_uri_path(pred.geometry_prediction_uri, 'geometry'))
    metadata = json.loads(str(geom['metadata']))
    assert metadata['matching_conf_thr'] == 'disabled:-inf'
    assert metadata['rendered_png_sha256']['0'] == views[0].png_sha256
    assert metadata['raw_pairwise_trace_uri'].startswith('file:')
    assert len(metadata['raw_pairwise_trace_sha256']) == 64
    assert mast3r_risk_from_confidence(np.array([2.0, 0.5]))[0] < mast3r_risk_from_confidence(np.array([2.0, 0.5]))[1]


def test_mast3r_accepts_frozen_upstream_flattened_dense_points(tmp_path):
    runtime = _runtime(tmp_path, mast3r=True)
    adapter = MASt3RAdapter(
        runtime,
        output_root=tmp_path / 'out-flat-dense',
        device='cpu',
        upstream=FakeMASt3RFlattenedDenseOutput(),
        git_probe=_git,
        env_probe=_env,
    )

    pred = adapter.predict_sample(
        _manifest('MASt3R', runtime, mast3r=True),
        SampleKey.parse('dtu/test/scan001/views-0001/clean/0/0'),
        _views(tmp_path),
    )

    assert not pred.invalid_prediction
    geom = np.load(_file_uri_path(pred.geometry_prediction_uri, 'geometry'))
    assert geom['points_world'].shape == (32, 3)
    assert geom['pixel_xy'].shape == (32, 2)
    assert set(geom['view_id'].tolist()) == set(range(8))


def test_mast3r_forbidden_cleaning_thresholds_and_invalid_retention(tmp_path):
    runtime = _runtime(tmp_path, mast3r=True)
    adapter = MASt3RAdapter(runtime, output_root=tmp_path / 'out', device='cpu', upstream=FakeMASt3R(forbidden=('clean_pointcloud',)), git_probe=_git, env_probe=_env)
    pred = adapter.predict_sample(_manifest('MASt3R', runtime, mast3r=True), __import__('georeliab_mve.contracts').contracts.SampleKey.parse('dtu/test/scan001/views-0001/clean/0/0'), _views(tmp_path))
    assert pred.invalid_prediction
    geom = np.load(_file_uri_path(pred.geometry_prediction_uri, 'geometry'))
    assert geom['points_world'].shape == (0, 3)
    assert pred.payload_digests['geometry_prediction_uri'] == _sha(Path(_file_uri_path(pred.geometry_prediction_uri, 'geometry')))


def test_real_mast3r_upstream_disables_default_confidence_threshold(monkeypatch, tmp_path):
    calls = {}

    class Model:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls['from_pretrained'] = (path, kwargs)
            return cls()
        def eval(self): return self
        def to(self, device): return self

    class Scene:
        intrinsics = np.repeat(np.eye(3)[None], 8, axis=0)
        def get_dense_pts3d(self, *, clean_depth):
            calls['clean_depth'] = clean_depth
            return [TensorLike(np.zeros((1, 1, 3))) for _ in range(8)], None, [TensorLike(np.ones((1, 1))) for _ in range(8)]
        def get_im_poses(self):
            return np.repeat(np.eye(4)[None], 8, axis=0)

    def fake_import(name):
        if name == 'mast3r.model': return types.SimpleNamespace(AsymmetricMASt3R=Model)
        if name == 'mast3r.image_pairs': return types.SimpleNamespace(make_pairs=lambda *a, **kw: calls.setdefault('pairs', kw) or 'pairs')
        if name == 'mast3r.cloud_opt.sparse_ga':
            def align(*a, **kw):
                calls['alignment'] = kw
                return Scene()
            return types.SimpleNamespace(sparse_global_alignment=align)
        raise ImportError(name)

    monkeypatch.setattr('georeliab_mve.adapters.importlib.import_module', fake_import)
    runtime = _runtime(tmp_path, mast3r=True)
    cache_dir = tmp_path / 'cache'
    forward = cache_dir / 'forward' / 'a'
    forward.mkdir(parents=True)
    trace_file = forward / 'b.pth'
    trace_file.write_bytes(b'pairwise')
    upstream = RealMASt3RUpstream(runtime, device='cpu', cache_dir=cache_dir)
    result = upstream.infer([view.png_path for view in _views(tmp_path)], [{} for _ in range(8)])
    assert result['raw_pairwise_cache_files'][0]['sha256'] == _sha(trace_file)
    assert calls['from_pretrained'][0] == str(runtime.checkpoint.parent)
    assert calls['from_pretrained'][1] == {'local_files_only': True}
    assert calls['pairs']['scene_graph'] == 'complete'
    assert calls['pairs']['symmetrize'] is True
    assert calls['alignment']['matching_conf_thr'] == float('-inf')
    assert calls['alignment']['shared_intrinsics'] is False
    assert calls['alignment']['cache_path'] == str(cache_dir)
    assert calls['clean_depth'] is False


def test_real_vggt_upstream_loads_local_state_dict_without_network(monkeypatch, tmp_path):
    calls = {}

    class Model:
        def load_state_dict(self, state): calls['state'] = state
        def eval(self): calls['eval'] = True; return self
        def to(self, device): calls['device'] = device; return self

    class Torch:
        @staticmethod
        def load(path, **kwargs):
            calls['torch_load'] = (path, kwargs)
            return {'weights': 1}

    def fake_import(name):
        if name == 'torch': return Torch
        if name == 'vggt.models.vggt': return types.SimpleNamespace(VGGT=Model)
        raise ImportError(name)

    monkeypatch.setattr('georeliab_mve.adapters.importlib.import_module', fake_import)
    runtime = _runtime(tmp_path)
    upstream = RealVGGTUpstream(runtime, device='cpu')
    assert isinstance(upstream.load_model(), Model)
    assert calls['torch_load'] == (str(runtime.checkpoint), {'map_location': 'cpu', 'weights_only': True})
    assert calls['state'] == {'weights': 1}
    assert calls['eval'] is True
    assert calls['device'] == 'cpu'



def test_adapter_payload_view_id_uses_frozen_source_view_ids_not_local_indices(tmp_path):
    runtime = _runtime(tmp_path)
    views = _views(tmp_path)
    remapped = [RenderedView(20 + i, view.png_path, view.png_sha256, width=view.width, height=view.height, source_sha256=view.source_sha256) for i, view in enumerate(views)]
    adapter = VGGTAdapter(runtime, output_root=tmp_path / 'out-viewids', device='cpu', upstream=FakeVGGT(), git_probe=_git, env_probe=_env)
    pred = adapter.predict_sample(_manifest('VGGT', runtime), __import__('georeliab_mve.contracts').contracts.SampleKey.parse('dtu/test/scan001/views-0001/clean/0/0'), remapped)
    geom = np.load(_file_uri_path(pred.geometry_prediction_uri, 'geometry'))
    assert set(geom['view_id'].tolist()) == set(range(20, 28))




def test_real_mast3r_upstream_requires_explicit_cache_dir(tmp_path):
    with pytest.raises(AdapterError, match='explicit writable cache_dir'):
        RealMASt3RUpstream(_runtime(tmp_path, mast3r=True), device='cpu')








def test_adapter_rejects_manifest_provenance_not_bound_to_runtime(tmp_path):
    runtime = _runtime(tmp_path)
    bad_payload = _manifest('VGGT', runtime).to_dict()
    bad_payload['checkpoint_hash'] = '9' * 64
    bad_manifest = RunManifest.from_dict(bad_payload)
    adapter = VGGTAdapter(runtime, output_root=tmp_path / 'out-bound', device='cpu', upstream=FakeVGGT(), git_probe=_git, env_probe=_env)
    pred = adapter.predict_sample(bad_manifest, __import__('georeliab_mve.contracts').contracts.SampleKey.parse('dtu/test/scan001/views-0001/clean/0/0'), _views(tmp_path))
    assert pred.invalid_prediction
    geom = np.load(_file_uri_path(pred.geometry_prediction_uri, 'geometry'))
    metadata = json.loads(str(geom['metadata']))
    assert 'checkpoint_hash does not match' in metadata['failure_message']

class BadTensorLike:
    def detach(self):
        raise RuntimeError('cannot detach numeric trace')


class FakeVGGTInvalid(FakeVGGT):
    def infer(self, images):
        result = dict(super().infer(images))
        result['depth'] = result['depth'].copy()
        result['depth'][0, 0, 0, 0] = np.nan
        result['depth_conf'] = result['depth_conf'].copy()
        result['depth_conf'][0, 1, 0, 0] = np.inf
        result['intrinsics'] = result['intrinsics'].copy()
        result['intrinsics'][0] = 0.0
        return result


class FakeMASt3RInvalid(FakeMASt3R):
    def infer(self, paths, images):
        result = dict(super().infer(paths, images))
        pts = np.asarray([item.value for item in result['pts3d']])
        conf = np.asarray([item.value for item in result['conf']])
        pts[0, 0, 0, 0] = np.nan
        conf[1, 0, 0] = np.inf
        camera = result['camera_c2w'].copy()
        camera[0, :3, :3] = 0.0
        result['pts3d'] = [TensorLike(item) for item in pts]
        result['conf'] = [TensorLike(item) for item in conf]
        result['camera_c2w'] = camera
        return result


def test_vggt_nonfinite_and_singular_intrinsics_fail_closed_without_emptying_arrays(tmp_path):
    runtime = _runtime(tmp_path)
    adapter = VGGTAdapter(runtime, output_root=tmp_path / 'out-vggt-invalid', device='cpu', upstream=FakeVGGTInvalid(), git_probe=_git, env_probe=_env)
    pred = adapter.predict_sample(_manifest('VGGT', runtime), SampleKey.parse('dtu/test/scan001/views-0001/clean/0/0'), _views(tmp_path))
    assert pred.invalid_prediction
    geom = np.load(_file_uri_path(pred.geometry_prediction_uri, 'geometry'))
    conf = np.load(_file_uri_path(pred.native_confidence_uri, 'confidence'))
    mask = np.load(_file_uri_path(pred.valid_mask_uri, 'mask'))
    metadata = json.loads(str(geom['metadata']))
    assert geom['points_world'].shape == (32, 3)
    assert conf['raw_confidence'].shape == (32,)
    assert not mask['valid_mask'].all()
    assert 'non-finite' in metadata['failure_message']
    assert 'singular' in metadata['failure_message']


def test_mast3r_nonfinite_and_degenerate_camera_fail_closed_without_emptying_arrays(tmp_path):
    runtime = _runtime(tmp_path, mast3r=True)
    adapter = MASt3RAdapter(runtime, output_root=tmp_path / 'out-mast3r-invalid', device='cpu', upstream=FakeMASt3RInvalid(), git_probe=_git, env_probe=_env)
    pred = adapter.predict_sample(_manifest('MASt3R', runtime, mast3r=True), SampleKey.parse('dtu/test/scan001/views-0001/clean/0/0'), _views(tmp_path))
    assert pred.invalid_prediction
    geom = np.load(_file_uri_path(pred.geometry_prediction_uri, 'geometry'))
    conf = np.load(_file_uri_path(pred.native_confidence_uri, 'confidence'))
    mask = np.load(_file_uri_path(pred.valid_mask_uri, 'mask'))
    metadata = json.loads(str(geom['metadata']))
    assert geom['points_world'].shape == (32, 3)
    assert conf['raw_confidence'].shape == (32,)
    assert not mask['valid_mask'].all()
    assert 'non-finite' in metadata['failure_message']
    assert 'degenerate' in metadata['failure_message']


def test_mast3r_preflight_requires_dust3r_and_croco_sources_before_inference(tmp_path):
    full_runtime = _runtime(tmp_path, mast3r=True)
    runtime = replace(full_runtime, dust3r_source=None, dust3r_source_commit=None)
    adapter = MASt3RAdapter(runtime, output_root=tmp_path / 'out-preflight', device='cpu', upstream=FakeMASt3R(), git_probe=_git, env_probe=_env)
    pred = adapter.predict_sample(_manifest('MASt3R', full_runtime, mast3r=True), SampleKey.parse('dtu/test/scan001/views-0001/clean/0/0'), _views(tmp_path))
    assert pred.invalid_prediction
    assert not adapter.upstream.calls
    geom = np.load(_file_uri_path(pred.geometry_prediction_uri, 'geometry'))
    metadata = json.loads(str(geom['metadata']))
    assert 'requires frozen DUSt3R and CroCo' in metadata['failure_message']


def test_mast3r_pairwise_numeric_trace_failure_marks_invalid_but_preserves_output(tmp_path):
    class FakePairwiseFailure(FakeMASt3R):
        def infer(self, paths, images):
            result = dict(super().infer(paths, images))
            result['pairwise_pts3d'] = BadTensorLike()
            return result

    runtime = _runtime(tmp_path, mast3r=True)
    adapter = MASt3RAdapter(runtime, output_root=tmp_path / 'out-pairwise-fail', device='cpu', upstream=FakePairwiseFailure(), git_probe=_git, env_probe=_env)
    pred = adapter.predict_sample(_manifest('MASt3R', runtime, mast3r=True), SampleKey.parse('dtu/test/scan001/views-0001/clean/0/0'), _views(tmp_path))
    assert pred.invalid_prediction
    geom = np.load(_file_uri_path(pred.geometry_prediction_uri, 'geometry'))
    metadata = json.loads(str(geom['metadata']))
    assert geom['points_world'].shape == (32, 3)
    assert metadata['failure_type'] == 'AdapterPairwiseTraceError'
    assert metadata['raw_pairwise_trace_status'] == 'numeric-trace-conversion-failed'


def test_mast3r_raw_pairwise_nonnumeric_key_is_recorded_as_skipped(tmp_path):
    class FakeRawSkip(FakeMASt3R):
        def infer(self, paths, images):
            result = dict(super().infer(paths, images))
            result['raw_pairwise'] = {'bad': object()}
            return result

    runtime = _runtime(tmp_path, mast3r=True)
    adapter = MASt3RAdapter(runtime, output_root=tmp_path / 'out-raw-skip', device='cpu', upstream=FakeRawSkip(), git_probe=_git, env_probe=_env)
    pred = adapter.predict_sample(_manifest('MASt3R', runtime, mast3r=True), SampleKey.parse('dtu/test/scan001/views-0001/clean/0/0'), _views(tmp_path))
    assert not pred.invalid_prediction
    geom = np.load(_file_uri_path(pred.geometry_prediction_uri, 'geometry'))
    metadata = json.loads(str(geom['metadata']))
    assert metadata['raw_pairwise_skipped'] == [{'exception_type': 'ObjectDTypeUnsupported', 'key': 'bad'}]


def test_v11_payload_npz_bytes_are_deterministic_across_dirs_and_prefixes(tmp_path):
    runtime = _runtime(tmp_path)
    manifest = _manifest('VGGT', runtime)
    sample_key = SampleKey.parse('dtu/test/scan001/views-0001/clean/0/0')
    output = AdapterOutput(
        points_world=np.arange(6, dtype=float).reshape(2, 3),
        camera_c2w=np.eye(4)[None],
        intrinsics=np.eye(3)[None],
        pixel_xy=np.array([[0.5, 1.5], [2.5, 3.5]]),
        view_id=np.array([20, 20]),
        raw_confidence=np.array([2.0, 3.0]),
        valid_mask=np.array([True, True]),
        metadata={'b': 2, 'a': 1},
    )
    first = serialize_prediction_output(manifest=manifest, sample_key=sample_key, output=output, output_dir=tmp_path / 'a', prefix='first', runtime_seconds=1.0, peak_memory_mb=2.0, invalid_prediction=False, write_prediction_json=False)
    second = serialize_prediction_output(manifest=manifest, sample_key=sample_key, output=output, output_dir=tmp_path / 'b', prefix='second', runtime_seconds=9.0, peak_memory_mb=8.0, invalid_prediction=False, write_prediction_json=False)
    assert first.payload_digests == second.payload_digests
    assert Path(_file_uri_path(first.geometry_prediction_uri, 'geometry')).read_bytes() == Path(_file_uri_path(second.geometry_prediction_uri, 'geometry')).read_bytes()
    assert Path(_file_uri_path(first.native_confidence_uri, 'confidence')).read_bytes() == Path(_file_uri_path(second.native_confidence_uri, 'confidence')).read_bytes()
    assert Path(_file_uri_path(first.valid_mask_uri, 'mask')).read_bytes() == Path(_file_uri_path(second.valid_mask_uri, 'mask')).read_bytes()
