'''Explicit, atomic success paths for every prepare-georeliab operation.'''
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .archive_round1 import download_archive, verify_archive
from .inventory_round1 import parse_dtu_inventory
from .preparation import (
    CALIBRATION_SCENES, A100Overlay, PreparationError, atomic_json,
    build_split_view_manifest, calibration_qa, calibrate_corruptions,
    deterministic_png, fog_render, load_prepared_records, low_light_noise_render,
    render_defocus, tartanair_native_fog_sanity,
)
from .tartanair_range import index_remote_zip

_REQUIRED_RESOURCES = (
    'dtu_sampleset_url', 'dtu_sampleset_bytes', 'dtu_sampleset_sha256',
    'dtu_points_url', 'dtu_points_bytes', 'dtu_points_sha256',
    'dtu_rectified_url', 'dtu_rectified_bytes', 'dtu_rectified_etag',
    'tartanair_image_url', 'tartanair_image_bytes', 'tartanair_image_etag',
    'tartanair_depth_url', 'tartanair_depth_bytes', 'tartanair_depth_etag',
    'tartanair_hf_commit', 'vggt_checkpoint', 'vggt_checkpoint_sha256',
    'vggt_source_commit', 'mast3r_checkpoint', 'mast3r_checkpoint_sha256',
    'mast3r_config', 'mast3r_config_sha256', 'mast3r_source_commit',
    'dust3r_source_commit', 'croco_source_commit',
)
_REQUIRED_RUNTIME = ('vggt_source', 'mast3r_source', 'vggt_env', 'mast3r_env', 'vggt_python', 'vggt_torch', 'mast3r_python', 'mast3r_torch')


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    atomic_json(path, payload)


def _normal_etag(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreparationError('remote archive evidence is missing an ETag')
    return value.strip().removeprefix('W/').strip().strip('"').lower()


def _overlay(overlay_path: Path | None) -> A100Overlay:
    if overlay_path is None:
        raise PreparationError('this non-dry preparation operation requires the frozen A100 overlay')
    overlay = A100Overlay.load(overlay_path)
    missing = [key for key in _REQUIRED_RESOURCES if key not in overlay.resources]
    missing += [key for key in _REQUIRED_RUNTIME if key not in overlay.execution and key not in _runtime_values(overlay)]
    if missing:
        raise PreparationError('overlay missing required frozen identities: ' + ', '.join(missing))
    return overlay


def _runtime_values(overlay: A100Overlay) -> dict[str, object]:
    # A100Overlay intentionally stores only root/resources/execution. Preserve
    # extra runtime identities from TOML for validation without exposing writes.
    import tomllib
    return tomllib.loads(overlay.source.read_text(encoding='utf-8')).get('runtime', {})


def _inventory_payload(scenes: tuple[Any, ...]) -> dict[str, Any]:
    return {'schema_version': 'dtu-official-inventory-v1', 'scenes': [
        {'scene_id': scene.scene_id, 'rgb_files': list(scene.rgb_files),
         'camera_centers': {str(key): list(map(float, value)) for key, value in scene.camera_centers.items()},
         'points_path': scene.points_path, 'mask_path': scene.mask_path}
        for scene in scenes
    ]}


def _read_inventory(path: Path) -> tuple[Any, ...]:
    from .preparation import DtuScene
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('schema_version') != 'dtu-official-inventory-v1':
            raise ValueError('schema')
        return tuple(DtuScene(int(row['scene_id']), tuple(row['rgb_files']),
            {int(key): __import__('numpy').asarray(value, dtype=float) for key, value in row['camera_centers'].items()},
            row['points_path'], row['mask_path']) for row in payload['scenes'])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PreparationError(f'verified inventory is unreadable: {exc}') from exc


def _remote_evidence(name: str, url: str, expected_bytes: int, expected_etag: str) -> dict[str, Any]:
    index = index_remote_zip(url)
    if index.content_length != expected_bytes or _normal_etag(index.etag) != _normal_etag(expected_etag):
        raise PreparationError(f'{name} remote identity mismatch')
    return {'name': name, 'url': url, 'bytes': index.content_length, 'etag': _normal_etag(index.etag),
            'central_directory_sha256': index.central_directory_sha256, 'member_count': len(index.entries)}


def _run_download(root: Path, overlay: A100Overlay) -> dict[str, Any]:
    resources = overlay.resources
    samples = download_archive(str(resources['dtu_sampleset_url']), root / 'SampleSet.zip',
        expected_bytes=int(resources['dtu_sampleset_bytes']), expected_sha256=str(resources['dtu_sampleset_sha256']))
    points = download_archive(str(resources['dtu_points_url']), root / 'Points.zip',
        expected_bytes=int(resources['dtu_points_bytes']), expected_sha256=str(resources['dtu_points_sha256']))
    remote = [
        _remote_evidence('Rectified.zip', str(resources['dtu_rectified_url']), int(resources['dtu_rectified_bytes']), str(resources['dtu_rectified_etag'])),
        _remote_evidence('tartanair-image', str(resources['tartanair_image_url']), int(resources['tartanair_image_bytes']), str(resources['tartanair_image_etag'])),
        _remote_evidence('tartanair-depth', str(resources['tartanair_depth_url']), int(resources['tartanair_depth_bytes']), str(resources['tartanair_depth_etag'])),
    ]
    evidence = {'schema_version': 'official-resource-evidence-v1', 'sample_set': verify_archive(samples, expected_bytes=int(resources['dtu_sampleset_bytes']), expected_sha256=str(resources['dtu_sampleset_sha256'])),
                'points': verify_archive(points, expected_bytes=int(resources['dtu_points_bytes']), expected_sha256=str(resources['dtu_points_sha256'])),
                'remote_indexes': remote}
    atomic_json(root / 'evidence' / 'official_resources.json', evidence)
    return evidence


def _run_verify(root: Path, overlay: A100Overlay) -> dict[str, Any]:
    resources = overlay.resources
    samples = verify_archive(root / 'SampleSet.zip', expected_bytes=int(resources['dtu_sampleset_bytes']), expected_sha256=str(resources['dtu_sampleset_sha256']))
    points = verify_archive(root / 'Points.zip', expected_bytes=int(resources['dtu_points_bytes']), expected_sha256=str(resources['dtu_points_sha256']))
    evidence_path = root / 'evidence' / 'official_resources.json'
    try:
        evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
        remote = evidence['remote_indexes']
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise PreparationError('verify requires canonical HTTP Range index evidence for Rectified and TartanAir') from exc
    required = {'Rectified.zip': (int(resources['dtu_rectified_bytes']), _normal_etag(resources['dtu_rectified_etag'])),
                'tartanair-image': (int(resources['tartanair_image_bytes']), _normal_etag(resources['tartanair_image_etag'])),
                'tartanair-depth': (int(resources['tartanair_depth_bytes']), _normal_etag(resources['tartanair_depth_etag']))}
    actual = {row.get('name'): (row.get('bytes'), _normal_etag(row.get('etag'))) for row in remote}
    if actual != required:
        raise PreparationError('verify remote index evidence does not match frozen overlay identities')
    return {'sample_set': samples, 'points': points, 'remote_indexes': remote}


def _run_calibration(root: Path) -> dict[str, Any]:
    records = load_prepared_records(root / 'prepared' / 'calibration_inputs.json', required_scenes=CALIBRATION_SCENES)
    calibration = calibrate_corruptions([row[4] for row in records], [row[3] for row in records])
    qa = calibration_qa(calibration, [(row[0], row[3], row[4], row[5], row[6]) for row in records])
    manifest = calibration.manifest()
    manifest.write(root / 'manifests' / 'corruption_calibration.json')
    atomic_json(root / 'manifests' / 'corruption_calibration_qa.json', qa)
    return {'parameter_manifest_sha256': manifest.sha256, 'qa_path': str(root / 'manifests' / 'corruption_calibration_qa.json'), 'qa_passed': True}


def _run_rendering(root: Path) -> dict[str, Any]:
    calibration_path = root / 'manifests' / 'corruption_calibration.json'
    qa_path = root / 'manifests' / 'corruption_calibration_qa.json'
    try:
        qa = json.loads(qa_path.read_text(encoding='utf-8'))
        if qa.get('passed') is not True:
            raise ValueError('QA did not pass')
        calibration_payload = json.loads(calibration_path.read_text(encoding='utf-8'))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise PreparationError('rendering requires a passing fail-closed calibration QA artifact') from exc
    from .preparation import CorruptionCalibration
    calibration = CorruptionCalibration(float(calibration_payload['d_ref']), tuple(calibration_payload['airlight']),
        tuple(calibration_payload['fog_betas']), tuple(calibration_payload['defocus_scales']), calibration_payload['implementation_version'])
    records = load_prepared_records(root / 'prepared' / 'render_inputs.json')
    outputs = []
    for scene, view, key, image, depth, raw_digest, gt_digest in records:
        for name, render in (('fog', lambda sev: fog_render(image, depth, calibration, severity=sev, gt_digest=gt_digest, raw_source_sha256=raw_digest)),
                             ('low-light-noise', lambda sev: low_light_noise_render(image, key, view, severity=sev, gt_digest=gt_digest, calibration=calibration, raw_source_sha256=raw_digest)),
                             ('defocus', lambda sev: render_defocus(image, depth, calibration, severity=sev, gt_digest=gt_digest, raw_source_sha256=raw_digest))):
            for severity in (1, 2, 3):
                rendered, metadata = render(severity)
                target = root / 'rendered' / f'scan{scene:03d}_view{view:03d}_{name}_s{severity}.png'
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix('.png.partial')
                partial.write_bytes(deterministic_png(rendered))
                partial.replace(target)
                if metadata['rendered_png_sha256'] != __import__('hashlib').sha256(target.read_bytes()).hexdigest():
                    raise PreparationError('atomic rendered PNG digest mismatch')
                metadata['actual_parameter_manifest_sha256'] = __import__('hashlib').sha256(calibration_path.read_bytes()).hexdigest()
                atomic_json(target.with_suffix('.json'), metadata)
                outputs.append(str(target))
    return {'rendered_count': len(outputs), 'rendered': outputs, 'parameter_manifest_sha256': __import__('hashlib').sha256(calibration_path.read_bytes()).hexdigest()}


def _run_sanity(root: Path) -> dict[str, Any]:
    qa_path = root / 'manifests' / 'corruption_calibration_qa.json'
    if not qa_path.is_file() or json.loads(qa_path.read_text(encoding='utf-8')).get('passed') is not True:
        raise PreparationError('sanity requires passing DTU calibration fog QA')
    try:
        pairs = json.loads((root / 'prepared' / 'tartanair_p000_pairs.json').read_text(encoding='utf-8'))['pairs']
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise PreparationError('sanity requires exactly 100 aligned TartanAir P000 pairs') from exc
    if len(pairs) != 100:
        raise PreparationError('sanity requires exactly 100 aligned TartanAir P000 pairs')
    frames = [( __import__('numpy').load(Path(row['rgb_npy']), allow_pickle=False), __import__('numpy').load(Path(row['depth_npy']), allow_pickle=False)) for row in pairs]
    result = tartanair_native_fog_sanity(frames)
    if not result.passed:
        raise PreparationError('TARTANAIR_NATIVE_FOG_SANITY failed')
    payload = {'reason_code': result.reason_code, 'passed': result.passed, 'negative_frames': result.negative_frames,
               'evaluated_frames': result.evaluated_frames, 'correlations': list(result.correlations)}
    atomic_json(root / 'evidence' / 'tartanair_native_fog_sanity.json', payload)
    return payload


def run_prepare_operation(*, operation: str, data_root: Path, state_path: Path, dry_run: bool, overlay_path: Path | None) -> dict[str, Any]:
    operations = ('verify', 'download', 'index', 'manifests', 'calibration', 'rendering', 'sanity')
    if operation not in operations:
        raise PreparationError(f'unsupported prepare operation: {operation}')
    base = {'schema_version': 'preparation-state-v3', 'operation': operation, 'data_root': str(data_root),
            'dry_run': dry_run, 'scientific_ready': False}
    if dry_run:
        payload = {**base, 'resources_ready': False, 'notice': 'Dry-run did not acquire, verify, or claim scientific readiness.'}
        _write_state(state_path, payload)
        return payload
    overlay = _overlay(overlay_path) if operation in ('verify', 'download') else None
    if operation == 'download':
        result = _run_download(data_root, overlay)
    elif operation == 'verify':
        result = _run_verify(data_root, overlay)
    elif operation == 'index':
        scenes = parse_dtu_inventory(data_root)
        inventory = _inventory_payload(scenes)
        atomic_json(data_root / 'manifests' / 'dtu_inventory.json', inventory)
        import hashlib
        result = {'inventory_path': str(data_root / 'manifests' / 'dtu_inventory.json'), 'inventory_sha256': hashlib.sha256((data_root / 'manifests' / 'dtu_inventory.json').read_bytes()).hexdigest(), 'scene_count': len(scenes)}
    elif operation == 'manifests':
        scenes = _read_inventory(data_root / 'manifests' / 'dtu_inventory.json')
        manifest = build_split_view_manifest(scenes)
        manifest.write(data_root / 'manifests' / 'split_view_manifest.json')
        result = {'split_view_manifest_sha256': manifest.sha256, 'manifest_path': str(data_root / 'manifests' / 'split_view_manifest.json')}
    elif operation == 'calibration':
        result = _run_calibration(data_root)
    elif operation == 'rendering':
        result = _run_rendering(data_root)
    else:
        result = _run_sanity(data_root)
    payload = {**base, **result, 'resources_ready': True, 'state_transition': f'{operation}:completed'}
    _write_state(state_path, payload)
    return payload