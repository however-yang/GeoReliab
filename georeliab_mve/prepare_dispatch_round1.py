'''Explicit, atomic success paths for every prepare-georeliab operation.'''
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .archive_round1 import download_archive, verify_archive
from .materialization import (
    build_dtu_archive_inventory,
    materialize_frozen_selection,
    sha256_file,
    validate_remote_indexes,
    verify_frozen_overlay_identities,
    verify_materialization_manifest,
)
from .prepared_inputs import write_prepared_inputs
from .preparation import (
    CALIBRATION_SCENES, A100Overlay, PreparationError, atomic_json,
    build_split_view_manifest, calibration_qa, calibrate_corruptions,
    deterministic_png, fog_render, load_prepared_batch,
    load_tartanair_prepared_pairs, low_light_noise_render, render_defocus,
    tartanair_native_fog_sanity,
)

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
    'typing_extensions_version', 'typing_extensions_sha256',
)
_REQUIRED_RUNTIME = (
    'vggt_source', 'mast3r_source', 'dust3r_source', 'croco_source',
    'vggt_env', 'mast3r_env', 'vggt_python', 'vggt_torch',
    'mast3r_python', 'mast3r_torch', 'typing_extensions_site',
)


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    atomic_json(path, payload)


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
    from . import toml_compat as tomllib
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


def _run_download(root: Path, overlay: A100Overlay) -> dict[str, Any]:
    resources = overlay.resources
    samples = download_archive(str(resources['dtu_sampleset_url']), root / 'SampleSet.zip',
        expected_bytes=int(resources['dtu_sampleset_bytes']), expected_sha256=str(resources['dtu_sampleset_sha256']))
    points = download_archive(str(resources['dtu_points_url']), root / 'Points.zip',
        expected_bytes=int(resources['dtu_points_bytes']), expected_sha256=str(resources['dtu_points_sha256']))
    remote, _ = validate_remote_indexes(resources)
    evidence = {'schema_version': 'official-resource-evidence-v3', 'sample_set': verify_archive(samples, expected_bytes=int(resources['dtu_sampleset_bytes']), expected_sha256=str(resources['dtu_sampleset_sha256'])),
                'points': verify_archive(points, expected_bytes=int(resources['dtu_points_bytes']), expected_sha256=str(resources['dtu_points_sha256'])),
                **remote}
    atomic_json(root / 'evidence' / 'official_resources.json', evidence)
    return evidence


def _run_verify(root: Path, overlay: A100Overlay) -> dict[str, Any]:
    resources = overlay.resources
    samples = verify_archive(root / 'SampleSet.zip', expected_bytes=int(resources['dtu_sampleset_bytes']), expected_sha256=str(resources['dtu_sampleset_sha256']))
    points = verify_archive(root / 'Points.zip', expected_bytes=int(resources['dtu_points_bytes']), expected_sha256=str(resources['dtu_points_sha256']))
    live_remote, _ = validate_remote_indexes(resources)
    identities = verify_frozen_overlay_identities(
        runtime=_runtime_values(overlay), resources=resources,
        cache_root=root / 'cache' / 'identity',
    )
    identity_path = root / 'evidence' / 'frozen_runtime_identity.json'
    atomic_json(identity_path, identities)
    evidence_path = root / 'evidence' / 'official_resources.json'
    payload = {'schema_version': 'official-resource-evidence-v3',
               'sample_set': samples, 'points': points, **live_remote,
               'runtime_identity_sha256': sha256_file(identity_path)}
    atomic_json(evidence_path, payload)
    return {**payload, 'identity_path': str(identity_path)}


def _run_calibration(root: Path) -> dict[str, Any]:
    batch = load_prepared_batch(
        root / 'prepared' / 'calibration_inputs.json',
        expected_stage='calibration',
    )
    records = batch.records
    if {row[0] for row in records} != set(CALIBRATION_SCENES):
        raise PreparationError('calibration inputs do not use the frozen calibration scenes')
    calibration = calibrate_corruptions(
        [row[4] for row in records], [row[3] for row in records],
    )
    qa = calibration_qa(
        calibration,
        [(row[0], row[1], row[3], row[4], row[5], row[6]) for row in records],
    )
    qa['split_view_manifest_sha256'] = batch.split_view_manifest_sha256
    qa['materialization_sha256'] = batch.materialization_sha256
    manifest = calibration.manifest()
    manifest.write(root / 'manifests' / 'corruption_calibration.json')
    atomic_json(root / 'manifests' / 'corruption_calibration_qa.json', qa)
    return {'parameter_manifest_sha256': manifest.sha256,
            'qa_path': str(root / 'manifests' / 'corruption_calibration_qa.json'),
            'qa_passed': True, 'calibration_record_count': len(records),
            'split_view_manifest_sha256': batch.split_view_manifest_sha256,
            'materialization_sha256': batch.materialization_sha256}


def _write_render_artifact(target: Path, png: bytes, metadata: dict[str, Any]) -> str:
    digest = hashlib.sha256(png).hexdigest()
    if metadata.get('rendered_png_sha256') != digest:
        raise PreparationError('renderer metadata digest does not match deterministic PNG')
    sidecar = target.with_suffix('.json')
    if target.exists():
        if target.read_bytes() != png:
            raise PreparationError(f'existing rendered artifact is tampered: {target}')
        if sidecar.exists():
            try:
                existing = json.loads(sidecar.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError) as exc:
                raise PreparationError(f'existing render metadata is unreadable: {sidecar}') from exc
            if existing != metadata:
                raise PreparationError(f'existing render metadata is not provenance-identical: {sidecar}')
        else:
            atomic_json(sidecar, metadata)
        return 'reused'
    if sidecar.exists():
        raise PreparationError(f'orphan rendered metadata refuses overwrite: {sidecar}')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.partial')
    partial.write_bytes(png)
    if hashlib.sha256(partial.read_bytes()).hexdigest() != digest:
        raise PreparationError('atomic rendered PNG digest mismatch')
    partial.replace(target)
    atomic_json(sidecar, metadata)
    return 'written'


def _run_rendering(root: Path, *, stage: str) -> dict[str, Any]:
    if stage not in ('smoke', 'test'):
        raise PreparationError('rendering requires explicit stage smoke or test')
    calibration_path = root / 'manifests' / 'corruption_calibration.json'
    qa_path = root / 'manifests' / 'corruption_calibration_qa.json'
    try:
        qa = json.loads(qa_path.read_text(encoding='utf-8'))
        if qa.get('passed') is not True:
            raise ValueError('QA did not pass')
        if qa.get('parameter_manifest_sha256') != __import__('hashlib').sha256(calibration_path.read_bytes()).hexdigest():
            raise ValueError('QA is not bound to the exact corruption calibration manifest')
        calibration_payload = json.loads(calibration_path.read_text(encoding='utf-8'))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise PreparationError('rendering requires a passing fail-closed calibration QA artifact') from exc
    from .preparation import CorruptionCalibration
    calibration = CorruptionCalibration(float(calibration_payload['d_ref']), tuple(calibration_payload['airlight']),
        tuple(calibration_payload['fog_betas']), tuple(calibration_payload['defocus_scales']), calibration_payload['implementation_version'])
    input_path = root / 'prepared' / f'render_inputs_{stage}.json'
    batch = load_prepared_batch(input_path, expected_stage=stage)
    if (
        qa.get('split_view_manifest_sha256') != batch.split_view_manifest_sha256
        or qa.get('materialization_sha256') != batch.materialization_sha256
    ):
        raise PreparationError(
            'rendering batch is not bound to the calibration QA manifests'
        )
    records = batch.records
    parameter_sha = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    lock_payload = {
        'schema_version': 'test-render-lock-v1',
        'stage': batch.stage,
        'split': batch.split,
        'split_view_manifest_sha256': batch.split_view_manifest_sha256,
        'materialization_sha256': batch.materialization_sha256,
        'parameter_manifest_sha256': parameter_sha,
        'calibration_qa_sha256': sha256_file(qa_path),
        'prepared_input_sha256': sha256_file(input_path),
    }
    if batch.stage == 'test':
        lock_path = root / 'manifests' / 'test_render_lock.json'
        if lock_path.exists():
            try:
                existing_lock = json.loads(lock_path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError) as exc:
                raise PreparationError('test render lock is unreadable') from exc
            if existing_lock != lock_payload:
                raise PreparationError('test rendering parameters or split changed after freeze')
        else:
            atomic_json(lock_path, lock_payload)
    outputs = []
    dispositions = {'written': 0, 'reused': 0}
    for scene, view, key, image, depth, raw_digest, gt_digest in records:
        # Clean is also a shared deterministic PNG condition for both later models.
        clean = root / 'rendered' / batch.stage / f'scan{scene:03d}_view{view:03d}_clean_s0.png'
        clean_png = deterministic_png(image)
        common = {'stage': batch.stage, 'split': batch.split,
                  'sample_key': key, 'scene_id': scene, 'view_id': view,
                  'raw_source_sha256': raw_digest, 'gt_digest': gt_digest,
                  'split_view_manifest_sha256': batch.split_view_manifest_sha256,
                  'materialization_sha256': batch.materialization_sha256,
                  'actual_parameter_manifest_sha256': parameter_sha,
                  'implementation_version': calibration.implementation_version}
        clean_metadata = {**common, 'corruption': 'clean', 'severity': 0,
                          'rendered_png_sha256': hashlib.sha256(clean_png).hexdigest()}
        dispositions[_write_render_artifact(clean, clean_png, clean_metadata)] += 1
        outputs.append(str(clean))
        for name, render in (('fog', lambda sev: fog_render(image, depth, calibration, severity=sev, gt_digest=gt_digest, raw_source_sha256=raw_digest)),
                             ('low-light-noise', lambda sev: low_light_noise_render(image, key, view, severity=sev, gt_digest=gt_digest, calibration=calibration, raw_source_sha256=raw_digest)),
                             ('defocus', lambda sev: render_defocus(image, depth, calibration, severity=sev, gt_digest=gt_digest, raw_source_sha256=raw_digest))):
            for severity in (1, 2, 3):
                rendered, metadata = render(severity)
                target = root / 'rendered' / batch.stage / f'scan{scene:03d}_view{view:03d}_{name}_s{severity}.png'
                metadata.update(common)
                png = deterministic_png(rendered)
                dispositions[_write_render_artifact(target, png, metadata)] += 1
                outputs.append(str(target))
    expected_count = len(records) * 10
    if len(outputs) != expected_count:
        raise PreparationError('rendering did not produce clean plus nine corruptions per scheduled view')
    return {'rendered_count': len(outputs), 'rendered': outputs,
            'stage': batch.stage, 'split': batch.split, **dispositions,
            'parameter_manifest_sha256': parameter_sha,
            'split_view_manifest_sha256': batch.split_view_manifest_sha256,
            'materialization_sha256': batch.materialization_sha256}


def _run_sanity(root: Path) -> dict[str, Any]:
    qa_path = root / 'manifests' / 'corruption_calibration_qa.json'
    calibration_path = root / 'manifests' / 'corruption_calibration.json'
    try:
        qa = json.loads(qa_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError('sanity requires passing DTU calibration fog QA') from exc
    if (
        qa.get('passed') is not True
        or qa.get('checks', {}).get('synthetic_fog') is not True
        or qa.get('parameter_manifest_sha256') != sha256_file(calibration_path)
    ):
        raise PreparationError('sanity requires exact passing DTU calibration fog QA')
    pairs = load_tartanair_prepared_pairs(
        root / 'prepared' / 'tartanair_p000_pairs.json'
    )
    frames = [(image, depth) for _, image, depth in pairs]
    result = tartanair_native_fog_sanity(frames)
    if not result.passed:
        raise PreparationError('TARTANAIR_NATIVE_FOG_SANITY failed')
    payload = {'reason_code': result.reason_code, 'passed': result.passed,
               'negative_frames': result.negative_frames,
               'evaluated_frames': result.evaluated_frames,
               'correlations': list(result.correlations),
               'prepared_input_sha256': sha256_file(root / 'prepared' / 'tartanair_p000_pairs.json'),
               'calibration_qa_sha256': sha256_file(qa_path)}
    atomic_json(root / 'evidence' / 'tartanair_native_fog_sanity.json', payload)
    return payload


def run_prepare_operation(*, operation: str, data_root: Path, state_path: Path,
                          dry_run: bool, overlay_path: Path | None,
                          stage: str | None = None) -> dict[str, Any]:
    operations = ('verify', 'download', 'index', 'manifests', 'prepared',
                  'calibration', 'rendering', 'sanity')
    if operation not in operations:
        raise PreparationError(f'unsupported prepare operation: {operation}')
    if operation == 'rendering' and stage not in ('smoke', 'test'):
        raise PreparationError('rendering requires explicit --stage smoke or test')
    if operation != 'rendering' and stage is not None:
        raise PreparationError('--stage is valid only for rendering')
    base = {'schema_version': 'preparation-state-v5', 'operation': operation,
            'stage': stage, 'data_root': str(data_root),
            'dry_run': dry_run, 'scientific_ready': False}
    if dry_run:
        payload = {**base, 'resources_ready': False, 'notice': 'Dry-run did not acquire, verify, or claim scientific readiness.'}
        _write_state(state_path, payload)
        return payload
    overlay = _overlay(overlay_path) if operation in (
        'verify', 'download', 'index', 'manifests', 'prepared',
    ) else None
    if overlay is not None and os.name != 'nt':
        if data_root.resolve() != Path(overlay.runtime_root).resolve():
            raise PreparationError(
                'non-dry A100 preparation data root must equal the frozen runtime root'
            )
    if operation == 'download':
        result = _run_download(data_root, overlay)
    elif operation == 'verify':
        result = _run_verify(data_root, overlay)
    elif operation == 'index':
        resources = overlay.resources
        verify_archive(data_root / 'SampleSet.zip',
            expected_bytes=int(resources['dtu_sampleset_bytes']),
            expected_sha256=str(resources['dtu_sampleset_sha256']))
        verify_archive(data_root / 'Points.zip',
            expected_bytes=int(resources['dtu_points_bytes']),
            expected_sha256=str(resources['dtu_points_sha256']))
        _, indexes = validate_remote_indexes(resources)
        scenes, provenance = build_dtu_archive_inventory(
            data_root / 'SampleSet.zip', data_root / 'Points.zip',
            indexes['Rectified.zip'],
        )
        inventory = _inventory_payload(scenes)
        atomic_json(data_root / 'manifests' / 'dtu_inventory.json', inventory)
        provenance_path = data_root / 'manifests' / 'dtu_inventory_provenance.json'
        atomic_json(provenance_path, provenance)
        result = {'inventory_path': str(data_root / 'manifests' / 'dtu_inventory.json'),
                  'inventory_sha256': sha256_file(data_root / 'manifests' / 'dtu_inventory.json'),
                  'inventory_provenance_path': str(provenance_path),
                  'inventory_provenance_sha256': sha256_file(provenance_path),
                  'scene_count': len(scenes)}
    elif operation == 'manifests':
        scenes = _read_inventory(data_root / 'manifests' / 'dtu_inventory.json')
        manifest = build_split_view_manifest(scenes)
        split_path = data_root / 'manifests' / 'split_view_manifest.json'
        manifest.write(split_path)
        materialization = materialize_frozen_selection(
            root=data_root, resources=overlay.resources,
            split_manifest_path=split_path,
            dtu_inventory_provenance_path=data_root / 'manifests' / 'dtu_inventory_provenance.json',
            typing_extensions_site=Path(str(_runtime_values(overlay)['typing_extensions_site'])),
        )
        verify_materialization_manifest(
            Path(materialization['materialization_path']),
            split_manifest_path=split_path,
        )
        result = {'split_view_manifest_sha256': manifest.sha256,
                  'manifest_path': str(split_path), **materialization}
    elif operation == 'prepared':
        result = write_prepared_inputs(data_root)
    elif operation == 'calibration':
        result = _run_calibration(data_root)
    elif operation == 'rendering':
        result = _run_rendering(data_root, stage=stage)
    else:
        result = _run_sanity(data_root)
    readiness = operation == 'verify'
    payload = {**result, **base, 'resources_ready': readiness,
               'state_transition': f'{operation}:completed'}
    if operation == 'download':
        payload['pending_reason'] = 'PENDING_RESOURCE_VERIFICATION'
    _write_state(state_path, payload)
    return payload
