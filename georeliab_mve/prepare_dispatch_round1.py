'''Real fail-closed prepare-georeliab operation dispatch.'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .preparation import A100Overlay, PreparationError, build_split_view_manifest, parse_dtu_inventory, verify_archive


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    partial.replace(path)


def run_prepare_operation(*, operation: str, data_root: Path, state_path: Path, dry_run: bool, overlay_path: Path | None) -> dict[str, Any]:
    if operation not in ('verify', 'download', 'index', 'manifests', 'calibration', 'rendering', 'sanity'):
        raise PreparationError(f'unsupported prepare operation: {operation}')
    overlay = A100Overlay.load(overlay_path) if overlay_path else None
    base = {'schema_version': 'preparation-state-v2', 'operation': operation, 'data_root': str(data_root), 'dry_run': dry_run,
            'overlay': str(overlay.source) if overlay else None, 'scientific_ready': False}
    if dry_run:
        payload = {**base, 'resources_ready': False, 'notice': 'Dry-run did not inspect, acquire, or claim readiness.'}
        _write_state(state_path, payload)
        return payload
    if operation == 'index':
        scenes = parse_dtu_inventory(data_root)
        manifest = build_split_view_manifest(scenes)
        manifest_path = data_root / 'manifests' / 'split_view_manifest.json'
        manifest.write(manifest_path)
        payload = {**base, 'resources_ready': True, 'index_manifest': str(manifest_path), 'split_view_manifest_sha256': manifest.sha256}
    elif operation == 'verify':
        archives = tuple(data_root / name for name in ('SampleSet.zip', 'Points.zip', 'Rectified.zip'))
        if not all(path.is_file() for path in archives):
            raise PreparationError('verify requires all complete official DTU archives')
        payload = {**base, 'resources_ready': True, 'archives': [verify_archive(path) for path in archives]}
    elif operation == 'download':
        if overlay is None:
            raise PreparationError('download requires an A100 overlay with official resource URLs')
        raise PreparationError('download requires explicit resource selection; no implicit multi-GB acquisition')
    else:
        prerequisite = data_root / 'manifests' / 'split_view_manifest.json'
        if not prerequisite.is_file():
            raise PreparationError(f'{operation} requires verified index manifest before work')
        raise PreparationError(f'{operation} requires its verified input artifact; refusing to synthesize scientific evidence')
    _write_state(state_path, payload)
    return payload
