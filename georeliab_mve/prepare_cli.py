'''CLI plumbing for the fail-closed DTU/TartanAir preparation workflow.'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .preparation import A100Overlay, PreparationError


PREPARE_OPERATIONS = ('verify', 'download', 'index', 'manifests', 'calibration', 'rendering', 'sanity')


def run_prepare_operation(
    *,
    operation: str,
    data_root: Path,
    state_path: Path,
    dry_run: bool,
    overlay_path: Path | None,
) -> dict[str, Any]:
    '''Record an explicit preparation state without declaring missing data ready.

    Resource acquisition is intentionally not implicit: ``download`` only becomes
    stateful when an operator has supplied an A100 overlay and invokes the
    corresponding resource-specific helper.  This prevents a dry run/index-only
    command from becoming accidental scientific readiness evidence.
    '''

    if operation not in PREPARE_OPERATIONS:
        raise PreparationError(f'unsupported prepare operation: {operation}')
    overlay = A100Overlay.load(overlay_path) if overlay_path else None
    resources_ready = False
    if operation == 'verify' and not dry_run:
        # Verification requires the complete archive/extracted-layout routines;
        # this conservative state is upgraded only by their successful caller.
        resources_ready = False
    payload = {
        'schema_version': 'preparation-state-v1',
        'operation': operation,
        'data_root': str(data_root),
        'dry_run': dry_run,
        'overlay': str(overlay.source) if overlay else None,
        'runtime_root': overlay.runtime_root if overlay else None,
        'resources_ready': resources_ready,
        'scientific_ready': False,
        'notice': 'Preparation state only; not scientific evidence and not resource readiness.',
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    partial = state_path.with_suffix(state_path.suffix + '.partial')
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    partial.replace(state_path)
    return payload
