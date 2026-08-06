'''Strict deployment-overlay validation for review round 1.'''

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from . import toml_compat as tomllib
from .preparation import A100Overlay as _BaseOverlay, PreparationError


_FORBIDDEN = frozenset({'geometry_gate', 'georeliab_gate', 'budgets', 'splits', 'geometry', 'georeliab', 'bootstrap_resamples', 'multiple_testing', 'rho_decline', 'failure_auroc', 'zero_update_auroc_gain'})
_TOP_LEVEL = frozenset({'runtime', 'resources', 'execution'})


def _walk(value: object, trail: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise PreparationError('A100 overlay keys must be strings')
            if key in _FORBIDDEN:
                raise PreparationError('A100 overlay must not override scientific threshold config: ' + '.'.join((*trail, key)))
            _walk(child, (*trail, key))


class A100Overlay(_BaseOverlay):
    @classmethod
    def load(cls, path: Path) -> 'A100Overlay':
        try:
            payload = tomllib.loads(path.read_text(encoding='utf-8'))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise PreparationError(f'cannot load A100 overlay: {exc}') from exc
        if set(payload) - _TOP_LEVEL:
            raise PreparationError('A100 overlay must not override scientific threshold config or add unsupported tables')
        _walk(payload)
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
