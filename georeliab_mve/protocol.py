'''Load and validate the frozen dual-MVE protocol configuration.'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from . import toml_compat as tomllib

from .gates import (
    GEOMETRY_DELTA_THRESHOLD,
    GEOMETRY_EQUIVALENCE_MARGIN,
    GEOMETRY_RECOVERY_THRESHOLD,
    GEOMETRY_REQUIRED_BENCHMARKS,
    GEOMETRY_MODEL_CANDIDATES,
    GEORELIAB_FAILURE_AUROC_THRESHOLD,
    GEORELIAB_MVE_REQUIRED_MODELS,
    GEORELIAB_RHO_DECLINE_THRESHOLD,
    GEORELIAB_ZERO_UPDATE_GAIN_THRESHOLD,
    GEORELIAB_REQUIRED_CORRUPTIONS,
)
from .readiness import ResourceGroupRequirement, ResourceSpec

Self = TypeVar('Self')

GEOMETRY_CHECKPOINT_RESOURCES = frozenset(
    {
        'spatial-mllm-checkpoint',
        'spatialstack-checkpoint',
        'guide-checkpoint',
    }
)
GEORELIAB_CHECKPOINT_RESOURCES = frozenset(
    {'vggt-checkpoint', 'mast3r-checkpoint'}
)


class ProtocolError(ValueError):
    '''Raised when configuration drifts from the approved plan.'''


@dataclass(frozen=True, slots=True)
class ProtocolConfig:
    path: Path
    protocol_version: str
    internal_freeze: str
    resources: tuple[ResourceSpec, ...]
    resource_groups: tuple[ResourceGroupRequirement, ...]
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> Self:
        try:
            raw = tomllib.loads(path.read_text(encoding='utf-8'))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ProtocolError(f'cannot load protocol {path}: {exc}') from exc
        if raw.get('protocol_version') != '1.0':
            raise ProtocolError('protocol_version must be 1.0')
        internal_freeze = raw.get('internal_freeze')
        if internal_freeze != '2026-09-15':
            raise ProtocolError('internal_freeze must remain 2026-09-15')
        resources_raw = raw.get('resource')
        if not isinstance(resources_raw, list) or not resources_raw:
            raise ProtocolError('at least one [[resource]] entry is required')
        resources: list[ResourceSpec] = []
        for item in resources_raw:
            if not isinstance(item, dict):
                raise ProtocolError('resource entries must be tables')
            try:
                resources.append(
                    ResourceSpec(
                        name=item['name'],
                        kind=item['kind'],
                        local_path=item.get('local_path'),
                        required=item.get('required', True),
                        sha256=item.get('sha256'),
                    )
                )
            except KeyError as exc:
                raise ProtocolError(
                    f'resource missing field: {exc.args[0]}'
                ) from exc
        groups_raw = raw.get('resource_group')
        if not isinstance(groups_raw, list) or not groups_raw:
            raise ProtocolError('at least one [[resource_group]] entry is required')
        resource_groups: list[ResourceGroupRequirement] = []
        for item in groups_raw:
            if not isinstance(item, dict):
                raise ProtocolError('resource_group entries must be tables')
            try:
                members = item['members']
                if not isinstance(members, list) or any(
                    not isinstance(member, str) or not member for member in members
                ):
                    raise ProtocolError('resource_group members must be strings')
                resource_groups.append(
                    ResourceGroupRequirement(
                        name=item['name'],
                        members=tuple(members),
                        minimum_ready=item['minimum_ready'],
                    )
                )
            except KeyError as exc:
                raise ProtocolError(
                    f'resource_group missing field: {exc.args[0]}'
                ) from exc
        config = cls(
            path=path,
            protocol_version=raw['protocol_version'],
            internal_freeze=internal_freeze,
            resources=tuple(resources),
            resource_groups=tuple(resource_groups),
            raw=raw,
        )
        config.validate_thresholds()
        return config

    def validate_thresholds(self) -> None:
        geometry = self.raw.get('geometry_gate', {})
        georeliab = self.raw.get('georeliab_gate', {})
        expected = {
            'geometry_gate.delta_geom': (
                geometry.get('delta_geom'),
                GEOMETRY_DELTA_THRESHOLD,
            ),
            'geometry_gate.recovery': (
                geometry.get('recovery'),
                GEOMETRY_RECOVERY_THRESHOLD,
            ),
            'geometry_gate.equivalence_margin': (
                geometry.get('equivalence_margin'),
                GEOMETRY_EQUIVALENCE_MARGIN,
            ),
            'georeliab_gate.rho_decline': (
                georeliab.get('rho_decline'),
                GEORELIAB_RHO_DECLINE_THRESHOLD,
            ),
            'georeliab_gate.failure_auroc': (
                georeliab.get('failure_auroc'),
                GEORELIAB_FAILURE_AUROC_THRESHOLD,
            ),
            'georeliab_gate.zero_update_auroc_gain': (
                georeliab.get('zero_update_auroc_gain'),
                GEORELIAB_ZERO_UPDATE_GAIN_THRESHOLD,
            ),
        }
        drift = [
            f'{name}: configured={actual!r}, frozen={frozen!r}'
            for name, (actual, frozen) in expected.items()
            if actual != frozen
        ]
        if drift:
            raise ProtocolError('gate threshold drift: ' + '; '.join(drift))
        geometry_protocol = self.raw.get('geometry', {})
        configured_benchmarks = geometry_protocol.get('benchmarks')
        if (
            not isinstance(configured_benchmarks, list)
            or frozenset(configured_benchmarks) != GEOMETRY_REQUIRED_BENCHMARKS
        ):
            raise ProtocolError('Geometry benchmarks must remain VSI-Bench and CVT-Bench')
        configured_geometry_models = geometry_protocol.get('model_order')
        if (
            not isinstance(configured_geometry_models, list)
            or frozenset(configured_geometry_models) != GEOMETRY_MODEL_CANDIDATES
        ):
            raise ProtocolError(
                'Geometry model candidates must remain Spatial-MLLM/SpatialStack/GUIDE'
            )
        georeliab_protocol = self.raw.get('georeliab', {})
        configured_georeliab_models = georeliab_protocol.get('model_order')
        configured_corruptions = georeliab_protocol.get('corruptions')
        if (
            not isinstance(configured_corruptions, list)
            or frozenset(configured_corruptions) != GEORELIAB_REQUIRED_CORRUPTIONS
            or georeliab_protocol.get('severity') != [1, 2, 3]
        ):
            raise ProtocolError('GeoReliab must retain three corruptions at severities 1/2/3')
        groups = {group.name: group for group in self.resource_groups}
        if (
            not isinstance(configured_georeliab_models, list)
            or not GEORELIAB_MVE_REQUIRED_MODELS
            <= frozenset(configured_georeliab_models[:2])
        ):
            raise ProtocolError('GeoReliab MVE models must remain VGGT and MASt3R')
        geometry = groups.get('geometry-model-candidates')
        if (
            geometry is None
            or frozenset(geometry.members) != GEOMETRY_CHECKPOINT_RESOURCES
            or geometry.minimum_ready != 2
        ):
            raise ProtocolError('Geometry readiness must be an explicit 2-of-3 group')
        geo = groups.get('georeliab-mve-models')
        if (
            geo is None
            or frozenset(geo.members) != GEORELIAB_CHECKPOINT_RESOURCES
            or geo.minimum_ready != 2
        ):
            raise ProtocolError('GeoReliab MVE readiness must be a 2-of-2 group')
