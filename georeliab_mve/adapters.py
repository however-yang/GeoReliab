'''External-model boundaries; implementations live beside upstream checkouts.'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .contracts import PredictionArtifact, RunManifest, SampleKey


@dataclass(frozen=True, slots=True)
class GeometryIntervention:
    name: str
    source_scene: str | None = None
    hook_location: str | None = None


@dataclass(frozen=True, slots=True)
class CorruptionCondition:
    name: str
    severity: int


@runtime_checkable
class GeometryModelAdapter(Protocol):
    '''Adapter must keep RGB, prompt, decoder, and seed fixed across arms.'''

    @property
    def model_name(self) -> str: ...

    def reproducible(self) -> bool: ...

    def hook_locations(self) -> tuple[str, ...]: ...

    def predict(
        self,
        manifest: RunManifest,
        sample_key: SampleKey,
        intervention: GeometryIntervention,
    ) -> PredictionArtifact: ...


@runtime_checkable
class GeoReliabModelAdapter(Protocol):
    '''Frozen GFM adapter exposing native confidence and geometry output.'''

    @property
    def model_name(self) -> str: ...

    def reproducible(self) -> bool: ...

    def predict(
        self,
        manifest: RunManifest,
        sample_key: SampleKey,
        condition: CorruptionCondition,
    ) -> PredictionArtifact: ...

