"""Fail-closed validation of external models, checkpoints, and datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Iterable

from .contracts import RunMode


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    name: str
    kind: str
    local_path: str | None
    required: bool = True
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ResourceCheck:
    name: str
    kind: str
    ready: bool
    reason: str
    local_path: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResourceGroupRequirement:
    name: str
    members: tuple[str, ...]
    minimum_ready: int

    def __post_init__(self) -> None:
        if not self.name or not self.members:
            raise ValueError('resource group name and members must be non-empty')
        if len(set(self.members)) != len(self.members):
            raise ValueError('resource group members must be unique')
        if not 1 <= self.minimum_ready <= len(self.members):
            raise ValueError('minimum_ready must be within the member count')


@dataclass(frozen=True, slots=True)
class ResourceGroupCheck:
    name: str
    ready: bool
    minimum_ready: int
    ready_members: tuple[str, ...]
    missing_members: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    ready: bool
    mode: RunMode
    checks: tuple[ResourceCheck, ...]
    group_checks: tuple[ResourceGroupCheck, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "mode": self.mode.value,
            "checks": [check.to_dict() for check in self.checks],
            'group_checks': [check.to_dict() for check in self.group_checks],
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assess_readiness(
    resources: Iterable[ResourceSpec],
    *,
    mode: RunMode,
    requirements: Iterable[ResourceGroupRequirement] = (),
) -> ReadinessReport:
    """Check all required resources.

    Fixture mode is runnable but explicitly not scientifically ready. Real mode
    blocks on every missing path or checkpoint hash mismatch.
    """

    resource_list = tuple(resources)
    requirement_list = tuple(requirements)
    checks: list[ResourceCheck] = []
    for resource in resource_list:
        if mode is RunMode.FIXTURE:
            checks.append(
                ResourceCheck(
                    name=resource.name,
                    kind=resource.kind,
                    ready=False,
                    reason="fixture mode is NON_SCIENTIFIC",
                    local_path=resource.local_path,
                )
            )
            continue
        if not resource.local_path:
            checks.append(
                ResourceCheck(
                    name=resource.name,
                    kind=resource.kind,
                    ready=False,
                    reason=(
                        "required local_path is not configured"
                        if resource.required
                        else "optional resource is not configured"
                    ),
                    local_path=None,
                )
            )
            continue
        path = Path(resource.local_path)
        if not path.exists():
            checks.append(
                ResourceCheck(
                    name=resource.name,
                    kind=resource.kind,
                    ready=False,
                    reason=(
                        "required path does not exist"
                        if resource.required
                        else "optional path does not exist"
                    ),
                    local_path=str(path),
                )
            )
            continue
        if resource.sha256:
            if not path.is_file():
                checks.append(
                    ResourceCheck(
                        name=resource.name,
                        kind=resource.kind,
                        ready=False,
                        reason="sha256 can only be verified for a file",
                        local_path=str(path),
                    )
                )
                continue
            actual = _sha256(path)
            if actual != resource.sha256:
                checks.append(
                    ResourceCheck(
                        name=resource.name,
                        kind=resource.kind,
                        ready=False,
                        reason=f"sha256 mismatch: expected {resource.sha256}, got {actual}",
                        local_path=str(path),
                    )
                )
                continue
        checks.append(
            ResourceCheck(
                name=resource.name,
                kind=resource.kind,
                ready=True,
                reason="verified",
                local_path=str(path),
            )
        )

    required_names = {
        resource.name for resource in resource_list if resource.required
    }
    required_checks = [check for check in checks if check.name in required_names]
    checks_by_name = {check.name: check for check in checks}
    group_checks: list[ResourceGroupCheck] = []
    for requirement in requirement_list:
        unknown = set(requirement.members) - set(checks_by_name)
        if unknown:
            raise ValueError(
                f'resource group {requirement.name!r} has unknown members: '
                + ', '.join(sorted(unknown))
            )
        ready_members = tuple(
            member
            for member in requirement.members
            if checks_by_name[member].ready
        )
        missing_members = tuple(
            member for member in requirement.members if member not in ready_members
        )
        group_ready = (
            mode is RunMode.REAL
            and len(ready_members) >= requirement.minimum_ready
        )
        group_checks.append(
            ResourceGroupCheck(
                name=requirement.name,
                ready=group_ready,
                minimum_ready=requirement.minimum_ready,
                ready_members=ready_members,
                missing_members=missing_members,
                reason=(
                    'minimum candidate count verified'
                    if group_ready
                    else f'requires {requirement.minimum_ready} of '
                    f'{len(requirement.members)} candidates'
                ),
            )
        )
    ready = (
        mode is RunMode.REAL
        and bool(required_checks or group_checks)
        and all(check.ready for check in required_checks)
        and all(check.ready for check in group_checks)
    )
    return ReadinessReport(
        ready=ready,
        mode=mode,
        checks=tuple(checks),
        group_checks=tuple(group_checks),
    )
