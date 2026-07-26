"""Scene-disjoint split validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


REQUIRED_SPLITS = ("dev", "reference-token", "calibration", "test")


class LeakageError(ValueError):
    """Raised when scene identities leak across protocol splits."""


@dataclass(frozen=True, slots=True)
class SplitValidation:
    scene_counts: dict[str, int]
    total_scenes: int


def validate_scene_disjoint(
    splits: Mapping[str, Iterable[str]],
    *,
    required_splits: tuple[str, ...] = REQUIRED_SPLITS,
) -> SplitValidation:
    """Reject missing, duplicate, empty, or overlapping scene partitions."""

    missing = sorted(set(required_splits) - set(splits))
    if missing:
        raise LeakageError(f"missing required splits: {', '.join(missing)}")

    normalized: dict[str, set[str]] = {}
    for split_name in required_splits:
        raw_scenes = list(splits[split_name])
        if not raw_scenes:
            raise LeakageError(f"split {split_name!r} is empty")
        if any(not isinstance(scene, str) or not scene for scene in raw_scenes):
            raise LeakageError(f"split {split_name!r} has an invalid scene id")
        if len(set(raw_scenes)) != len(raw_scenes):
            raise LeakageError(f"split {split_name!r} contains duplicate scene ids")
        normalized[split_name] = set(raw_scenes)

    overlap_messages: list[str] = []
    split_names = list(required_splits)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = sorted(normalized[left] & normalized[right])
            if overlap:
                overlap_messages.append(
                    f"{left}<->{right}: {', '.join(overlap[:5])}"
                )
    if overlap_messages:
        raise LeakageError(
            "scene leakage detected across splits: " + "; ".join(overlap_messages)
        )

    counts = {name: len(normalized[name]) for name in required_splits}
    return SplitValidation(scene_counts=counts, total_scenes=sum(counts.values()))

