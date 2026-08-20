"""Test-only Pilot resource, input-closure, and CPU preflight harness.

The RED contracts intentionally define this module before implementation.  It
must never download data, load a model, dispatch a GPU, or start Pilot work.
"""

from __future__ import annotations

from pathlib import Path


RESOURCE_READY = "V4_PILOT_RESOURCE_CANDIDATE_READY"
INPUT_PREFLIGHT_READY = "V4_PILOT_INPUT_RESOURCE_PREFLIGHT_READY"
DEVELOPMENT_EVIDENCE_ONLY = "DEVELOPMENT_EVIDENCE_ONLY"
NO_SCIENTIFIC_RESULT = "NO_SCIENTIFIC_RESULT"


class PilotInputResourceError(ValueError):
    """Raised when a Pilot input/resource contract fails closed."""


def prepare_pilot_resource_candidate(
    *, request_path: Path, output_path: Path, home_root: Path
) -> dict[str, object]:
    raise NotImplementedError("RED: Pilot resource audit is absent")


def prepare_pilot_input_resource_preflight(
    *,
    freeze_root: Path,
    staged_inventory_path: Path,
    output_root: Path,
    home_root: Path,
    source_root: Path,
) -> dict[str, object]:
    raise NotImplementedError("RED: Pilot input/resource preflight is absent")


def verify_pilot_input_resource_bundle(root: Path) -> dict[str, object]:
    raise NotImplementedError("RED: Pilot input/resource bundle verifier is absent")
