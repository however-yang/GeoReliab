"""Test-only Pilot partition-freeze and authorization audit harness.

This module is intentionally outside :mod:`georeliab_mve`.  The RED contract
suite defines the next development boundary here before any implementation is
allowed to prepare Pilot artifacts.  It must never dispatch a GPU, materialize
scientific inputs, or start a Pilot unit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping


SCHEMA_VERSION = "georeliab-v4-pilot-freeze-authorization-audit-1.0"
READY_STATUS = "V4_PILOT_PARTITION_AND_AUTHORIZATION_READY"
DEVELOPMENT_EVIDENCE_ONLY = "DEVELOPMENT_EVIDENCE_ONLY"
NO_SCIENTIFIC_RESULT = "NO_SCIENTIFIC_RESULT"


class PilotFreezeAuthorizationError(ValueError):
    """Raised when the test-only Pilot preparation chain fails closed."""


def prepare_pilot_partition_authorization(
    *,
    admission_report_path: Path,
    approval_request_path: Path,
    output_root: Path,
    home_root: Path,
) -> dict[str, object]:
    """Prepare a sealed, non-executing Pilot partition/authorization bundle."""

    raise NotImplementedError("RED: Pilot freeze/authorization harness is absent")


def validate_pilot_partition_manifest(
    payload: Mapping[str, object],
    *,
    expected_schedule_identity_sha256: str,
    expected_protocol_sha256: str,
    expected_model_bindings_sha256: str,
) -> dict[str, object]:
    """Validate the complete frozen Pilot partition identity."""

    raise NotImplementedError("RED: Pilot partition validation is absent")


def verify_pilot_freeze_bundle(root: Path) -> dict[str, object]:
    """Verify exact-file coverage and hashes for a prepared freeze bundle."""

    raise NotImplementedError("RED: Pilot freeze-bundle verification is absent")
