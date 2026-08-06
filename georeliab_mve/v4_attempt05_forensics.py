
"""Identity-only Attempt-05 forensic CLI.

The command writes reports under a new forensic root and never mutates the
frozen run, artifact, log, or ledger roots. It prints counts and hashes only;
scientific metric payloads are never parsed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .v4_attempt05_recovery import (
    ScheduleIdentityManifest,
    identity_only_audit,
    sha256_file,
    write_forensic_bundle,
)


def replay_schedule_hashes(path: Path) -> dict[str, object]:
    """Replay raw, semantic and identity hashes in their separate domains."""

    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("schedule JSON must contain an object")
    embedded = payload.get("schedule_sha256")
    unsigned = dict(payload)
    for name in (
        "schedule_sha256",
        "schedule_raw_sha256",
        "schedule_semantic_sha256",
        "schedule_identity_sha256",
    ):
        unsigned.pop(name, None)
    keys = _schedule_keys(path)
    manifest = ScheduleIdentityManifest.build(
        raw_sha256=sha256_file(path),
        semantic_payload=unsigned,
        ordered_unit_ids=keys,
    )
    return {
        "schedule_raw_sha256": manifest.raw_sha256,
        "schedule_semantic_sha256": manifest.semantic_sha256,
        "schedule_identity_sha256": manifest.schedule_identity_sha256,
        "ordered_unit_ids_sha256": manifest.ordered_unit_ids_sha256,
        "unit_count": manifest.unit_count,
        "embedded_schedule_sha256": embedded if isinstance(embedded, str) else None,
        "schedule_identity_manifest": manifest.to_dict(),
        # Legacy name retained for read-only reports; it is explicitly raw.
        "schedule_file_sha256": manifest.raw_sha256,
    }



def _schedule_keys(path: Path) -> list[object]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for name in ("schedule", "units", "scientific_schedule"):
            if name in payload:
                payload = payload[name]
                break
    if not isinstance(payload, list):
        raise ValueError("schedule JSON must be a list or contain schedule/units")
    keys: list[object] = []
    for item in payload:
        if isinstance(item, dict):
            keys.append((item["model_id"], item["scene_id"], item["state_id"]))
        elif isinstance(item, (list, tuple, str)):
            keys.append(item)
        else:
            raise ValueError("schedule item has no unit identity")
    return keys


def _sources(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            raise ValueError("--source requires label=path")
        result[label] = Path(path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", default="attempt-05")
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--forensic-root", type=Path, required=True)
    parser.add_argument("--schedule-sha256", required=True)
    parser.add_argument("--expected-schedule-sha256")
    parser.add_argument("--input-closure-schedule-sha256")
    parser.add_argument("--expected-schedule-identity-sha256")
    parser.add_argument("--expected-schedule-semantic-sha256")
    parser.add_argument("--expected-verified-count", type=int, default=199)
    parser.add_argument(
        "--require-binding-evidence",
        action="store_true",
        help="require adapter/config/split/corruption/GPU/tooling identity bindings",
    )
    parser.add_argument("--final-evidence-root", type=Path)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--postmortem-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    keys = _schedule_keys(args.schedule)
    hash_replay = replay_schedule_hashes(args.schedule)
    if args.schedule_sha256 != hash_replay["schedule_raw_sha256"]:
        raise ValueError("--schedule-sha256 must match the schedule raw file SHA-256")
    eligibility = identity_only_audit(
        attempt_id=args.attempt_id,
        schedule_keys=keys,
        ledger_path=args.ledger,
        run_root=args.run_root,
        schedule_sha256=args.schedule_sha256,
        expected_schedule_sha256=args.expected_schedule_sha256,
        input_closure_schedule_sha256=args.input_closure_schedule_sha256,
        schedule_identity_manifest=hash_replay["schedule_identity_manifest"],
        expected_schedule_identity_sha256=args.expected_schedule_identity_sha256,
        expected_schedule_semantic_sha256=args.expected_schedule_semantic_sha256,
        final_evidence_root=args.final_evidence_root,
        expected_verified_count=args.expected_verified_count,
        require_binding_evidence=args.require_binding_evidence,
    )
    postmortem: dict[str, object] = {
        "root_cause": "V4_ATTEMPT05_ROOT_CAUSE_UNOBSERVABLE_DUE_TO_EXCEPTION_COLLAPSE",
        "schedule_sha256": args.schedule_sha256,
        "expected_schedule_sha256": args.expected_schedule_sha256,
        "input_closure_schedule_sha256": args.input_closure_schedule_sha256,
        "binding_evidence_required": args.require_binding_evidence,
        "source_paths": {key: str(path) for key, path in _sources(args.source).items()},
        "schedule_hash_replay": hash_replay,
        "schedule_identity_manifest": hash_replay["schedule_identity_manifest"],
    }
    if args.postmortem_json is not None:
        loaded = json.loads(args.postmortem_json.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("--postmortem-json must contain an object")
        postmortem.update(loaded)
    outputs = write_forensic_bundle(
        args.forensic_root,
        source_paths=_sources(args.source),
        eligibility=eligibility,
        postmortem=postmortem,
    )
    print(
        json.dumps(
            {
                "verdict": eligibility.verdict,
                "corpus_classification": eligibility.corpus_classification,
                "verified_count": len(eligibility.verified_unit_keys),
                "missing_count": len(eligibility.missing_unit_keys),
                "output_paths": {name: str(path) for name, path in outputs.items()},
                "scientific_result": "NO_SCIENTIFIC_RESULT",
                "schedule_hash_replay": hash_replay,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
