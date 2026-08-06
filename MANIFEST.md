---
status: authoritative
authority: GEORELIAB_TASKBOOK.md
execution_entry: false
superseded_by: null
---

# GeoReliab v2.2 Manifest

This manifest is the authoritative navigation and file-identity index. Scientific and execution decisions are defined only by GEORELIAB_TASKBOOK.md. This v2.2 update is documentation-only: code, protocol TOML, CLI, data, budgets, and all Attempt-05 roots are unchanged.

## Current state

- Mainline: Reliability Awareness Evaluation Framework
- Main proposition: confidence ranking ability is not failure-warning ability.
- Current phase: Phase 0 — Infrastructure Qualification
- Terminal state: ATTEMPT05_TERMINAL_INFRASTRUCTURE_FAILURE
- Corpus state: PARTIAL_CORPUS_NOT_RESUMABLE
- Recovery state: RECOVERY_EXECUTOR_NOT_QUALIFIED
- Scientific state: NO_SCIENTIFIC_RESULT
- Automatic progression: disabled; no P3 or Warning-aware stage

## v2.2 document roles

| Path | Status | Authority | Execution entry | Role and rule |
|---|---|---|---|---|
| GEORELIAB_TASKBOOK.md | authoritative | GEORELIAB_TASKBOOK.md | true | Sole SSOT and only execution entry |
| idea-stage/NOVELTY_CLEARANCE.md | supporting | GEORELIAB_TASKBOOK.md | false | NC1-NC4 novelty boundary evidence |
| refine-logs/EXPERIMENT_PLAN.md | supporting | GEORELIAB_TASKBOOK.md | false | Claim-driven evidence and run-order specification |
| idea-stage/PILOT_PLAN.md | supporting | GEORELIAB_TASKBOOK.md | false | Fresh 3-scene, 60-unit development Pilot gates |
| refine-logs/EXPERIMENT_TRACKER.md | supporting | GEORELIAB_TASKBOOK.md | false | Status mirror only; never a protocol source |
| MANIFEST.md | authoritative | GEORELIAB_TASKBOOK.md | false | Navigation, version, role, and identity index |
| configs/georeliab_v4_protocol.toml | supporting | GEORELIAB_TASKBOOK.md | false | Frozen protocol asset; not modified by this documentation change |
| docs/GEORELIAB_V4_PROTOCOL.md | supporting | GEORELIAB_TASKBOOK.md | false | Existing protocol reference; not an execution entry |

All execution links in supporting documents must resolve to GEORELIAB_TASKBOOK.md. No supporting document may authorize a run.

## Historical documents

The following files are preserved for provenance and are not execution entries. Their original contents are intentionally retained.

| Path or family | Status | Superseded by | Rule |
|---|---|---|---|
| idea-stage/IDEA_REPORT.md | superseded_historical | GEORELIAB_TASKBOOK.md#v2.2 | Historical exploration and framing; cannot route execution |
| idea-stage/IDEA_REPORT_20260726_153000.md | superseded_historical | GEORELIAB_TASKBOOK.md#v2.2 | Timestamped snapshot; read-only history |
| idea-stage/IDEA_REPORT_20260726_170000.md | superseded_historical | GEORELIAB_TASKBOOK.md#v2.2 | Timestamped snapshot; read-only history |
| idea-stage/IDEA_REPORT_20260726_180000.md | superseded_historical | GEORELIAB_TASKBOOK.md#v2.2 | Timestamped snapshot; read-only history |
| refine-logs/EXPERIMENT_PLAN_20260726_190000.md | superseded_historical | GEORELIAB_TASKBOOK.md#v2.2 | Previous large-grid plan; cannot authorize execution |
| refine-logs/EXPERIMENT_TRACKER_20260726_190000.md | superseded_historical | GEORELIAB_TASKBOOK.md#v2.2 | Previous tracker snapshot; status only |
| idea-stage/IDEA_CANDIDATES.md | superseded_historical | GEORELIAB_TASKBOOK.md#v2.2 | Candidate-generation history |
| idea-stage/IDEA_CANDIDATES_V3.md | superseded_historical | GEORELIAB_TASKBOOK.md#v2.2 | Candidate-generation history |
| idea-stage/NOVELTY_REPORT.md | superseded_historical | GEORELIAB_TASKBOOK.md#v2.2 | Earlier novelty report |
| idea-stage/NOVELTY_REPORT_V2.md | superseded_historical | GEORELIAB_TASKBOOK.md#v2.2 | Earlier novelty report |
| idea-stage/NOVELTY_REPORT_V3.md | superseded_historical | GEORELIAB_TASKBOOK.md#v2.2 | Earlier novelty report |

Historical files may mention alternate ideas or old experiment grids; those mentions are not current routes.

Additional historical supporting files with the same non-entry status header:

- idea-stage/PROBLEM_FRAME.md
- idea-stage/LITERATURE_LANDSCAPE.md
- idea-stage/LITERATURE_LANDSCAPE_20260726_150000.md
- idea-stage/RESEARCH_REVIEW_REQUEST.md
- idea-stage/REVIEW_SUMMARY.md

## Frozen evidence and execution roots

Attempt-05 run, artifact, log, and ledger roots remain immutable and non-resumable. They are engineering postmortem material only. The 199/400 record cannot be used for statistics, Pilot decisions, confirmation, or paper claims.

Future Pilot and confirmation outputs must use independent run, artifact, ledger, and evidence roots. A cross-Attempt immutable union is rejected by the new runtime schema; only same-Attempt recovery sessions may be closed by a session-union manifest.

## Consistency rules

- Every current document carries a parseable status header with status, authority, execution_entry, and superseded_by.
- Every execution entry points to GEORELIAB_TASKBOOK.md.
- Recovery qualification precedes fresh Pilot; Pilot evidence precedes confirmation preparation; confirmation precedes method development.
- NO_SCIENTIFIC_RESULT remains true until the formal finalizer closes the approved scope.
- This manifest does not replace the Taskbook and does not authorize GPU, storage, or future-stage use.