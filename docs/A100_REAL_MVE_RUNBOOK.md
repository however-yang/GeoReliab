# A100 Real GeoReliab MVE Runbook

This runbook operates the frozen real GeoReliab dual-model MVE on the A100 host. It documents how to move from deployment readiness to an auditable GeoReliab PASS/FAIL decision. It does not claim that P0-P6 have already run.

Canonical operator path: use `scripts/a100/*.sh`. Manual CLI commands are included only to show what the scripts execute and for diagnosis when a script stops fail-closed.

## Scope and hard rules

- Runtime root: `/srv/private/smli/GeoReliab`
- Bare repository: `/srv/private/smli/GeoReliab/git/GeoReliab.git`
- Detached worktrees: `/srv/private/smli/GeoReliab/worktrees/REPLACE_WITH_FULL_PROJECT_COMMIT`
- Frozen overlay: `configs/a100_real_mve_overlay.toml`
- Scientific thresholds: `configs/dual_mve_protocol.toml` only
- Read-only reuse paths: `/home/smli/workspace`, `/home/smli/models`, `/home/smli/miniforge3/envs`, `/home/smli/miniforge3/pkgs/typing_extensions-4.15.0-pyhcf101f3_0/site-packages`

Do not write below `/home/smli`. Do not shrink P3/P5 schedules to fit budget. If the next frozen stage would exceed 50 GPU-hours or 1 TB, stop with `BLOCKED_RESOURCE_BUDGET`.

Current A100 facts to preserve during Task 7:

- frozen MASt3R `model.safetensors` and `config.json` exist and match the approved SHA-256 values;
- DTU `SampleSet.zip` and `Points.zip` are complete;
- Rectified DTU and TartanAir selected members are indexed/extracted through HTTP Range paths, so stale whole-archive `.partial` files are non-authoritative and must not be deleted or resumed automatically;
- `/srv` has about 1.7 TB free, while `/home` has about 2.1 GB free. Force caches, temporary files, logs, and outputs to `/srv/private/smli/GeoReliab`.
- `typing_extensions.py` is a frozen read-only runtime dependency: site `/home/smli/miniforge3/pkgs/typing_extensions-4.15.0-pyhcf101f3_0/site-packages`, version `4.15.0`, SHA-256 `433d11d170d3a24d2eb065ebc1bfe848cea7e3d7ce68567ab52bea2d4c2f7ed8`. It is injected ahead of model imports under `PYTHONNOUSERSITE=1`; `/home/smli/.local` must not be exposed.

## One-time shell setup

Run this in the source checkout that contains the implementation commit.

```bash
export ROOT=/srv/private/smli/GeoReliab
export COMMIT=REPLACE_WITH_FULL_PROJECT_COMMIT
export OVERLAY=configs/a100_real_mve_overlay.toml
export PYTHON=/home/smli/miniforge3/envs/vggt_env/bin/python

export TMPDIR=$ROOT/cache/tmp
export TMP=$ROOT/cache/tmp
export TEMP=$ROOT/cache/tmp
export PYTHONPYCACHEPREFIX=$ROOT/cache/pycache
export CUDA_CACHE_PATH=$ROOT/cache/cuda
export MPLCONFIGDIR=$ROOT/cache/matplotlib
export XDG_CACHE_HOME=$ROOT/cache/xdg
export HF_HOME=$ROOT/cache/huggingface
export TRANSFORMERS_CACHE=$ROOT/cache/huggingface/transformers
export TORCH_HOME=$ROOT/cache/torch
export MAST3R_CACHE=$ROOT/cache/mast3r

mkdir -p "$ROOT/logs" "$ROOT/artifacts" "$TMPDIR" "$PYTHONPYCACHEPREFIX" \
  "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$HF_HOME" \
  "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$MAST3R_CACHE"
```

`PYTHON` is the parent CLI interpreter. Real `--model all` runner calls isolate VGGT and MASt3R in their frozen environments from the overlay.

## Deploy/bootstrap: bare ref to detached worktree

Use the deployment script as the canonical path. It initializes or validates the bare repository, pushes/fetches the exact commit, and creates/reuses a detached worktree named by the full 40-character commit.

```bash
./scripts/a100/deploy_commit.sh "$OVERLAY" "$COMMIT"
export WORKTREE=$ROOT/worktrees/$COMMIT
cd "$WORKTREE"
git rev-parse HEAD
git diff --quiet
git diff --cached --quiet
```

Required result:

- `git rev-parse HEAD` equals `$COMMIT`;
- tracked worktree is clean;
- deploy manifest exists at `$ROOT/artifacts/deploy/$COMMIT.json` with a matching `.sha256` file.

Do not force-push, reset, or overwrite an existing worktree with a different commit.

## Schedule counts

| Stage | Count | Notes |
|---|---:|---|
| P1 repeated preflight | 8 per repeat | 1 dev scene x 4 conditions x 2 models |
| P2 smoke | 200 | 10 dev scenes x 10 conditions x 2 models; never gate evidence |
| P3 test | 400 | 20 test scenes x 10 conditions x 2 models |
| P5 downstream | 6 evidence files | 2 models x 3 severity-2 corruptions |
| P5 zero-update | 480 | 20 test scenes x 3 severity-2 corruptions x 2 models x 4 subsets |

## Expected directory and artifact tree

```text
/srv/private/smli/GeoReliab/
  SampleSet.zip
  Points.zip
  git/GeoReliab.git
  worktrees/<commit>/
  cache/
  logs/screens/
  artifacts/
  evidence/
    official_resources.json
    frozen_runtime_identity.json
    tartanair_native_fog_sanity.json
  manifests/
    dtu_inventory.json
    dtu_inventory_provenance.json
    split_view_manifest.json
    frozen_materialization.json
    prepared_inputs_writer.json
    corruption_calibration.json
    corruption_calibration_qa.json
    test_render_lock.json
  prepared/
    calibration_inputs.json
    render_inputs_smoke.json
    render_inputs_test.json
    tartanair_p000_pairs.json
    arrays/
  rendered/smoke/
  rendered/test/
  stage/
    smoke/
    test/
      stage_freeze.json
      stage_evidence_p3.json
      native_phenomenon_gate.json
      stage_evidence.json
      downstream/
      zero_update/
    zero-update/
      bundles/
      terminal_failure.json
    _worker_summaries/
  preflight-real/repeat-a/
  preflight-real/repeat-b/
```

Each completed prediction bundle contains `run_manifest.json`, `prediction_artifact.json`, `audit_record.json`, payload NPZs, dense audit NPZs, and `stage_item.json`. Worker stdout/stderr logs are persisted under `stage/_worker_summaries/` and referenced by SHA-256 in worker summaries.

## Common status/monitor commands

Run after every stage launch and before proceeding downstream:

```bash
./scripts/a100/status.sh "$OVERLAY" "$COMMIT" > "$ROOT/artifacts/status_latest.json"
screen -ls
find "$ROOT/logs/screens" -maxdepth 1 -type f -name '*.log' -print
find "$ROOT/stage" -name '*.partial' -print
find "$ROOT/stage" -name 'ledger.jsonl' -exec tail -n 5 {} \;
du -sh "$ROOT"
nvidia-smi
```

Use stage summary JSON and bound evidence JSON as sources of truth. Logs are diagnostic.

## No-`/home` write guard

Capture `/home` metadata before and after each stage. The diff must be empty.

```bash
find /home/smli/workspace /home/smli/models /home/smli/miniforge3/envs \
  -xdev -printf '%T@ %p\n' | sort > "$ROOT/artifacts/home_readonly_before_${COMMIT}.txt"

# Run exactly one stage here.

find /home/smli/workspace /home/smli/models /home/smli/miniforge3/envs \
  -xdev -printf '%T@ %p\n' | sort > "$ROOT/artifacts/home_readonly_after_${COMMIT}.txt"

diff -u "$ROOT/artifacts/home_readonly_before_${COMMIT}.txt" \
  "$ROOT/artifacts/home_readonly_after_${COMMIT}.txt"
```

Any changed path below `/home/smli` is a resource-policy failure until explained and resolved outside the scientific run.

## P0: readiness, extraction/indexing, manifests, rendering, sanity

Canonical launch:

```bash
cd "$WORKTREE"
./scripts/a100/launch_stage.sh "$OVERLAY" "$COMMIT" p0
```

`launch_stage.sh p0` uses per-operation state files under `$ROOT/artifacts/`:

- `p0_download.json`
- `p0_verify.json`
- `p0_index.json`
- `p0_manifests.json`
- `p0_prepared.json`
- `p0_calibration.json`
- `p0_render_smoke.json`
- `p0_render_test.json`
- `p0_sanity.json`

This is intentional. Do not reuse one `--state` file for all P0 operations because that overwrites history.

Manual equivalent for diagnosis:

```bash
$PYTHON -m georeliab_mve prepare-georeliab --operation download --data-root "$ROOT" --state "$ROOT/artifacts/p0_download.json" --overlay "$OVERLAY"
$PYTHON -m georeliab_mve prepare-georeliab --operation verify --data-root "$ROOT" --state "$ROOT/artifacts/p0_verify.json" --overlay "$OVERLAY"
$PYTHON -m georeliab_mve prepare-georeliab --operation index --data-root "$ROOT" --state "$ROOT/artifacts/p0_index.json" --overlay "$OVERLAY"
$PYTHON -m georeliab_mve prepare-georeliab --operation manifests --data-root "$ROOT" --state "$ROOT/artifacts/p0_manifests.json" --overlay "$OVERLAY"
$PYTHON -m georeliab_mve prepare-georeliab --operation prepared --data-root "$ROOT" --state "$ROOT/artifacts/p0_prepared.json" --overlay "$OVERLAY"
$PYTHON -m georeliab_mve prepare-georeliab --operation calibration --data-root "$ROOT" --state "$ROOT/artifacts/p0_calibration.json" --overlay "$OVERLAY"
$PYTHON -m georeliab_mve prepare-georeliab --operation rendering --stage smoke --data-root "$ROOT" --state "$ROOT/artifacts/p0_render_smoke.json" --overlay "$OVERLAY"
$PYTHON -m georeliab_mve prepare-georeliab --operation rendering --stage test --data-root "$ROOT" --state "$ROOT/artifacts/p0_render_test.json" --overlay "$OVERLAY"
$PYTHON -m georeliab_mve prepare-georeliab --operation sanity --data-root "$ROOT" --state "$ROOT/artifacts/p0_sanity.json" --overlay "$OVERLAY"
```

P0 passes only when all required manifests/evidence files exist and the latest status has no blocker. Treat Rectified/TartanAir whole-archive `.partial` files as stale non-authoritative leftovers; the authoritative evidence is the frozen HTTP Range index plus verified selected-member extraction. The prior P0 Torch import failure after `HOME` redirection is resolved only when `verify_prereqs.sh`, materialization provenance, and isolated adapter probes all confirm that `typing_extensions` imports from the frozen site above with the exact SHA-256.

## P1: repeated preflight

Canonical launch:

```bash
cd "$WORKTREE"
./scripts/a100/launch_stage.sh "$OVERLAY" "$COMMIT" p1 cuda:0
```

Manual equivalent:

```bash
$PYTHON -m georeliab_mve preflight-real \
  --config "$OVERLAY" \
  --output-root "$ROOT" \
  --device cuda:0
```

P1 passes when top-level `status` is `OK`, both repeat roots exist under `preflight-real/repeat-a` and `preflight-real/repeat-b`, and `repeatability.reason_code` is `OK`. If the result is `PREFLIGHT_REPEATABILITY_FAILED`, stop before P2.

## P2: 10-scene smoke

Canonical launch:

```bash
cd "$WORKTREE"
./scripts/a100/launch_stage.sh "$OVERLAY" "$COMMIT" p2 cuda:0
```

Manual equivalent:

```bash
$PYTHON -m georeliab_mve run-georeliab \
  --stage smoke --model all --device cuda:0 --shard 0/1 \
  --config "$OVERLAY" --output-root "$ROOT"
```

P2 passes when the smoke ledger/summary reports `scheduled: 200`, `completed: 200`, and `missing: 0`. Smoke is real-model evidence but permanently non-scientific; never feed it to `evaluate-gates` or P4.

## P3: frozen 20-scene test

Canonical launch:

```bash
cd "$WORKTREE"
./scripts/a100/launch_stage.sh "$OVERLAY" "$COMMIT" p3
```

The script launches two screen shards:

- `cuda:0`, `--shard 0/2`
- `cuda:1`, `--shard 1/2`

Manual equivalent:

```bash
$PYTHON -m georeliab_mve run-georeliab --stage test --model all --device cuda:0 --shard 0/2 --config "$OVERLAY" --output-root "$ROOT"
$PYTHON -m georeliab_mve run-georeliab --stage test --model all --device cuda:1 --shard 1/2 --config "$OVERLAY" --output-root "$ROOT"
```

P3 completes when:

- `stage/test/stage_freeze.json` exists;
- `stage/test/stage_evidence_p3.json` exists and has a matching SHA-256 in the runner summary;
- schedule counts are `scheduled: 400`, `completed: 400`, `missing: 0`.

Invalid outputs remain in evidence and count as failures. Do not remove them to pass a gate.

## P4: native-confidence phenomenon decision

Canonical launch:

```bash
cd "$WORKTREE"
./scripts/a100/launch_stage.sh "$OVERLAY" "$COMMIT" p4 cuda:0
```

Manual equivalent:

```bash
$PYTHON -m georeliab_mve audit-georeliab \
  --stage-evidence "$ROOT/stage/test/stage_evidence_p3.json" \
  --output "$ROOT/stage/test/native_phenomenon_audit.json"

$PYTHON - <<'PY'
import json
from pathlib import Path
from georeliab_mve.runner import RunnerContext, write_native_phenomenon_gate_from_audit_output
root = Path('/srv/private/smli/GeoReliab')
overlay = root / 'worktrees' / 'REPLACE_WITH_FULL_PROJECT_COMMIT' / 'configs/a100_real_mve_overlay.toml'
audit = json.loads((root / 'stage/test/native_phenomenon_audit.json').read_text(encoding='utf-8'))
print(write_native_phenomenon_gate_from_audit_output(RunnerContext(root=root, output_root=root, config_path=overlay, device='cuda:0'), audit))
PY
```

Interpretation of `$ROOT/stage/test/native_phenomenon_gate.json`:

- `PASS`: proceed to P5;
- `FAIL`: freeze negative GeoReliab result and skip P5;
- `BLOCKED_PENDING_EVIDENCE`: evidence, freeze, or digest binding is missing/inconsistent.

The loader recomputes the P3 decision and rejects tampered status/reasons with `NATIVE_CONFIDENCE_GATE_DECISION_MISMATCH`.

## P5: conditional downstream and zero-update

Run P5 only after P4 writes a bound `PASS` gate.

Canonical launch:

```bash
cd "$WORKTREE"
./scripts/a100/launch_stage.sh "$OVERLAY" "$COMMIT" p5
```

The script launches two zero-update shards:

- `cuda:0`, `--shard 0/2`
- `cuda:1`, `--shard 1/2`

Manual equivalent:

```bash
$PYTHON -m georeliab_mve run-georeliab --stage zero-update --model all --device cuda:0 --shard 0/2 --config "$OVERLAY" --output-root "$ROOT"
$PYTHON -m georeliab_mve run-georeliab --stage zero-update --model all --device cuda:1 --shard 1/2 --config "$OVERLAY" --output-root "$ROOT"
```

P5 completes when downstream evidence count is 6, zero-update schedule counts are `scheduled: 480`, `completed: 480`, `missing: 0`, and full `$ROOT/stage/test/stage_evidence.json` exists.

If `$ROOT/stage/zero-update/terminal_failure.json` exists with `reason_code: P5_INVALID_SUBSET_PREDICTION`, stop and report terminal FAIL. Do not rerun around invalid subsets.

## P6: final evidence bundle and one-page table

Canonical launch:

```bash
cd "$WORKTREE"
./scripts/a100/launch_stage.sh "$OVERLAY" "$COMMIT" p6 cuda:0
```

The canonical launcher runs `status.sh` and `finalize_p6.sh` itself. Do not hand-write,
repair, or replace the final bundle: the finalizer rejects incomplete schedules,
non-terminal gates, dirty/mismatched worktrees, unavailable resource evidence, and
digest drift. The only authoritative outputs are:

- `$ROOT/artifacts/p6_evidence_bundle.json`
- `$ROOT/artifacts/p6_one_page_decision.md`
- `$ROOT/artifacts/p6_one_page_decision.md.sha256`

To validate an already completed P6 idempotently, rerun the same canonical P6 launch;
the finalizer verifies every stored path and digest before reporting reuse.

The P6 bundle must include project commit, upstream/checkpoint/environment SHA values, split/materialization/prepared/calibration/render-lock hashes, schedule counts, invalid/missing counts, GPU/storage status, worker logs, gate reasons, and the pending-Geometry selection note.

If GeoReliab passes while Geometry remains non-terminal, report `GEORELIAB_PASS_PENDING_GEOMETRY` / `BLOCKED_PENDING_GEOMETRY`, not final track selection.

## Resume and stale partial recovery

Runner behavior:

- valid completed bundles are skipped;
- `.partial` prediction bundle directories are removed and rerun;
- stale Rectified/TartanAir whole-archive `.partial` files are non-authoritative and must not be deleted/resumed by this runbook;
- committed bundles with mismatched stage fingerprint or digest raise a provenance conflict;
- append-only ledgers live at `$ROOT/stage/<stage>/ledger.jsonl`;
- claims live below `$ROOT/stage/<stage>/claims/` and should not be deleted while workers may be active.

Recovery checklist:

```bash
./scripts/a100/status.sh "$OVERLAY" "$COMMIT"
find "$ROOT/stage" -name '*.partial' -print
find "$ROOT/stage" -path '*/claims/*' -type f -print
find "$ROOT/stage" -name 'ledger.jsonl' -print
```

If a screen died and only stage-local `.partial` bundles remain, rerun the same stage command with the same `--stage`, `--model`, `--shard`, `--config`, and `--output-root`. Do not change scientific parameters.

## Budget and storage blocks

If a runner returns `BLOCKED_RESOURCE_BUDGET`:

1. stop launching shards;
2. save the runner JSON, `./scripts/a100/status.sh` output, `du -sh "$ROOT"`, and `nvidia-smi`;
3. do not reduce P3 from 400 or P5 zero-update from 480;
4. do not reduce scenes, corruptions, severities, models, views, bootstrap count, or random masks.

Any cache/temp/log path below `/home` is a resource-policy failure even if `/srv` still has space.

## Invalid-output handling

Invalid outputs are scientific observations:

- P3 invalid predictions remain scheduled and count as failures;
- invalid predictions cannot be silently removed from verified conditions;
- P5 invalid subset predictions produce terminal `P5_INVALID_SUBSET_PREDICTION`;
- invalid bundle payloads must retain `run_manifest.json`, `prediction_artifact.json`, `audit_record.json`, and reason code.

## Freeze and provenance hard stops

Stop on these conditions:

- `provenance-conflict`
- `immutable P3 stage_evidence_p3.json would change`
- `test rendering parameters or split changed after freeze`
- `refusing to shrink frozen P3 test grid`
- `NATIVE_CONFIDENCE_GATE_FREEZE_MISMATCH`
- `NATIVE_CONFIDENCE_GATE_EVIDENCE_MISMATCH`
- `NATIVE_CONFIDENCE_GATE_DECISION_MISMATCH`

Do not edit evidence JSON to bypass these. Fix the underlying mismatch before scientific execution, or freeze the run as blocked/failed if P3 has already started.

## Reason-code reference

| Code | Meaning | Operator action |
|---|---|---|
| `OK` | Stage or repeat passed | Continue |
| `DRY_RUN` | Command shape checked without real execution | Run real command when prerequisites are ready |
| `WORKER_FAILED` | Isolated model worker failed | Inspect `stage/_worker_summaries/*.stderr.log` |
| `ADAPTER_EXCEPTION` | Adapter raised or emitted no valid payload | Keep invalid output; inspect adapter exception JSON |
| `INVALID_PREDICTION` | Model output invalid but retained | Count as failure; do not drop |
| `PREFLIGHT_REPEATABILITY_FAILED` | P1 repeatability thresholds failed | Stop before P2 |
| `P3_SCHEDULE_COUNTS_INVALID` | Test evidence incomplete or not frozen 400 grid | Complete/fix P3 without shrinking grid |
| `INVALID_OUTPUT_IN_VERIFIED_CONDITION` | P3 invalid output in required condition | Terminal gate failure unless protocol changes |
| `CONFIDENCE_FAILURE_GATE_NOT_MET` | Native-confidence phenomenon failed | Short-circuit P5 |
| `TARTANAIR_SANITY_GATE_NOT_MET` | Native fog sanity missing/failed | Stop; P4 cannot pass |
| `P5_DOWNSTREAM_SCHEDULE_COUNTS_INVALID` | P3-only gate reached pre-P5 boundary | In native-gate binding, this is the P4 PASS signal |
| `NATIVE_CONFIDENCE_GATE_NOT_PASS` | P5 requested after P4 FAIL | Do not run P5 |
| `NATIVE_CONFIDENCE_GATE_MISSING` | P5 requested before P4 gate exists | Run P4 first |
| `NATIVE_CONFIDENCE_GATE_INJECTION_FORBIDDEN` | Supplied gate differs from canonical bound gate | Use on-disk canonical gate |
| `P5_INVALID_SUBSET_PREDICTION` | Zero-update invalid subset terminal evidence | Stop P5 and report FAIL |
| `DOWNSTREAM_HARM_GATE_NOT_MET` | Downstream harm criterion failed | Final GeoReliab FAIL |
| `ZERO_UPDATE_GATE_NOT_MET` | Zero-update AUROC-gain criterion failed | Final GeoReliab FAIL |
| `ALL_GEORELIAB_GATES_MET` | GeoReliab gate passed | Report PASS pending Geometry |
| `BLOCKED_PENDING_EVIDENCE` | Evidence missing or unbound | Produce missing evidence; do not infer result |
| `BLOCKED_RESOURCE_BUDGET` | Next stage exceeds frozen resource limit | Stop without shrinking grid |
| `BLOCKED_PENDING_GEOMETRY` | GeoReliab may be terminal, Geometry is not | Do not make final track selection |

## Local verification performed for this runbook

The documentation task verified the current CLI command shapes locally:

```powershell
python -m georeliab_mve --help
python -m georeliab_mve prepare-georeliab --help
python -m georeliab_mve preflight-real --help
python -m georeliab_mve run-georeliab --help
python -m georeliab_mve audit-georeliab --help
python -m georeliab_mve evaluate-gates --help
```

A100 stage commands still require Task 7 execution against the real host, data, checkpoints, and GPUs.
