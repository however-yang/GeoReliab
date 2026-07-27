# GeoReliab real dual-model MVE implementation plan

Status: approved for execution on 2026-07-26.

## Global constraints

- Advance the project from `GEORELIAB_PROTOCOL_READY` to a real, auditable
  VGGT/MASt3R scientific gate. Geometry models, CUT3R, and Deformable World are
  out of scope.
- Runtime root is `/srv/private/smli/GeoReliab`. Treat `/home/smli` as
  read-only and reuse the frozen upstream repositories, environments, and
  checkpoints there.
- Preserve the protocol-ready baseline commit. Use Lore-format commits.
- Never put data, caches, logs, predictions, or scientific artifacts in Git.
- Smoke evidence is real-model evidence but permanently non-scientific and
  must never enter a gate.
- Test evidence is scene-disjoint and uses the frozen 20 DTU scenes. Test
  parameters may not be changed after test inference starts.
- Invalid model outputs remain in the schedule and count as failures.
- Primary statistics use the scene as the unit, 10,000 paired bootstrap
  replicates, and Holm correction. Pixels are not independent samples.
- Hard limits are 50 GPU-hours and 1 TB. If the next frozen stage cannot fit,
  return `BLOCKED_RESOURCE_BUDGET`; never shrink the test grid.
- GeoReliab PASS while Geometry is non-terminal yields
  `GEORELIAB_PASS_PENDING_GEOMETRY` / `BLOCKED_PENDING_GEOMETRY`, not final
  track selection.

## Frozen resources

- VGGT source commit:
  `a288dd0f14786c93483e45524328726ab7b1b4ce`
- VGGT checkpoint:
  `/home/smli/models/vggt/model.pt`
- VGGT checkpoint SHA-256:
  `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`
- MASt3R source commit:
  `f5209afc300cec36239a7ac992263f36847bbba0`
- DUSt3R source commit:
  `3cc8c88c413bb9e34c41db0e0eef99c2ee010b12`
- CroCo source commit:
  `d7de0705845239092414480bd829228723bf20de`
- MASt3R checkpoint SHA-256:
  `0a615eb05fa9db654050aa655945ee5696e7c6c1b7f93f1ee8c37249010f6feb`
- MASt3R config SHA-256:
  `718eb93dc4f9e4332b60cc0041af962d712cbd346d7770ce35c5b22cff68eae4`
- VGGT environment: Python 3.10.20, Torch 2.3.1+cu121.
- MASt3R environment: Python 3.10.20, Torch 2.5.1+cu121.
- Frozen typing_extensions dependency: site `/home/smli/miniforge3/pkgs/typing_extensions-4.15.0-pyhcf101f3_0/site-packages`, file `typing_extensions.py`, version `4.15.0`, SHA-256 `433d11d170d3a24d2eb065ebc1bfe848cea7e3d7ce68567ab52bea2d4c2f7ed8`. This is an environment dependency freeze only; it does not change scientific thresholds, scenes, schedule counts, or the test grid.

## Frozen data protocol

- DTU resources: official SampleSet, Points, and Rectified data. Use lighting
  condition 3, original 1600x1200 rectified RGB, official cameras, structured
  light point clouds, and observability masks.
- Frozen test scenes:
  `[1, 9, 10, 11, 12, 13, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]`.
- Exclude scans 4 and 15 from every support split.
- From eligible non-test scenes sorted by
  `SHA256("GeoReliab-DTU-v1:scan<ID>")`, assign dev=10, calibration=10,
  reference-token=5.
- Select eight views per scene with normalized camera-center farthest point
  sampling, starting at the smallest view id and breaking ties by smaller id.
- Generate and hash split/view manifests before any model inference.
- TartanAir sanity uses V2 `GreatMarsh/Data_easy/P000/lcam_front`, 100 uniformly
  sampled RGB/depth frames. Require negative depth/local-RMS-contrast Spearman
  in at least 80 frames. It is physical-direction sanity only, not paired
  scientific evidence.

## Frozen corruption protocol

- Render in linear RGB, then encode one shared 8-bit PNG consumed by both
  models.
- Fog uses Koschmieder rendering. `d_ref` is the global median valid
  calibration depth, airlight is the global median of each calibration
  image's brightest 0.1% colors, and target transmittance at `d_ref` is
  `[0.80, 0.50, 0.25]`.
- Low-light-noise uses exposure `[0.5, 0.25, 0.125]`, Poisson peak
  `[2048, 512, 128]`, and Gaussian read sigma `[0.002, 0.005, 0.010]`.
- Defocus uses focus depth=`d_ref`, 32 inverse-depth layers, a disk PSF, and
  calibration depth CoC p95 targets `[4, 10, 20]` original-image pixels.
- Random seed is the first 64 bits of
  `SHA256(sample_key + view_id)`.
- Calibration QA must verify monotonic fog transmittance, darkness/noise, and
  CoC/edge-energy before test rendering. Synthetic-fog physical direction is
  checked at scene grain by pooling the already-produced linear-RGB fog renders into
  depth-defined 32x32 patch median-depth/local-RMS contrast pairs and requiring
  every calibration scene to have `rho_fog - rho_clean < 0` with strictly
  increasing absolute effect across severities. GT geometry hashes never change.

## Frozen model protocol

- VGGT: `eval()`, `no_grad()`, A100 bfloat16, official preprocessing; use
  depth plus predicted camera unprojection, `depth_conf` as raw confidence,
  and `-log(max(depth_conf - 1, 1e-12))` as higher-is-worse risk.
- MASt3R: 512 input, complete symmetric graph for the same eight views,
  `lr1=0.07/niter1=300`, `lr2=0.01/niter2=300`, `refine+depth`,
  `shared_intrinsics=False`; call `get_dense_pts3d(clean_depth=False)`;
  prohibit `clean_pointcloud`, TSDF, and confidence thresholding. Use canonical
  raw excess confidence and risk `-log(max(conf, 1e-12))`. Preserve pairwise
  outputs for traceability.

## Frozen audit and decision protocol

- Align predicted camera centers to GT with reflection-free Umeyama Sim(3).
  Do not use GT-surface ICP.
- Apply official DTU observability masks after alignment; do not confidence
  filter.
- Voxelize at 0.2 mm; coordinate is centroid and risk is voxel median.
- GT error is nearest structured-light point distance. Primary failure is
  `error > 2 mm`, with 1 mm and 5 mm sensitivity.
- `rho = Spearman(risk, error)`. Clean must have `rho >= 0.2`; relative decline
  is `(rho_clean-rho_s3)/rho_clean`; require
  `rho_clean >= rho_s1 >= rho_s2 >= rho_s3`.
- Severity-3 failure AUROC uses the 2 mm label. Fewer than 16/20 scenes with
  both classes makes the AUROC branch ineligible.
- ECE is diagnostic only, using a monotonic logistic mapping fit on
  calibration-clean and applied unchanged to test.
- Phenomenon gate requires both models across all three corruptions to satisfy
  the frozen rho-decline or AUROC condition with corrected scene-level CIs.
- If the phenomenon gate fails, emit terminal GeoReliab FAIL and skip
  downstream and zero-update.
- Downstream uses the six `(model, severity-2 corruption)` conditions,
  coverages `[0.9, 0.7, 0.5, 0.3]`, F-score@2mm coverage-AUC, and 100 fixed
  random masks per scene. At least two conditions need the native-minus-random
  harm CI upper bound below zero.
- Zero-update uses four 6-of-8 subsets omitting `[0,4]`, `[1,5]`, `[2,6]`,
  `[3,7]`; align with predicted cameras only; use normalized NN-dispersion
  risk. AUROC gain must be at least 0.10 with CI lower bound above zero.

### Task 1: Artifact contract v1.1 and route governance

Implement schema v1.1 with backward-compatible v1.0 readers and v1.1-only
writers. Add `RunMode.SMOKE` and `NON_SCIENTIFIC_SMOKE`. REAL and SMOKE require
checkpoint SHA and project/upstream/environment provenance. Define and validate
the NPZ payload schemas for geometry, raw native confidence, valid mask, and
dense audit. Add a bundle validator that fails closed on missing keys, shape
mismatch, digest mismatch, and cross-run/sample linkage. Introduce explicit
terminal/non-terminal gate semantics and the pending-Geometry selection states.
Write failing tests first, then implementation.

### Task 2: DTU/TartanAir preparation and deterministic corruptions

Implement deterministic split generation, eligible-scene validation,
camera-center FPS, corruption parameter calibration, linear-RGB renderers,
calibration QA, TartanAir native-fog sanity, manifest hashes, and download/
resource verification plumbing. Implement the `prepare-georeliab` CLI. Keep
download URLs and runtime paths in an A100 overlay, not the scientific
threshold config. Write failing tests first.

### Task 3: Geometry audit and scientific statistics

Implement reflection-free Umeyama Sim(3), DTU mask application, 0.2 mm
voxel aggregation, nearest-GT error, invalid-output accounting, primary and
sensitivity labels, scene-level rho/AUROC, eligibility checks, 10,000 paired
bootstrap, Holm correction, diagnostic calibration/ECE, downstream
coverage-AUC, and zero-update disagreement scoring. Implement
`audit-georeliab` and update gate evidence parsing. Write synthetic regression
tests first.

### Task 4: Real VGGT and MASt3R adapters

Implement adapters around the frozen upstream repositories and checkpoints.
Both adapters consume the exact same rendered PNG digests and record
model-specific preprocessing digests. Enforce all frozen inference options,
raw-confidence direction, no thresholding, MASt3R `clean_depth=False`, and
payload schemas. Make optional heavyweight imports lazy so local contract tests
run without upstream packages. Write adapter contract tests first.

### Task 5: Resumable runner, preflight, budget, and CLI orchestration

Implement atomic `.partial` writes, valid-artifact skip/resume, invalid output
retention, schedule generation for P1/P2/P3/P5, shard/device selection,
GPU-hour/storage accounting, and hard budget blocking. Add `preflight-real`
and `run-georeliab --stage smoke|test|zero-update`. Scientific gates accept
only schema-v1.1 test evidence. Write resume and schedule-count tests first.

### Task 6: Deployment and operator documentation

Add the tracked A100 overlay, environment/source/hash verification scripts,
bare-repository and detached-worktree deployment scripts, stage launch/status
commands, and a runbook. The scripts must never write below `/home/smli`.
Document exact schedule counts, outputs, reason codes, and recovery steps.
Validate shell syntax and local CLI help.

### Task 7: Remote P0-P6 execution

P0 readiness must fail closed unless the frozen `typing_extensions.py` source, version, and SHA-256 are present in the overlay, environment lock, materialization provenance, and isolated model-runtime probes. This restores Torch imports after runtime cache setup redirects `HOME` away from `/home/smli` while keeping `/home` read-only.

Create `/srv/private/smli/GeoReliab/git/GeoReliab.git`, push the exact
implementation commit, create a detached runtime worktree, prepare official
data, run readiness and P1, then P2, P3, P4, and conditionally P5/P6. Preserve
all logs and manifests. Do not start a stage unless readiness and remaining
budget checks pass. Finish with the single-page decision table, evidence JSON,
schedule/missing/invalid counts, GPU-hours, peak memory, and all provenance
hashes.
