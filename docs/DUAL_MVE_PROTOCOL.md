# Dual-MVE Protocol v1.0

**冻结日期**：2026-07-26  
**内部冻结点**：2026-09-15；CVPR 2027 官方截止日仍记为 TBA。  
**当前状态**：基础设施已通过本地测试；真实模型、checkpoint 和数据尚未配置，因此所有科学决策保持 `BLOCKED`。

## 1. 不可漂移的决策

1. Geometry causal audit 是条件式主线，GeoReliab 是第一备选，Deformable World 暂停。
2. Geometry PASS 时无条件优先；Geometry 未通过且 GeoReliab PASS 时选择 GeoReliab。
3. 两条真实科学 gate 均 FAIL 时停止本轮项目，不自动转向 Deformable World。
4. fixture、dry-run、缺 checkpoint 或缺数据都不能选择论文主线。

阈值的唯一机器可读来源是 [`configs/dual_mve_protocol.toml`](../configs/dual_mve_protocol.toml)，代码加载时会拒绝阈值漂移。

## 2. Artifact contract

实现位于 [`georeliab_mve/contracts.py`](../georeliab_mve/contracts.py)，schema version 固定为 `1.0`。

- `RunManifest`：run id、真实/fixture 模式、科学有效性、模型、checkpoint SHA-256、dataset/split/seed、intervention/corruption 版本、环境与固定输入 digest。
- `PredictionArtifact`：必填 `run_id`、`sample_key`、geometry/native-confidence/valid-mask URI、hook 位置、runtime 和 peak memory。
- `AuditRecord`：必填 `run_id`、GT error、failure label、selection score、coverage、accept mask 与 downstream outcome。
- `sample_key` 严格为 `dataset/split/scene/view_set/condition/severity/seed`。
- invalid prediction 必须记为 failure 且不可 accepted；linkage 同时核对 invalid 状态与 `sample_key` 的 dataset/split/seed，禁止静默剔除或跨 run 混用。

`dev`、`reference-token`、`calibration`、`test` 必须 scene-disjoint。统计单位只能是 scene/sequence；bootstrap 固定 10,000 次，主要多重比较使用 Holm 校正。

## 3. Geometry gate

工程门：Spatial-MLLM、SpatialStack、GUIDE 三候选中任意至少两个公开 checkpoint 可复现且可挂接 geometry fusion layer，VSI-Bench 与 CVT-Bench 均已就绪；gate 只接受这三个冻结名称，不允许任意模型替代。

科学门满足以下一条即可：

- faithful use：至少两个模型分别在两个 benchmark、至少两个 geometry-required strata 上满足 `Δgeom ≥ 0.05`、95% CI 下界 `> 0`、patch recovery `≥ 0.30`；
- causal non-dependence：至少两个模型在两个 benchmark、至少两个 geometry-required strata 的 ±0.02 TOST 等效检验通过，同时确认 post-fusion representation 已改变；
- 任一 PASS 都要求 VSI-Bench 与 CVT-Bench 各有独立 semantic-control 记录，且 RGB、prompt、decoder、seed 的固定输入校验通过。

若只有 zero/mean replacement 有效，而统计匹配 swap 无效，固定判为 OOD intervention artifact。

## 4. GeoReliab gate

工程门：冻结的 VGGT、MASt3R 二者与 DTU-20 scenes 均已就绪，任意同名替代不能通过。CUT3R 是完整实验第三模型，不阻塞 MVE。

GO 必须同时满足：

- 至少两个模型分别覆盖 fog、low-light-noise、defocus 三类退化的 severity 1/2/3 完整记录；`ρ` 相对下降的 scene-block CI 下界严格大于 0.50，或 failure AUROC 的 CI 上界严格小于 0.65；
- 至少两个不同 `(model, condition)` 的 native-confidence filtering 相对 coverage-matched random 的 paired CI 上界小于 0；
- zero-update AUROC gain 至少 0.10 且 CI 下界大于 0。

点估计、endpoint-only severity 记录、重复记录、仅极端 OOD 信号都不能通过 gate。代码直接验证 `|ρ_clean| ≥ |ρ_s1| ≥ |ρ_s2| ≥ |ρ_s3|`，并另行要求 corruption severity QA、跨视图一致性、GT 几何不变与 TartanAir 原生雾 sanity 全部通过。

## 5. 本地命令

```powershell
# 验证外部资源；当前预期 exit code 2 / BLOCKED
python -m georeliab_mve readiness --protocol configs/dual_mve_protocol.toml

# 验证全链路；输出永久标记 NON_SCIENTIFIC_FIXTURE
python -m georeliab_mve dry-run `
  --protocol configs/dual_mve_protocol.toml `
  --output-dir artifacts/dry-run

# 验证单个 artifact
python -m georeliab_mve validate-artifact `
  --type manifest artifacts/dry-run/run_manifest.json

# 从预注册证据 JSON 计算两个 gate 与唯一主线
python -m georeliab_mve evaluate-gates `
  --geometry geometry_evidence.json `
  --georeliab georeliab_evidence.json `
  --output decision.json

python -m pytest
```

真实执行前必须为所有 required resource 以及各模型候选组中足够数量的成员增加 `local_path`；checkpoint 文件同时增加 SHA-256。`readiness` 会明确报告 Geometry 2-of-3 与 GeoReliab 2-of-2 的候选计数；返回 ready 前不得把任何输出写成科学结果。

## 6. 外部阻塞与责任边界

- 本仓库不包含 Spatial-MLLM、SpatialStack、VGGT、MASt3R 的代码或权重，也没有 VSI-Bench、CVT-Bench、DTU、TartanAir 数据。
- [`georeliab_mve/adapters.py`](../georeliab_mve/adapters.py) 冻结了外部 adapter 边界，但不能在未知上游版本上伪造 hook 或 tensor 名称。
- 获得真实 checkout 后，应在独立 integration 模块实现 adapter，并将上游 commit/checkpoint hash 写入 `RunManifest`。
- dry-run 只验证 schema、统计、gate、选择矩阵和文件流，不代表任何模型行为。
