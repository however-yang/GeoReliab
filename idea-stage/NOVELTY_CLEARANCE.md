---
status: supporting
authority: GEORELIAB_TASKBOOK.md
execution_entry: false
superseded_by: GEORELIAB_TASKBOOK.md#v2.2
---

## v2.2 mainline execution override

This document is supporting novelty evidence for the single authoritative mainline in `GEORELIAB_TASKBOOK.md`. Its NC1-NC4 clearance remains valid as literature-boundary evidence, but it does not authorize an experiment. The current phase is infrastructure qualification: Gate 1 CPU fault-matrix qualification and Gate 2 12-unit GPU recovery smoke must both pass before the fresh development Pilot. Any older sentence that directs an immediate DTU or 20-scene run is superseded by the Taskbook Phase 0 order. The Attempt-05 partial corpus remains non-resumable and there is no scientific result. The [authoritative Taskbook](../GEORELIAB_TASKBOOK.md#v2.2) controls execution order.
# NOVELTY_CLEARANCE — GeoReliab v2.2 定稿主线查新（任务书 Phase 0 · Step 1 产出）

**日期**: 2026-08-06
**对象**: `GEORELIAB_TASKBOOK.md` 定稿版四主张（NC1–NC4）
**执行**: /novelty-check（Phase A–D 完整流程）
⚠️ **降级标注**: Codex MCP 本会话不可用 → Phase C 跨模型验证未执行，本报告为 Claude 单模型判断（与 NOVELTY_REPORT / V2 / V3 相同降级）。`verify_papers.py` 未解析 → 按 Policy D1 人工经 arXiv/出版页直接核验，未标 [UNVERIFIED] 的条目均已核验。WebSearch 被安全过滤误拦 → 检索经 web-search-prime 备用通道完成。

---

## Proposed Method

GeoReliab v2.2 "Ranking Is Not Warning"：双轴协议（DTU 7-lighting 真实配对反事实 → CRR/SFR；TartanAir 单调 severity → Boundary Lag）审计 3D 重建系统 native confidence 的失败预警能力，加点级→任务级迁移（NC3）与 reliability-aware view admission（NC4）。

## Core Claims

| # | Claim | Novelty | Closest（均已直接核验） | 本轮变化 |
|---|-------|---------|------------------------|---------|
| NC1 | 同场景真实配对反事实协议测 confidence 预警 | **HIGH（本轮增强）** | VGGT-UQ 2606.16479；RealX3D 2512.23437 | **排雷通过**：VGGT-UQ 正文明确只用 DTU 单一光照 L3（"选 L3 因过曝更少"），光照非实验变量；指标族为 AUSE/PAvPU/稀疏化 = 纯 ranking/校准。RealX3D 有真实退化但零 confidence 评测、仅 1 个配对场景且测试集扣留 GT |
| NC2 | Ranking≠Warning 指标体系（CRR/SFR/severity-Lag） | MEDIUM-HIGH（不变） | Ovadia 2019 (1906.02530) | 无新增碰撞；"warning/failure-awareness + GFM confidence" 组合检索为空。Ovadia 仍是必须正面划界的最大威胁 |
| NC3 | 点级 confidence → pose/fusion 任务级失败迁移 | **HIGH（不变）** | ViPE（工程性 top-50% confidence 过滤，无系统研究）；Trust3R 2605.19539 | 检索仍为空白 |
| NC4 | Reliability-aware view admission control | **HIGH-MEDIUM（新邻居，划界后成立）** | **RobustVGGT / Emergent Outlier View Rejection 2512.04012（本轮新发现，前两轮查新均遗漏）** | 见下 |

## 本轮新发现的相邻工作（三篇，全部必引）

### 1. RobustVGGT — Emergent Outlier View Rejection in VGGT（2512.04012, KAIST）
- **做什么**: 处理 in-the-wild 图像集中的 **distractor/无关图像**（错误场景），用 VGGT 末层 cross-view attention（τ=0.05）+ 特征余弦相似度（τ=0.65）做 training-free 过滤。数据：Phototourism/On-the-Go/RobustNeRF/ETH3D。
- **不做什么**: 不评 confidence 预警/校准；无配对反事实；无 DTU 光照；噪声模型 = 混入其他场景图像，非同场景退化观测。
- **对 NC4 的划界**: 失败模式正交——它剔除"错场景的图"，我们裁决"对场景但退化的图"是否可信 + defer 形式化。**必须在 admission control 章节正面引用划界**。
- **对 C5' 的警示**: 其机制（内部统计、training-free 评分）与 C5' 同源。C5' 不得声称"首个 training-free 内部信号评分"，卖点应为"首个针对退化致失效的 warning 修复"。

### 2. RealX3D — Physically-Degraded 3D Benchmark（2512.23437）
- **做什么**: 真实物理退化 benchmark（低光 −2.7EV、原位烟雾发生器多密度、遮挡/反射各 5 级、运动/失焦模糊），评 VGGT/PI3/MapAnything/DepthAnything3 + GS 变体 + 经典管线，指标 PSNR/SSIM/LPIPS + Chamfer/IoU。定位 = restoration + robustness。
- **不做什么**: 零 confidence/uncertainty 评测；clean+degraded 全配对仅 1 个验证场景，测试集扣留 GT 几何。
- **对 NC1 的影响**: 不碰撞（无 confidence 维度）；且反证 DTU 7-lighting 的不可替代性（全场景配对 + 完整 GT）。**叙事素材**: "已有真实退化 benchmark 只问'退化多少'，不问'系统知不知道'"。
- **机会**: 其真实 severity 层级（烟雾密度/曝光档）可作 Axis B 的真实性补充（受 GT 扣留限制，以 pose 类 failure 为主）——列为可选扩展，非承诺。

### 3. CorAl — Introspection for radar/lidar alignment（Adolfsson et al., RAS 2022，无 arXiv ID 引用出版版）
- lidar/radar 配准质量自省、misalignment 拒识。补入威胁地图 Introspection 族（与 1607.08665 并列），扩展"failure awareness 在其他传感器模态已有先例、在视觉 GFM confidence 上空白"的论证。

## Closest Prior Work（增量表，存量见 NOVELTY_REPORT_V2）

| Paper | Year | Venue | Overlap | Key Difference |
|-------|------|-------|---------|----------------|
| RobustVGGT 2512.04012 | 2025 | arXiv | training-free view 过滤 | distractor 剔除 vs 退化观测可信裁决；不评预警 |
| RealX3D 2512.23437 | 2025 | arXiv | 真实物理退化 + severity | 零 confidence 评测；配对/GT 覆盖不足以支撑反事实协议 |
| VGGT-UQ 2606.16479 | 2026 | arXiv (ISPRS?) | DTU 上 VGGT 不确定性 | **仅 L3 单光照，光照非变量**；纯稀疏化/校准指标，无预警、无配对 |
| CorAl (RAS) | 2022 | RAS | 感知自省先例 | lidar/radar 配准域，非视觉 GFM confidence |

## Overall Novelty Assessment

- **Score: 8/10 — PROCEED**（分数持平 V2，置信度显著上升：最大未决风险 VGGT-UQ 排雷通过）
- **Key differentiator**: 双分离命题（会排序≠会报警；点级≠系统级）+ 全 GT 真实配对反事实载体（DTU 7-lighting），现有全部近邻（含本轮新发现三篇）均不评 confidence 预警
- **Risk（审稿人会引什么）**:
  1. Ovadia 2019 — 地位不变，划界三重差异必须写透（paired same-scene / 几何任务 / severity-lag 操作化）
  2. RobustVGGT — admission control 与 C5' 两处都要主动引用划界，否则"training-free 内部信号"显得撞车
  3. RealX3D — 若不引用，审稿人会问"为何不用真实退化 benchmark"；引用后反而是 DTU 选择的论据
- **前两轮查新的教训**: 2512.04012 与 2512.23437 均为 2025-12 论文，V2/V3 两轮均未检出——检索词偏 "reliability/uncertainty" 族，漏掉 "robustness/outlier/benchmark" 族。后续 Phase 2 补查新时需双词族并检。

## Suggested Positioning（对任务书的三处增补，已同步写入）

1. 威胁地图新增 RobustVGGT / RealX3D / CorAl 三行
2. C5' stretch goal 措辞收紧：卖点 = "修复 warning 失效"，不声称 training-free 首创
3. RealX3D 列为 Axis B 可选真实性扩展（非承诺项）

## 结论（对应任务书 Phase 0 Gate）

**NC1 成立，Literature Threat Audit 通过。** 无 counterfactual reliability evaluation 先例、无 severity-lag formulation 先例、VGGT-UQ 未用 DTU 光照变体。可进入 Phase 0 Step 2（DTU 20-scene pilot）。
