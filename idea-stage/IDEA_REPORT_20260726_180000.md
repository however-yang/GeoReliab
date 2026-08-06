---
status: superseded_historical
authority: GEORELIAB_TASKBOOK.md
execution_entry: false
superseded_by: GEORELIAB_TASKBOOK.md#v2.2
---

> Historical document retained for provenance. It is not an execution entry; the v2.2 mainline and all current phase gates are defined by GEORELIAB_TASKBOOK.md.
# Research Idea Report (v3 — 完全放开的视觉方向)

**Direction**: CVPR 2027，视觉为中心，围绕技术栈（深度/3D重建/不确定性/具身感知）开放探索；医疗/机器人仅为可选应用域
**Generated**: 2026-07-26（v3；v2=GeoReliab pipeline 报告见 `IDEA_REPORT_20260726_153000.md`）
**Ideas evaluated**: 15 generated (5-lens fan-out) → 15 survived feasibility → 0 piloted (无 GPU 环境，全部 SKIPPED: needs pilot) → 3 recommended
⚠️ Codex MCP 不可用：triage 与 novelty 判断为 Claude 单模型（降级标注）；建议配置后用 `RESEARCH_REVIEW_REQUEST.md` 模式补外部评审

## Landscape Summary（v3 增量）

前馈 3D 几何基础模型（VGGT 血统）的能力竞赛已扩展到 4D（PAGE-4D、Any4D）、长序列（VGGT-Long）、速度（FastVGGT）；可靠性侧持续升温（Trust3R、VGGT-UQ、2605.18754 metric-reliability——**三个月内第三篇了**）。空间推理 MLLM benchmark 已severely crowded（VSI/3DSR/Spatial457/OmniSpatial/ViewSpatial/MMSI/SpatialScore），但**meta-评估（benchmark 测的是什么、几何注入是否被真正使用）几乎空白**。视频世界模型×几何一致性正在起步（GEM-4D、HY-World 2.0、2605.15185）。**连续非刚性形变是 4D 竞赛的盲区**：现有 4D 模型处理"刚体在动"，不处理"表面在变形"；手术/内窥镜重建社区仍在逐场景优化（EndoGS/SAGS），无人把形变能力灌入前馈基础模型。

## Recommended Ideas (ranked)

### Idea 1: "Feed-Forward 3D Models Meet a Deforming World" — **NEW #1**（聚类 α）
- **Method（我们实际做什么）**:
  1. 构建**形变幅度分级评估**：用光流残差非刚性指数把公开序列（DyCheck/PointOdyssey/DeformingThings4D 合成可控 + StereoMIS/Hamlyn stereo GT）按形变强度分箱，测 VGGT/MASt3R/CUT3R/PAGE-4D 误差曲线，定位"相变边界"，并测 native confidence 能否察觉越界
  2. **逐层归因**：缓存逐层 token 与跨视图注意力，用 linear probe 定位"刚性假设住在哪几层"——形变区域的多视图证据在哪一层被丢弃、退化为单目先验
  3. **轻量修复**：在归因指出的层位插 LoRA deformation adapter，用形变一致性损失（scene-flow 平滑、as-isometric-as-possible——移植自作者内窥镜 SLAM 经验）在 4-8 GPU 上微调
  4. 通用 benchmark 为主战场（布料/面部/组织），手术数据仅为 zero-shot showcase
- **Hypothesis**: GFM 在连续形变下不是"噪声式失败"而是层局部化的证据丢弃→单目回退；误差随形变指数存在陡峭相变；针对性 adapter 比全量微调/重新架构更划算
- **Minimum experiment**: 冻结 VGGT+MASt3R 于 15 段 Hamlyn/StereoMIS 分箱序列，画误差 vs 形变指数曲线（<2 GPU·h）
- **Expected outcome**: 相变存在+confidence 测不到 → 完整三段论文；adapter 无效 → 诊断+benchmark 仍立得住
- **Novelty**: 8/10 — closest: PAGE-4D/Any4D（刚体运动 4D，不做连续形变）；EndoGS 等（逐场景优化非前馈）；"deformation adapter for GFM" 检索为空 ✅
- **Feasibility**: 诊断/benchmark 全冻结推理；adapter 训练 4-8 GPU 数周；数据全公开或作者已有
- **Risk**: MEDIUM（adapter 可能不收敛；PAGE-4D 团队可能正在做形变）
- **Contribution type**: diagnostic + benchmark + method
- **Pilot result**: SKIPPED: needs GPU
- **Reviewer's likely objection**: "形变数据稀缺，adapter 会不会只是过拟合 DeformingThings4D 的合成形变？"——用跨形变类型（布料训→组织测）泛化实验防御
- **Why we should do this**: 唯一同时满足「4D 最热赛道 + 已验证空白 + 作者独有形变先验/数据资产 + 失败兜底完整」的方向；比 GeoReliab 的赛道拥挤度低一档

### Idea 2: "Do Geometry-Injected Spatial MLLMs Actually Use Geometry?" — 便宜爆点（#5）
- **Method（我们实际做什么）**:
  1. 取开源几何注入式空间 MLLM（SpatialRGPT/SSR/GUIDE 类）及其无几何基座
  2. 在 VSI-Bench/3DSRBench 上以 {真深度, 场景打乱深度, 常数平面深度, 对抗扭曲深度} 四种输入对照评测
  3. 产出逐模型逐 benchmark 的 **geometry-reliance score**；同时反向审计哪些"空间推理" benchmark 不需要 3D 也能解
- **Hypothesis**: 相当比例的报告增益在打乱深度下依然保留 → 增益主要来自空间微调数据而非几何 grounding
- **Minimum experiment**: 1 个 checkpoint × 300-500 题 × 3 种深度条件（1-2 GPU·h）
- **Novelty**: 7.5/10 — closest: CVT-Bench (2603.21114 ✅) 审计视角环一致性，非输入通道因果性；GUIDE/SpatialStack/Dual-Pathway (均 ✅) 无此对照
- **Risk**: MEDIUM（若模型真的在用几何——结果不那么爆但仍是有价值的正面认证）| **Effort**: weeks
- **Pilot result**: SKIPPED: needs GPU
- **Reviewer's likely objection**: "只审计了 N 个开源模型，结论能否外推？"——把 reliance score 做成可复用协议来防御
- **Why**: 最便宜、最快出信号；可与 Idea 1 并行作为第二篇/备胎；正负结果都有明确叙事

### Idea 3: GeoReliab（v2 主线，降为 #3）
- 内容不变（见 `IDEA_REPORT_20260726_153000.md`）：GFM confidence 校准 under shift + 下游损害 + training-free 修复；可吸收 #1/#6/#3 候选增强
- **降位原因**: 可靠性赛道 3 个月内已出现第三篇相邻工作（2605.18754），拥挤度上升最快；相比之下聚类 α 的空白更干净、资产护城河更深
- **仍然成立的理由**: novelty 7.5/10 未变；若用户更偏好低执行风险（全冻结推理为主、无训练环节）则它仍是最稳选择

## Eliminated / Deprioritized Ideas
| Idea | 处置原因 |
|------|---------|
| #4+#8 视频深度指标审计（聚类β） | 扎实但"审计文"天花板取决于发现的杀伤力；Tier 2 候补 |
| #10 GFM vs SfM regime map | 好问题但计算面广、结论可能是"各对一半"；Tier 2 |
| #13 test-time compute scaling | 叙事时髦但发现可能偏薄；Tier 2 |
| #11 空间相关性 UQ 指标 | 受众窄（UQ 方法论社区）；可作 GeoReliab 附属发现 |
| #3 世界模型几何评分去污染 | 好切口但依赖世界模型公开输出的可得性；Tier 2 |
| #9 记忆化审计 | 相似度≠记忆化的因果攻击难防；Tier 3 |
| #12 spatial-g 因子分析 | venue-fit 风险（偏 psychometrics）；Tier 3 |
| #15 不确定性 NBV | 偏机器人会议口味；Tier 3 |

## Pilot Experiment Results
全部 SKIPPED（本仓库无 GPU 执行环境）。三个推荐 idea 的 pilot 均设计为 ≤2 GPU·h 冻结推理，见各 idea 的 Minimum experiment。

## Suggested Execution Order
1. **并行跑两个 MVE**（各半天）: Idea 1 形变分箱曲线 + Idea 2 单模型三条件对照 → 用真实信号定主攻
2. 信号均强 → **Idea 1 为主文**（贡献结构厚），Idea 2 为快速第二篇
3. Idea 1 的 adapter 环节若 6 周内不收敛 → 回落为"诊断+benchmark"文，或切换 GeoReliab（其代码资产完全复用）
4. 关键日期: 9 月中旬前完成主实验并挂 arXiv（两个方向的竞争者都在移动）

## Next Steps
- [ ] 决定主攻: Idea 1 (Deformable World) vs Idea 3 (GeoReliab) —— 或先跑两个 MVE 再定
- [ ] 配置 Codex MCP 后补跨模型评审
- [ ] 主攻确定后 → `/experiment-plan` 展开全量实验计划

---

## v3.1 决策附录（2026-07-26，新增约束后的最终建议）

**新信息**: 用户有进行中的 Science Robotics 论文（VP 预测世界模型 + 手术安全监督），其 Stage B 扩大实验与 Stage C ex vivo 将在未来数月与 CVPR 截稿（11月中旬）**争夺时间与算力**。用户策略：最大化中稿率、难度不要太大、CVPR 不必迁就 SR。

**权重修正**: 时间带宽受限 → floor（保底可发性）与执行负载的权重上调，ceiling 与 novelty 护城河的权重下调。

### 难度调整后对比

| 维度 | Idea 3 GeoReliab | Idea 1 Deformable World | Idea 2 Geometry审计 |
|------|-----------------|------------------------|-------------------|
| 执行负载 | **低**（几乎全冻结推理；probing 代码直接复用） | 中-高（含 adapter 训练 + 形变数据整备） | 最低（但单独成文偏薄） |
| Floor | **高**（正/负结果均可发；退化渲染管线成熟） | 中（相变不陡峭则叙事变薄；形变 GT 整备繁琐） | 中 |
| Ceiling | 中-高 | **高** | 中 |
| 赛道拥挤 | 上升中（3个月3篇相邻） | 低 | 低-中 |
| 与 SR 时间冲突 | **最小** | 最大 | 最小 |
| SR 方法论协同（免费） | 有（conformal calibration / applicability-boundary 同源） | 有（组织形变几何） | 无 |

### 最终建议
**主攻 Idea 3 GeoReliab**（在"最大化中稿率 + 难度可控 + SR 并行"的三重约束下 floor 最高），执行提速以对冲拥挤风险（9 月中旬 arXiv）。
**保留切换阈值**: 仍先花一天跑两个 ≤2 GPU·h MVE；当且仅当 (a) GeoReliab MVE 显示"模型出人意料稳健"（主发现落空）**且** (b) Deformable MVE 显示陡峭相变，才切换到瘦身版 Idea 1（诊断+benchmark 为主、training-free 缓解为辅、adapter 为 stretch goal）。
Idea 2 作为低成本备用（若主线中途受阻，6 周内可独立成文的逃生舱）。
