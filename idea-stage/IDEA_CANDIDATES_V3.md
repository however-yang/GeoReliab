# Idea Candidates v3 — 完全放开的视觉方向（5-lens fan-out，Tier 2 并行）

**日期**: 2026-07-26 | **生成**: 5 个 lens 子代理并行（method-transfer / untested-assumption / diagnostic / contradiction / scaling-regime），共 15 个原子候选
**验证**: 候选引用的全部 arXiv ID 已经 API 逐一验证 ✅
⚠️ **降级声明**: Codex MCP 不可用，Phase 4 跨模型 triage 为 Claude 同族自评审（audit-visible）

## 原子候选总表（15）

| # | dedup_key | lens | 一句话 | risk | 处置 |
|---|-----------|------|--------|------|------|
| 1 | training-free-abstention-selective-3d-gfm | transfer | GFM 内部信号（注意力熵/TTA分歧）做 training-free 选择性重建 | M | 并入 GeoReliab 家族 |
| 2 | deformation-adapter-feedforward-4d | transfer | 轻量 deformation adapter 让 4D 前馈模型处理连续形变 | M | **聚类 α 主干** |
| 3 | uncertainty-aware-geometric-consistency-world-models | transfer | 世界模型几何一致性评分被 probe 自身不确定性污染 | L | Tier 2 独立候选 |
| 4 | temporal-consistency-vs-accuracy-video-depth | assumption | 时序一致性≠精度：平滑攻击可刷高一致性指标 | L | **聚类 β 主干** |
| 5 | geometry-injection-corrupted-depth-control | assumption | 空间 MLLM 真的在用注入的深度吗？打乱深度对照审计 | M | **Tier 1 独立候选** |
| 6 | gfm-confidence-filtering-downstream-utility | assumption | confidence 阈值过滤 vs 等稀疏度对照的下游效用 | L | 并入 GeoReliab 家族 |
| 7 | gfm-rigidity-prior-localization | diagnostic | 刚性假设住在 GFM 哪一层？逐层归因 | M | **聚类 α 诊断章节** |
| 8 | video-depth-photometric-shortcut | diagnostic | 反事实解耦外观运动/几何运动：photometric shortcut index | L | **聚类 β 反事实章节** |
| 9 | gfm-benchmark-memorization-retrieval | diagnostic | 前馈重建有多少是检索？训练集相似度分层审计 | M | Tier 3（因果性攻击风险） |
| 10 | ff-gfm-vs-bundle-adjustment-regime-map | contradiction | GFM vs SfM 的 regime 地图 + 误差分解 + BA 可恢复性 | M | Tier 2 |
| 11 | pixelwise-depth-uq-vs-3d-spatial-correlation | contradiction | 逐像素 UQ 指标对空间相关性致盲，correlation-aware 指标重排座次 | L | Tier 2（受众偏窄） |
| 12 | spatial-benchmark-latent-factor-scaling | contradiction | 空间 benchmark 因子分析："spatial-g" 存在吗 | M | Tier 3（venue-fit 风险） |
| 13 | inference-compute-scaling-curves-3d-gfm | scaling | 3D 的 test-time compute scaling law + 不确定性路由分配 | L | Tier 2 |
| 14 | deformation-regime-phase-boundary-gfm | scaling | 形变幅度相变边界 benchmark；confidence 测不到越界 | M | **聚类 α benchmark 章节** |
| 15 | view-budget-scaling-uncertainty-nbv-frozen-gfm | scaling | 冻结 GFM 自身不确定性驱动的 next-best-view | M | Tier 3（偏机器人） |

## 聚类合并结果

### 聚类 α — "Deformable World"（#2 + #7 + #14）⭐ 新晋最强
诊断（刚性先验逐层归因）+ benchmark（形变幅度相变边界，含 confidence 失效）+ 修复（deformation adapter，4-8 GPU 可训）三合一。
- **空白已验证**: 手术/内窥镜重建仍停留在逐场景优化（EndoGS/SAGS/BASED），无 3R 式前馈模型；4D 前馈模型（PAGE-4D 2510.17568 / Any4D 2512.10935）只处理刚体运动，连续形变（组织/布料/面部）无人做；"deformation adapter for 4D GFM" 检索为空
- **资产护城河**: 连续形变的先验知识、数据管线（StereoMIS/Hamlyn/SCARED stereo GT）、形变正则化经验——全部来自 T-RO 工作，但 headline 是通用视觉（DeformingThings4D/DyCheck/布料），手术仅为 zero-shot showcase
- **竞争风险**: PAGE-4D/Any4D 团队可能下一步就做形变——与 GeoReliab 同样有时间压力

### 聚类 β — "Gameable Metrics"（#4 + #8）
视频深度评估审计：时序一致性 vs 精度弱相关/负相关 + 平滑攻击 + 反事实外观/几何解耦 + 修正指标。LOW risk，全冻结模型推理。

### 独立 Tier 1 — #5 "Geometry-Reliance Audit"
打乱/常数/对抗扭曲深度对照，审计几何注入式空间 MLLM（GUIDE 2604.05695 / SpatialStack 2603.27437 / Dual-Pathway 2605.25334 均无此对照；CVT-Bench 2603.21114 审计的是视角环一致性而非输入通道因果性——差异化明确）。便宜（≤2 GPU·h/模型）、爆点足、正负结果皆可发。

### GeoReliab 家族（上轮成果 + #1/#6/#3 增强）
仍然成立（novelty 7.5/10），但可靠性赛道竞品在密集出现（Trust3R、VGGT-UQ、2605.18754 metric-reliability）——先发压力最大。
