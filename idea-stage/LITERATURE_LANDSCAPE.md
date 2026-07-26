# Literature Landscape — Reliable Visual Intelligence (CVPR 2027 选题)

**日期**: 2026-07-26 | **来源**: arXiv API + WebSearch + web-search-prime（Zotero/Obsidian/本地PDF不可用，跳过）
**验证状态**: 下表所有 arXiv ID 均通过 arXiv API 逐一验证 ✅（Step 1.5 反幻觉门通过）

## Cluster A — 2D 深度基础模型的不确定性 (H1-2D)

| Paper | Venue | 方法 | 与我们的关系 | 状态 |
|-------|-------|------|------------|------|
| A Critical Synthesis of UQ and Foundation Models in MDE ([2501.08188](https://arxiv.org/abs/2501.08188)) | arXiv 2025-01 | 5种UQ方法 × DepthAnythingV2，GNLL fine-tune 最优 | **直接占位**："给深度基础模型加UQ"的朴素形式已做 | ✅ verified |
| Robust-MonoDepth benchmark (IJCV'24) | IJCV | ImageNet-C 式 15 corruptions × 6 数据集 | 只测精度退化，**不测校准/自知** | ✅ (DOI) |
| PDE: Procedural Depth Evaluation | NeurIPS 2025 | 程序化3D场景扰动测鲁棒性 | 同上，无 reliability 轴 | ✅ (OpenReview) |
| OOD Detection for MDE ([2308.06072](https://arxiv.org/abs/2308.06072)) | arXiv 2023 | 图像级OOD检测 | 早期工作，粒度粗（图像级非像素级） | ✅ verified |

**结论**: 2D 深度 UQ 方法层面基本覆盖；**"校准是否在分布偏移下成立"仍是公开缺口**（多个综述明确指出）。

## Cluster B — 3D 几何基础模型 (VGGT/DUSt3R 族) 的可靠性 ⭐ 核心战场

| Paper | Venue | 方法 | 与我们的关系 | 状态 |
|-------|-------|------|------------|------|
| VGGT ([2503.11651](https://arxiv.org/abs/2503.11651)) | **CVPR 2025 Best Paper** | 前馈网络直接输出 pose+depth+pointmap | 目标模型；其 confidence 是启发式的 | ✅ verified |
| Trust3R ([2605.19539](https://arxiv.org/abs/2605.19539)) | arXiv 2026-05 | NIW evidential head + gated residual refinement → 逐点 Student-t 不确定性，**需训练** | **最强竞品**：方法层面占了"给3D GFM加概率不确定性"，但未涉及分布偏移/退化域/下游决策 | ✅ verified |
| Uncertainty Quality of VGGT ([2606.16479](https://arxiv.org/abs/2606.16479)) | arXiv 2026-06 | 分析 VGGT confidence 质量 | **仅 DTU、仅干净数据、纯分析**。证明社区刚意识到该问题 | ✅ verified |
| E3D-Bench | 网页 | 3D GFM 综合 benchmark（5任务） | 无 reliability/calibration 轴 | ✅ (site) |
| PAGE-4D / VGGT-X / HD-VGGT / Mamba-VGGT / SwiftVGGT / VGGT-Long | 2025-2026 | 能力扩展竞赛（动态、高分辨率、长序列） | 全部在卷"能力"，**无人做"可靠性"** | ✅ verified (抽查) |

**结论**: 能力竞赛白热化，可靠性研究刚萌芽（1篇分析+1篇训练式方法）。**空白点：(a) 分布偏移/退化域下的 confidence 校准评估；(b) training-free/test-time 的可靠性修复；(c) selective/abstention 式几何预测（risk-coverage 保证）；(d) 不确定性驱动的下游决策**。

## Cluster C — 退化域深度估计（用户主场）

| Paper | Venue | 方法 | 与我们的关系 | 状态 |
|-------|-------|------|------------|------|
| Underwater Monocular Metric Depth ([2507.02148](https://arxiv.org/abs/2507.02148)) | arXiv 2025-07 | 水下 metric depth benchmark (FLSea/SQUID) + VFM fine-tune | 水下深度 benchmark 已有；无 reliability 轴 | ✅ verified |
| Physics-Informed Underwater Depth (ECCV 2024) | ECCV | 物理引导知识迁移 | 散射物理先验可复用 | ✅ (Springer) |
| EndoGMDE ([2509.01206](https://arxiv.org/abs/2509.01206)) | arXiv | MoLE experts 泛化内窥镜深度 | 内窥镜泛化已卷（EndoDAC/EndoUFM/Surgical-DINO 同类） | ✅ verified |
| Endoscopic Depth Robustness ([2409.16063](https://arxiv.org/abs/2409.16063)) + EndoDepth ([2409.19930](https://arxiv.org/abs/2409.19930)) | arXiv 2024 | 内窥镜 corruption 鲁棒性 benchmark | 只测精度退化，不测自知 | ✅ verified |
| ER-LoRA / Always Clear Depth (IJCAI'25) / WeatherDepth | 各处 | 天气退化域适配 | 单域适配套路成熟 | ✅ |

**结论**: 各退化域**各自为战**（水下/内窥镜/天气三个社区互不引用）；无统一散射介质框架；所有 benchmark 都缺 reliability 轴。

## Cluster D — VLA 失效检测 ⚠️ 已饱和

SAFE ([2506.09937](https://arxiv.org/abs/2506.09937), NeurIPS'25)、VLA-FAIL ([2606.21386](https://arxiv.org/abs/2606.21386))、Perturbation-UQ ([2606.20754](https://arxiv.org/abs/2606.20754))、Tri-Info、ReconVLA、Pre-VLA、VLAConf、Hide-and-Seek ([2605.30834](https://arxiv.org/abs/2605.30834))、Shifting Uncertainty ([2603.18342](https://arxiv.org/abs/2603.18342))、UNISafe (CoRL'25)、Failing Forward——仅 2026 上半年就 10+ 篇。

**结论**: **避开作为主线**。用户 AAAI 工作已在此空间；CVPR 也非该方向主场。可作为 motivation/下游应用引用。

## 可乘之隙（Gap 总结，按可行性排序）

1. **G1 — 3D GFM 的失效自知（epistemic reliability under shift）**: VGGT-class 模型的 confidence 在退化域（散射介质/低纹理/动态）下是否仍有意义？没人测过；用户 NeurIPS epistemic probing 方法论可直接迁移。分析 + training-free 修复 + benchmark 三合一，CVPR 口味。
2. **G2 — Selective 3D Reconstruction（几何层面的拒绝）**: 像素/点级 abstention + risk-coverage 保证，把用户 AAAI 的 refusal 思想从动作层下沉到几何感知层。检索显示**基本空白**。
3. **G3 — 跨散射介质统一几何感知**: 水下+内窥镜双域是用户独有资产组合，但单独做容易被视为"应用适配"，更适合作为 G1/G2 的评估域而非独立贡献。

## 已饱和/避开

- VLA failure detection（Cluster D）
- "给 DepthAnything 加 UQ"（2501.08188 占位）
- "给 3D GFM 加训练式 evidential head"（Trust3R 占位）
- 单域适配 adapter（EndoDAC/EndoGMDE/ER-LoRA 套路化）
