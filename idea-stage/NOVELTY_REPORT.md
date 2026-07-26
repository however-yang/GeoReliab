# Novelty Check Report

**日期**: 2026-07-26
**验证方式**: arXiv API 多查询 + web-search-prime；所有引用 ID 经 arXiv API 验证 ✅
⚠️ **Cross-model verification (Codex) 不可用** — 本报告为 Claude 单模型判断，降级标注；建议用户有条件时用外部 LLM 复核一次。

## Proposed Method (IDEA-A)
系统评估前馈几何基础模型（VGGT/DUSt3R/MASt3R/MoGe）的 confidence 在分布偏移下的可靠性，提出 training-free test-time epistemic 信号恢复失效检测，附 GeoReliab benchmark。

## Core Claims 逐项判定

| # | Claim | Novelty | Closest Prior | Delta |
|---|-------|---------|---------------|-------|
| 1 | 首个跨退化域 3D GFM confidence 可靠性系统评估 | **HIGH** | VGGT-UQ ([2606.16479](https://arxiv.org/abs/2606.16479)) ✅ | 它仅 DTU 干净数据、单模型、纯分析；我们 5 退化族 × 4-5 模型 + 修复方法 |
| 2 | confidence-error 相关性在 shift 下坍塌（finding） | **HIGH**（取决于实验结果） | 2D 深度域有类似观察 ([2501.08188](https://arxiv.org/abs/2501.08188)) ✅ | 3D GFM 域无人报告；多视图误差相关性使 2D 结论不可外推 |
| 3 | training-free 视图重采样集成 + 特征距离 epistemic 信号 | **MEDIUM** | **Test3R** ([2506.13750](https://arxiv.org/abs/2506.13750), NeurIPS'25) ✅ 用 cross-pair consistency 做 test-time *training*；TTA/deep ensemble 是经典 UQ 机制 | 机制存在但用途不同：Test3R 用一致性**修模型**，我们用不一致性**测自知**并校准评估；无人做过 frozen GFM 的 failure-detection 评估 |
| 4 | (IDEA-B) 空间感知 conformal 选择性 3D 重建 | **HIGH** | ReconVLA ([2604.16677](https://arxiv.org/abs/2604.16677)) ✅ conformal 用于动作 token；conformal depth/pointmap 检索为空白 | 几何域 + 空间相关性处理 + abstention 框架均为新 |

## Closest Prior Work（全部经 arXiv API 验证）

| Paper | Date | 类型 | Overlap | Key Difference |
|-------|------|------|---------|----------------|
| Trust3R (2605.19539) | 2026-05 | method | 给 GFM 概率不确定性 | **需训练** evidential head；未测 shift；无 abstention/保证；无退化域 |
| VGGT-UQ (2606.16479) | 2026-06 | analysis | 分析 VGGT confidence 质量 | 仅 DTU、仅干净、无方法、无跨模型 |
| Test3R (2506.13750) | 2025-06 | method | cross-pair consistency 机制 | 用于 test-time training 提精度，非不确定性估计 |
| 2D-UQ Synthesis (2501.08188) | 2025-01 | analysis | UQ × 深度基础模型 | 2D 单目、单视图、非 GFM |
| EndoDepth / 内窥镜鲁棒性 (2409.19930, 2409.16063) | 2024-09 | benchmark | 退化域深度评估 | 只测精度退化，无 reliability/校准轴、非 3D GFM |
| Underwater Metric Depth (2507.02148) | 2025-07 | benchmark | 水下深度评估 | 同上 |

## Overall Novelty Assessment

### IDEA-A: **7.5/10 — PROCEED**
- **Key differentiator**: 「评估退化域下的自知能力 + training-free 修复」三合一；双散射域（水下+内窥镜）数据资产是作者独有组合；epistemic probing 方法论直接迁移自作者 NeurIPS 工作，可信度高。
- **Risk（审稿人会引什么）**: ① Trust3R 若在 camera-ready 补 shift 实验会压缩 delta；② VGGT-UQ 作者大概率在扩展其分析——**时间窗紧迫，建议尽快占坑（先挂 arXiv）**；③ Claim 3 会被指「TTA ensemble 不新」——必须把贡献重心放在 *finding + benchmark*，方法定位为 "simple strong fix"。
- **若方法部分无增益**: finding + benchmark 仍构成完整论文（类似 "surprisingly brittle/robust" 类 CVPR 分析文）。

### IDEA-B: **8/10 (method novelty) — PROCEED WITH CAUTION**
- 空白确认，但理论风险实在：像素级误差空间相关 → 朴素 split-conformal 的 marginal 保证语义弱，需要 block/functional conformal 设计，4 个月窗口内理论+实验都要打磨。
- **建议**: 作为 IDEA-A 的第二贡献章节（selective evaluation 协议）先行落地；若信号极好再拆成独立论文投后续会议。

## Suggested Positioning（IDEA-A）
- 标题方向: "Do 3D Geometry Foundation Models Know When They Fail?" / "GeoReliab: Benchmarking and Restoring the Reliability of Feed-Forward 3D Reconstruction"
- Framing: **不要**写成 "UQ for VGGT"（撞 Trust3R）；写成 **「下游生态盲目信任 GFM confidence——我们证明这个信任在真实部署域是错的，并给出即插即用修复」**——安全/部署叙事恰好接上作者 NeurIPS/AAAI 主线。
- 引用策略: Test3R 必须主动引用并在 related work 划清界限（consistency-for-accuracy vs inconsistency-for-self-knowledge）。
