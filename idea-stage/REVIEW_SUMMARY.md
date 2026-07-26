# Research Review Summary — IDEA-A

**日期**: 2026-07-26
⚠️ **降级声明**: Codex MCP 未配置，跨模型评审未执行。以下为 Claude 同族自我对抗评审——**不能替代跨模型验证**。brief 已备好（RESEARCH_REVIEW_REQUEST.md），配置 Codex 后应重跑 `/research-review` 并以其结论为准。

## 对抗评审发现（按杀伤力排序）

### W1 — GT 质量悖论（benchmark 的最尖锐攻击）🔴
退化域数据集的 GT 本身有噪：SCARED 是猪尸体结构光+插值位姿、C3VD 是硅胶假体、FLSea 的 GT 来自 SLAM/立体。审稿人：**「你的 miscalibration 指标里有多少是 GT 误差贡献的？」**
**修复（必须做）**: 加一条**受控合成退化轴**——对 GT 完美的干净数据（DTU/Hypersim/TartanAir）施加物理散射渲染（水下衰减模型/内窥镜光照模型），得到"GT 完美 + 退化可控可分级"的主实验；真实退化数据降为 secondary 验证。这同时给出 severity-ramp 分析（退化强度 vs 校准坍塌曲线）——更强的实验设计。

### W2 — 「OOD 过信众所周知」的 triviality 攻击 🔴
审稿人：「NN 在 OOD 上过信是常识（Hein et al. 2019 起），发现不惊人。」
**防御（必须做）**: 把贡献从「校准指标坍塌」升级为**「下游损害量化」**——展示信任 confidence 做过滤/融合的下游管线（点云过滤、NVS、SLAM 初始化）在退化域被 confidence 误导的实际精度损失。「社区每天在用这个 confidence 过滤点云，我们量化了这个信任的代价」——这是 CVPR 叙事，不是 UQ 叙事。

### W3 — 立体/MVS confidence 文献被忽略 🟠
stereo confidence estimation 有 15 年文献（Poggi/Mattoccia 系、learned confidence measures）。审稿人若来自该社区必问。
**修复**: related work 必须覆盖，并把 2-3 个经典 stereo confidence 度量作为移植 baseline。

### W4 — 方法成本攻击 🟠
视图重采样集成 = N 次前向；VGGT 大场景下昂贵，Trust3R 单次。
**修复**: 双档方法——单次前向的特征空间评分（cheap）+ 集成（expensive upper bound），给出效率-质量 Pareto 表。

### W5 — 混合单目/多视图模型混淆变量 🟡
MoGe 是单目、VGGT/DUSt3R 多视图；内窥镜小基线是"几何 regime"混杂而非"退化"。
**修复**: 主张收窄为多视图 GFM（VGGT/DUSt3R/MASt3R/CUT3R）；单目模型移到附录；小基线单独作为一个受控变量分析而非混入散射族。

### W6 — 「规模不买自知」finding 可能不可执行 🟡
VGGT 公开 checkpoint 规模档位有限。**降为 optional/附录**，不写进承诺贡献。

### W7 — conformal「保证」措辞风险 🟡
像素误差空间相关 → marginal 保证语义弱。**修复**: 第二贡献措辞从 "guarantees" 软化为 "risk-coverage evaluation protocol"；只在 image-level 或 block-level 交换性可辩护处使用 conformal 术语。

### W8 — 时间窗竞争 🟠
VGGT-UQ (2606.16479) 作者显然会扩展；Trust3R camera-ready 可能补 shift 实验。**对策**: 9 月前完成核心实验并挂 arXiv 占坑。

## 最小可信证据包（Minimum Viable Evidence Package）
1. **模型**: VGGT + MASt3R + CUT3R（3个多视图 GFM，MoGe/DAv2 附录）
2. **域**: 受控合成散射（DTU/Hypersim 基底，severity 3 级）为主轴 + SCARED + FLSea 真实域验证 + 动态/低纹理各一
3. **指标**: AUSE、ECE-depth、failure-AUROC、risk@coverage + **下游损害**（confidence-filtered 重建的 Chamfer/F-score 损失）
4. **Baselines**: 自带 confidence、GNLL 移植、MC-Dropout、光度一致性、**2 个 stereo confidence 经典度量**、Trust3R（可得则加）
5. **方法**: cheap(单遍特征评分) + full(重采样集成)，效率表，逐组件 ablation
6. **失败案例分册**: 每退化族 top-losing cases 可视化

## Mock CVPR Review（自评审模拟）
- **Summary**: 系统评估前馈 3D GFM 的 confidence 可靠性并给出 training-free 修复
- **Strengths**: 及时（VGGT 生态爆发）、双真实退化域少见、发现对部署者有直接价值、分析+方法+benchmark 完整
- **Weaknesses**: 方法组件借自 TTA/ensemble（如 W4 防御不力）；GT 噪声混杂（如 W1 不做受控轴）；OOD 过信不惊人（如 W2 不做下游损害）
- **预估**: 执行到位 → borderline-accept ~ accept（4.0-4.5/6）；W1/W2 缺失 → borderline-reject
- **What moves to accept**: 受控退化主轴 + 下游损害量化 + 效率可用的 cheap 方法档

## 共识结论
IDEA-A 主线成立，但必须吸收 W1/W2 两条结构性修改：**受控合成退化为主轴 + 下游损害叙事**。修改后风险降为 LOW-MEDIUM。
