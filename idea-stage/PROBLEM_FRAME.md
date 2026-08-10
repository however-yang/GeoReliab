---
status: superseded_historical
authority: GEORELIAB_TASKBOOK.md
execution_entry: false
superseded_by: GEORELIAB_TASKBOOK.md#v2.2
---

> Historical supporting material retained for provenance. It is not an execution entry; the v2.2 mainline and current phase gates are defined by GEORELIAB_TASKBOOK.md.
# Robotics/CV Problem Frame — CVPR 论文选题

**日期**: 2026-07-26（v2 修订）
**目标会议**: CVPR 2027（预计截稿 ~2026年11月中旬，剩余约 3.5–4 个月）
**注意**: 用户明确目标是 CVPR（视觉会议），因此 pipeline 默认的 CoRL/RSS/ICRA 定位调整为 **CVPR 主会**，机器人元素作为应用/动机而非硬件贡献。

## v2 修订（用户反馈 2026-07-26）
- **不需要考虑水下识别/水下域** —— TII 只代表退化视觉经验，不作为必须对齐的资产
- **选题以视觉为中心**（用户主兴趣），医疗+机器人是博士领域但**论文不必完全重合、不必是医疗**
- 影响: 评估域从"水下+内窥镜双散射域"改为 **CVPR 主流 shift 域**（合成受控腐蚀、恶劣天气/夜间、低纹理、动态非刚性）；内窥镜降为**可选的 hard deployment domain**（复用代码资产、支撑博士叙事，但论文不依赖它）

## 用户研究轨迹（已确认资产）

| 论文 | 领域 | 核心能力 | 可复用资产 |
|------|------|---------|-----------|
| TII: Underwater Object Detection | 退化视觉环境感知 | 看见目标（散射/低能见度/色偏） | 水下检测 pipeline、退化图像处理经验 |
| T-RO: Surgical Monocular 3D (TAURUS-3D → SURE-3D) | 单目深度/重建 | 理解空间（非刚性、遮挡、镜面反光） | 内窥镜深度估计代码、uncertainty-aware depth |
| NeurIPS: Surgical Epistemic Probing | 认知不确定性评估 | 知道自己是否可靠 | epistemic evaluation framework、distribution-shift 协议 |
| AAAI: SurgRefuse / AKTM | Safe VLA / 拒绝执行 | 决定是否行动 | action monitor、refusal 机制 |

**博士主线**: Reliable Visual Intelligence — 从"看见"到"理解空间"到"知道可靠性"到"安全行动"。

## Problem Frame（逐字段）

- **Embodiment**: 无真实硬件依赖（默认假设）。虚拟具身：手术内窥镜相机、水下相机、仿真机械臂（LIBERO/CALVIN 级别）。**不做真机实验**。
- **Task family**: 感知层任务为主 — 单目深度/3D 重建、退化环境感知、感知可靠性评估、VLA 视觉骨干的失效预测。
- **Environment type**: 退化视觉环境（散射介质：水下 + 内窥镜共享光学物理——散射、吸收、镜面反光、低纹理）、分布偏移场景。
- **Observation modalities**: 单目 RGB 为核心；可选 RGB-D 监督信号。
- **Action interface**: 仅在 VLA 相关 idea 中涉及（离线评估，不闭环执行）。
- **Learning regime**: 视觉基础模型（Depth Anything v2 / UniDepth / Metric3D / DINOv2 / SAM 类）的分析、适配、不确定性建模；或评估协议/benchmark 构建。
- **Available assets**（假设，需用户确认）:
  - 公开数据集：SCARED、Hamlyn、EndoNeRF、C3VD、StereoMIS（内窥镜）；KITTI/nuScenes/DDAD（驾驶深度）；UIEB/SUIM/DUO（水下）；LIBERO/CALVIN/SimplerEnv（VLA 离线）
  - 已有代码库：手术深度估计 + epistemic probing 框架 + AKTM monitor
  - 计算预算：假设 4–8 张消费级/A100 级 GPU（**不足以从头训练基础模型**，足以做 probing / fine-tune / adapter / benchmark）
- **Safety constraints**: 无（不做真机）。
- **Desired contribution type**: CVPR 口味 — **method + analysis/benchmark 双保险**：方法为主贡献，评估协议/诊断发现为次贡献。避免纯 benchmark（CVPR 主会对 pure benchmark 接受率低，除非规模/发现极强）。

## CVPR 定位约束（与默认机器人会议的差异）

1. CVPR 需要 **视觉侧的技术新颖性**，不能只是"把 UQ 用在机器人上"。
2. 需要在 **公开、有竞争 baseline 的视觉 benchmark** 上验证。
3. 机器人/手术/水下作为 **动机与应用域**，主战场是通用视觉问题。
4. 4 个月时间窗 → idea 必须能复用现有代码资产，pilot 在 2–3 周内出信号。

## 默认假设（用户未指定时生效）

- simulation/offline-first，无真机
- 公开 benchmark 优先
- 单一 idea 主攻 + 1 个备胎
- 语言：中文报告，英文术语保留

## 候选交叉点（Phase 1 调研的先验假设，待文献验证）

- **H1**: 深度/几何基础模型的 epistemic reliability（它们知道自己何时失效吗？尤其在散射介质/非刚性场景）— 复用 T-RO + NeurIPS 资产
- **H2**: 跨退化域（水下↔内窥镜）统一的散射介质单目几何感知 — 复用 TII + T-RO 资产
- **H3**: VLA 视觉表征的失效可预测性（感知不确定性 → 动作失败预测）— 复用 NeurIPS + AAAI 资产
- **H4**: 面向下游安全决策的 uncertainty-aware 3D 表征（重建质量的自我评估）
