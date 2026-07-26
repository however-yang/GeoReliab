# Pilot Plan — 最小验证包（Phase 3）

**日期**: 2026-07-26 | **原则**: offline-first，无真机，全部使用公开数据集与冻结模型
**本仓库无现成执行环境** → 按 pipeline 规则产出具体 pilot 计划而非强行执行。

---

## IDEA-A: Do 3D Geometry Foundation Models Know When They Fail?

- **Embodiment**: 虚拟（内窥镜相机 / 水下相机 / 通用多视图相机）；无真机
- **Benchmark / simulator**:
  - 干净域: DTU, ETH3D, ScanNet（GT depth/pose 齐全）
  - 腐蚀域: Robust-MonoDepth 式 15-corruption 套件叠加于上述数据
  - 散射域: SCARED / Hamlyn / C3VD（内窥镜，GT 来自结构光/CT）、FLSea / SQUID（水下，metric GT）
  - 动态域: 任选 Bonn Dynamic / TUM dynamic 子集
- **Baselines**:
  - GFM 自带 confidence（VGGT / DUSt3R / MASt3R / MoGe，各自的 heuristic head）
  - Trust3R（若代码可得；不可得则以 GNLL fine-tune 复现代理）
  - 2D UQ 移植: MC-Dropout、GNLL fine-tune（来自 2501.08188 的最优法）
  - 朴素信号: 光度重投影误差、Laplacian blur score
- **Pilot type**: offline（冻结模型推理 + 指标计算）
- **Compute estimate**: pilot 阶段 ≤ 8 GPU·h（单卡 24GB 足够，VGGT 推理为主）；全文实验约 200-300 GPU·h
- **Human/operator time**: 数据准备 2-3 天（SCARED 申请协议注意）
- **Success metrics**: AUSE↓, failure-detection AUROC↑, ECE-depth, risk@coverage
- **Failure metrics**: 逐退化族的 confidence-error 相关系数、灾难性过信率（高confidence×大误差像素占比）、失败案例目录
- **Safety concerns**: 无（离线）
- **正信号判据**: 任一 GFM 在散射域 confidence-error Spearman ρ 相比干净域下降 >50%，且 training-free 信号将 AUROC 拉回 ≥0.15
- **可发表的负结果**: 若 confidence 出人意料地稳健 → "Surprisingly Calibrated" 分析论文路线（改写 framing，benchmark 贡献不变）
- **Pilot（第1周即可跑，v2 修订: 去水下）**:
  1. VGGT 冻结推理: DTU 干净 200帧 + DTU 施加合成雾/低照度（2 个 severity）+ nuScenes-night 200帧
  2. 画 confidence-error 散点 + sparsification 曲线
  3. 判据: 退化档位上 AUSE 相对干净 DTU 的恶化幅度、confidence-error 相关系数随 severity 的衰减曲线
  4. 预算: 1 GPU × 半天（合成退化用现成物理模型脚本，无需训练）
  5. （可选）SCARED 200帧作为 hard-deployment 附加点——不阻塞主线

## IDEA-B: Selective 3D Reconstruction with Risk-Coverage Guarantees

- **Embodiment**: 同上
- **Benchmark**: 同 IDEA-A 域集（复用数据管线——两个 idea 共享 70% 基础设施）
- **Baselines**: GFM 自带 confidence 阈值化、Trust3R 不确定性阈值化、softmax-entropy 式朴素评分、非选择性 full-coverage 基线
- **Pilot type**: offline
- **Compute estimate**: pilot ≤ 6 GPU·h；全文约 150-250 GPU·h
- **Success metrics**: risk@coverage 曲线 AUC、目标风险 α∈{5%,10%} 下实际风险违反率 ≤ α+1%
- **Failure metrics**: 强 shift 下 guarantee 违反率、coverage 崩塌点、块级交换性检验
- **正信号判据**: split-conformal 在同域上保证成立（违反率≤α）且 coverage ≥60%；跨域违反可控或可用 shift-aware 变体修复
- **可发表的负结果**: 空间相关性/shift 导致朴素 conformal 失效的系统性刻画 → 本身即贡献（指出社区需要 geometry-specific conformal）
- **Pilot（第1-2周）**:
  1. 复用 IDEA-A pilot 的推理输出
  2. 3 种 post-hoc 评分 × DTU split-conformal 校准 → risk-coverage 曲线
  3. 判据: 至少一种评分显著优于自带 confidence（AUC 差 >5%）
  4. 预算: 1 GPU × 半天（大部分是 CPU 统计计算）

## IDEA-C（备胎，不设 pilot）
标记为 `hold`；若 A/B novelty-check 双双失败再启动。

## 执行顺序建议
1. **Week 1**: IDEA-A pilot（同时为 B 产出推理缓存）
2. **Week 2**: IDEA-B pilot（复用缓存）
3. 两个 pilot 共享基础设施 → 实际上是一次数据准备、两次分析，边际成本低
4. **决策点**: pilot 后若 A 的"坍塌"信号强 → A 为主文，B 的 selective 机制并入 A 的 method 章节做次贡献；若 A 信号弱（模型稳健）→ B 升为主文
