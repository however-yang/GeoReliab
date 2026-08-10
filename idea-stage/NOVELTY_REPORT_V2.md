---
status: superseded_historical
authority: GEORELIAB_TASKBOOK.md
execution_entry: false
superseded_by: GEORELIAB_TASKBOOK.md#v2.2
---

> Historical document retained for provenance. It is not an execution entry; the v2.2 mainline and all current phase gates are defined by GEORELIAB_TASKBOOK.md.
# Novelty Check Report — GeoReliab 重构主线 "Ranking Is Not Warning"

**日期**: 2026-07-26
**对象**: 外部方案文档（重构 GeoReliab 为 counterfactual warning + task-transfer reliability paper）
⚠️ **Phase C 降级**: Codex MCP 不可用，跨模型验证未执行；本报告为 Claude 单模型判断。

## Proposed Method
同场景配对反事实协议（DTU 7-lighting / UAVLight 真实配对 + TartanAir 合成可控）下，检验 GFM 原生 confidence 能否在任务失败前发出预警（CRR/SFR/Boundary Lag 指标体系），并测点级 confidence 向 pose/fusion 的可靠性迁移。

## 引用真实性审计（该文档来自外部 LLM，带 citeturn 残留 → 全部人工复验）

| 文档引用 | 验证结果 |
|---------|---------|
| E3D-Bench | ✅ [2506.01933](https://arxiv.org/abs/2506.01933)，真实 |
| Trust3R | ✅ 2605.19539（先前已验证） |
| Uncertainty Quality of VGGT | ✅ 2606.16479（"ISPRS 2026" 出处未验证，arXiv 实锤） |
| Can These Views Be One Scene? | ✅ 2605.18754。⚠️ 文档称其 "SysCON3D"——摘要中 benchmark 名被遮蔽，**该名称未经验证，勿在论文中使用** |
| On Geometric Understanding... | ✅ [2512.11508](https://arxiv.org/abs/2512.11508)，真实 |
| UAVLight | ✅ [2511.21565](https://arxiv.org/abs/2511.21565)，**CVPR 2026 正式论文**（openaccess 确认） |
| MapAnything | ✅ [2509.13414](https://arxiv.org/abs/2509.13414)，真实 |
| DTU 7 lighting × 49/64 位姿 | ✅ DTU 官方数据集属性，成立 |
| CVPR 2026 деadline 11/7 + 11/13、CVPR 2027 Seattle | 未逐一验证，低风险 |

**审计结论**: 引用全部真实，无幻觉。另发现文档未提的相邻工作 **UAVFF3D** ([2605.17942](https://arxiv.org/abs/2605.17942))：UAV 相机几何变化的前馈重建 benchmark——与 NC1 互补不冲突，必须引用。

## Core Claims 判定

| # | Claim | Novelty | Closest Prior（均已验证） | Delta |
|---|-------|---------|--------------------------|-------|
| NC1 | 同场景 geometry-preserving 配对反事实协议测 confidence 风险响应 | **HIGH** | UAVLight（配对光照数据，但评的是重建鲁棒性非 confidence 预警）；VGGT-UQ（静态、单条件） | 协议目的不同：数据可复用，问题是新的。⚠️ 待办：精读 VGGT-UQ 确认其未用 DTU 的 7 lighting 变体 |
| NC2 | Ranking–Warning Gap 指标体系（CRR/SFR/Lag） | **MEDIUM-HIGH** | **Ovadia et al. 2019** ([1906.02530](https://arxiv.org/abs/1906.02530))——分类任务上"不确定性随 shift severity 的响应"正是该文经典问题；failure-prediction/misclassification-detection 文献族 | 差异真实：paired 同场景（非跨数据集）、几何任务、任务级失败事件、lag 操作化。**但 Ovadia 必须在 related work 正面引用并划界，否则内行审稿人一击即中** |
| NC3 | 点级 confidence → pose/fusion 的跨任务可靠性迁移协议 | **HIGH** | Trust3R（点级 UQ + 融合加权使用，但从未系统问"点 confidence 能否预测 pose/fusion 失败"）；CullNet（物体位姿 confidence，范式不同） | 检索为空白，成立 |
| NC4 | 真实配对 + 合成可控双证据图谱 | MEDIUM（实验设计优点，非独立新颖性主张） | UAVLight/DTU 数据本身 | 文档已正确声明不抢"首发"，定位为协议载体 ✅ |

## Overall Novelty Assessment

- **Score: 8/10 — PROCEED**（重构版在新颖性轴上优于原 B1-B4 计划：原计划的 headline "confidence 在退化下失效" 确实已被 Trust3R/VGGT-UQ/E3D-Bench/2605.18754 四面围拢）
- **Key differentiator**: "会排序 ≠ 会报警" + "点级 ≠ 系统级" 的双分离命题——现有近邻全部停留在静态排序层
- **Risk（审稿人会引什么）**:
  1. **Ovadia et al. 2019**（NC2 的概念先例，最大威胁）——防御：paired same-scene / 几何 / lag 操作化三重差异必须写透
  2. VGGT-UQ 若已用 DTU lighting 变体（待精读确认）会侵蚀 NC1
  3. "纯评测无方法" 的 CVPR 口味风险——见下方整合建议

## Suggested Positioning（与原计划的整合建议）

1. **采纳重构主线**：counterfactual warning + task transfer 作为双支柱，标题方向 "Ranking Is Not Warning"
2. **DTU 7-lighting 是本次重构最value的发现**：真实配对反事实 = 零渲染工程 + 同时击破 W1（GT噪声）与 R3（"renders not reality"）两个旧风险——原计划的 Koschmieder 渲染管线降为 TartanAir 补充轴
3. **保留一个 training-free 修复作为可选 C5'**（原 B4 的单遍特征评分，非学习型）：对冲"纯评测文"的 CVPR 接受率风险；它不是 learned head（不落 Trust3R 陷阱），且若 gap 成立它就是第一个"修复 warning 失效"的信号——文档反对的是 learned transport，training-free 版不冲突
4. **等稀疏度下游对照（原 B3）压缩为 actionability 层的一个实验**，不再是独立 Block
5. 引用义务清单: Ovadia 2019、UAVLight、UAVFF3D、Trust3R、VGGT-UQ、E3D-Bench、2605.18754（勿用 "SysCON3D" 名）、2512.11508、Test3R
6. MVE 的 GO/NO-GO 门设计合理，采纳；建议 GO-1 的双指标阈值（Spearman≥0.35 / CRR≤0.15）在 MVE 前用 20 场景 pilot 校准一次数值合理性
