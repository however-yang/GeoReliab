---
status: superseded_historical
authority: GEORELIAB_TASKBOOK.md
execution_entry: false
superseded_by: GEORELIAB_TASKBOOK.md#v2.2
---

> Historical document retained for provenance. It is not an execution entry; the v2.2 mainline and all current phase gates are defined by GEORELIAB_TASKBOOK.md.
# Idea Candidates — CVPR 2027（lens fan-out 全集）

**日期**: 2026-07-26
**生成方式**: 5-lens 单遍生成（Tier 3）。⚠️ Codex MCP 不可用，跨模型 triage 降级为同族自评审（audit-visible，Phase 4/5 由 novelty-check + research-review 部分补偿）。
**过滤规则**: 仅 budget 客观不可行才淘汰；质量判断交给后续 jury。

---

## Lens: untested-assumption（人人默认、无人检验）

### C1. 「3D 几何基础模型的 confidence 是可信的」——从未被检验 ⭐
- **dedup_key**: `gfm-confidence-calibration-under-shift`
- **Summary**: 系统性检验 VGGT/DUSt3R/MASt3R/MoGe 输出的 confidence 在分布偏移（散射介质/低纹理/动态/模糊）下是否仍与误差相关，预期发现「confidence-error 相关性坍塌」，并给出 training-free 修复。
- **Hypothesis**: GFM 的启发式 confidence 在干净数据上与误差弱相关，在退化域下相关性坍塌甚至反转；下游所有用 confidence 做过滤/融合的工作（VGGT-X 等）因此不可靠。
- **MVE**: 冻结 VGGT，在 DTU（干净）vs SCARED（内窥镜）vs FLSea（水下）各取 200 帧，画 confidence-error 散点 + sparsification curve（AUSE）。1 GPU 半天。
- **prior_work**: VGGT-UQ analysis (2606.16479) 仅 DTU 干净数据；Trust3R (2605.19539) 训练式方法未测 shift。差异点明确。
- **so_what**: 若坍塌成立→整个下游生态的地基有裂缝，正负结果都重要；若不坍塌→同样是重要负结果。
- **contribution**: diagnostic + benchmark + method
- **risk**: LOW（分析部分必出结果）| **effort**: weeks | **真机**: 否

### C2. 「多视图越多越好」假设检验
- **dedup_key**: `view-count-monotonicity`
- **Summary**: 检验 GFM 精度/置信度是否随视图数单调提升，尤其在退化输入下坏视图是否污染全局。
- **prior_work**: VGGT-Long/VGGT-X 部分触及序列长度问题。**降级理由候补**：范围小，适合作为 C1 的一个实验而非独立论文。
- **risk**: MEDIUM | **effort**: days | **真机**: 否

## Lens: method-transfer（A域方法未在B域尝试）

### C3. Selective 3D Reconstruction：把「拒绝执行」从动作层下沉到几何层 ⭐
- **dedup_key**: `selective-geometry-risk-coverage`
- **Summary**: 为前馈几何模型引入像素/点级 abstention：post-hoc 不确定性评分 + split-conformal 校准，输出带 risk-coverage 保证的「认证重建」，模型只重建它有把握的部分。
- **Hypothesis**: 在给定目标风险 α 下，selective 重建能以可控 coverage 换取误差保证，且在退化域下 coverage 自动收缩（=正确的自知行为）。
- **MVE**: 冻结 VGGT + 3 种 post-hoc 评分（view-resampling disagreement / feature-space distance / photometric consistency），DTU 上做 risk-coverage 曲线，对比 VGGT 自带 confidence。1 GPU 1 天。
- **prior_work**: 检索显示 conformal/selective prediction for pointmap geometry 空白；最近邻 Trust3R（训练式、无保证、无 abstention 框架）、ReconVLA（conformal 用在动作 token 非几何）。
- **so_what**: 给下游（SLAM/NVS/手术导航）第一个带统计保证的几何信任接口。
- **contribution**: method + analysis
- **risk**: MEDIUM（conformal 在空间相关误差下的有效性需处理 exchangeability）| **effort**: weeks | **真机**: 否

### C4. 散射物理先验 → 失效预测
- **dedup_key**: `scattering-physics-failure-prior`
- **Summary**: 用散射介质物理模型（衰减/后向散射参数）作为自监督信号预测几何失效区域。
- **prior_work**: Physics-informed underwater depth (ECCV'24) 用物理做迁移而非失效预测。
- **so_what**: 有趣但域窄，CVPR 会问「为什么只有散射介质」。适合作为 C1/C3 的域特定评分函数。
- **risk**: MEDIUM | **effort**: weeks | **真机**: 否

## Lens: contradiction（文献矛盾）

### C5. 2D UQ 结论不迁移到多视图 3D
- **dedup_key**: `2d-uq-conclusions-break-in-3d`
- **Summary**: 2501.08188 说 GNLL fine-tune 给出可靠单目深度不确定性；检验该结论在多视图 GFM 上是否成立——多视图误差跨视图相关，逐像素独立不确定性假设破产。
- **so_what**: 方法论级发现，但单独成文偏薄；作为 C1 的核心 finding 之一更强。
- **risk**: MEDIUM | **effort**: days-weeks | **真机**: 否

## Lens: scaling-regime

### C6. 「更大的几何模型更自知吗？」
- **dedup_key**: `scale-vs-self-knowledge-geometry`
- **Summary**: 沿 VGGT-S/B/L、DAv2-S/B/L/G 测 reliability 随规模的变化：能力涨，自知涨得更慢（预期发现"reliability gap 随规模扩大"或相反）。
- **so_what**: 单独成文太薄（一张图的发现），并入 C1 分析章节。
- **risk**: LOW | **effort**: days | **真机**: 否

## Lens: diagnostic

### C7. GFM 在散射介质中失效的机制归因
- **dedup_key**: `mechanistic-attribution-gfm-scattering`
- **Summary**: 逐层 probing 定位失效源头：encoder 特征坍塌 vs 几何 head 外推错误；决定修复该往哪打（adapter 位置选择的原则性依据）。
- **so_what**: 解释性贡献，审稿人喜欢但难独立撑起 CVPR 主会；并入 C1。
- **risk**: MEDIUM | **effort**: weeks | **真机**: 否

---

## 机械去重合并后的复合候选（进入 jury 的最终集）

### IDEA-A ⭐⭐: "Do 3D Geometry Foundation Models Know When They Fail?"
**= C1 主干 + C5/C6/C7 作为 findings + 一个 training-free 方法**
- 结构：(i) **GeoReliab benchmark**：5 个退化族（散射-水下、散射-内窥镜、低纹理、动态非刚性、光度腐蚀）× 4-5 个 GFM 的 confidence 校准评估（AUSE/AURG/failure-detection AUROC/risk@coverage）；(ii) **预期发现**：confidence-error 相关坍塌、2D UQ 结论不迁移、规模不买自知；(iii) **方法**：training-free test-time epistemic 信号（视图重采样扰动集成 + 特征空间距离）恢复失效检测能力，任何冻结 GFM 即插即用；(iv) **下游验证**：不确定性加权融合提升退化域重建精度。
- **target benchmark**: DTU/ETH3D/ScanNet（干净）→ Robust-corruption 变体 → SCARED/Hamlyn/C3VD（内窥镜）→ FLSea/SQUID（水下）→ 动态场景
- **bottleneck addressed**: 下游生态盲目信任启发式 confidence
- **mandatory metrics**: AUSE, ECE-depth, failure-detection AUROC/AUPR, risk-coverage, 加权融合后 Chamfer/abs-rel;失败案例分册
- **expected failure mode**: 若 VGGT confidence 在 shift 下依然良好校准 → 转为正面结论论文（"惊人地稳健"）仍可发；方法部分若无增益 → 分析+benchmark 仍立得住
- **真机**: 否 | **risk**: LOW-MEDIUM | **effort**: 3 个月

### IDEA-B ⭐: "Selective 3D Reconstruction with Risk-Coverage Guarantees"
**= C3 主干 + C4 作为域特定评分**
- 方法主导：post-hoc 评分 + 空间感知 conformal 校准（处理像素相关性，可用 block/superpixel 级交换性）→ 认证式选择性重建；退化域展示 coverage 自适应收缩。
- **target benchmark**: 同 IDEA-A 域集
- **mandatory metrics**: risk@coverage 曲线、guarantee 违反率、coverage-vs-degradation 单调性、与 Trust3R/self-confidence baseline 对比
- **expected failure mode**: conformal 保证在强 shift 下失效（exchangeability 破产）→ 需要 shift-aware conformal，本身就是贡献点
- **真机**: 否 | **risk**: MEDIUM | **effort**: 3-4 个月

### IDEA-C: "Unified Scattering-Media Geometry Perception"（备胎）
**= C4 扩展**：水下+内窥镜统一物理参数化适配 + 可靠性轴
- **降位理由**: 易被读为"应用适配"，且 EndoGMDE/EndoUFM/水下 benchmark 已卷；作为 A/B 的评估域价值更大。
- **真机**: 否 | **risk**: MEDIUM-HIGH（novelty 角度）| **effort**: 3 个月

## 淘汰记录（budget/objective 之外均为降位并入，未删除）
| 候选 | 处置 |
|------|------|
| C2 | 并入 IDEA-A 实验 |
| C5/C6/C7 | 并入 IDEA-A findings |
| C4 | 并入 IDEA-B 评分函数 / IDEA-C 主干 |
| VLA 失效检测类 | Phase 1 已判饱和，未生成 |
