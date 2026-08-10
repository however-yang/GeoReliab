---
status: superseded_historical
authority: GEORELIAB_TASKBOOK.md
execution_entry: false
superseded_by: GEORELIAB_TASKBOOK.md#v2.2
---

> Historical supporting material retained for provenance. It is not an execution entry; the v2.2 mainline and current phase gates are defined by GEORELIAB_TASKBOOK.md.
# Review Brief — 供外部评审模型使用（Codex MCP 配置后重跑）

> 状态: ⚠️ 2026-07-26 Codex MCP 未配置，本 brief 尚未送审。
> 配置命令: `claude mcp add codex -s user -- codex mcp-server`
> 送审方式: `/research-review` 会自动读取本文件。

## 角色设定
你是资深 CVPR 审稿人/AC。假设这份计划某处是坏的，你的任务是找到坏在哪。逐条对抗式审查。

## 论文计划

**标题方向**: "Do 3D Geometry Foundation Models Know When They Fail?"
**目标**: CVPR 2027（截稿约 2026-11 中旬）

### 贡献结构
1. **GeoReliab benchmark**: 5 退化族（水下 FLSea/SQUID、内窥镜 SCARED/Hamlyn/C3VD、低纹理、动态非刚性、光度腐蚀）× 4-5 个前馈几何基础模型（VGGT/DUSt3R/MASt3R/MoGe）confidence 校准评估（AUSE / ECE-depth / failure-detection AUROC / risk@coverage）
2. **Findings**: (a) confidence-error 相关性在 shift 下坍塌; (b) 2D UQ 结论（GNLL 最优等）不迁移到多视图 3D; (c) 模型规模增大不改善自知
3. **Training-free 方法**: 视图重采样扰动集成 + 特征空间距离 → test-time epistemic 信号，冻结 GFM 即插即用
4. **第二贡献**: selective 3D reconstruction 协议（post-hoc 评分 + conformal 校准 risk-coverage 保证）

### 基线
GFM 自带 confidence / Trust3R (2605.19539, 训练式 evidential) / MC-Dropout / GNLL fine-tune 移植 / 光度重投影误差 / blur score

### 已知近邻（已验证）
- VGGT-UQ analysis (2606.16479): 仅 DTU 干净数据的分析
- Test3R (2506.13750, NeurIPS'25): cross-pair consistency 用于 test-time training 提精度（非 UQ）
- Trust3R (2605.19539): 训练式 evidential head，未测 shift
- 2501.08188: 2D 单目深度 UQ 综合

### 约束
无真机；4 个月窗口；作者资产：手术单目深度代码、NeurIPS epistemic probing 框架、AAAI action monitor；GPU 假设 4-8 卡。

## 请回答
1. 贡献对 CV 社区是否真正新颖？哪一条最弱？
2. 最小可信证据包（minimum viable evidence package）是什么？
3. 缺了哪些 baseline / failure analysis 会被拒？
4. benchmark 设计会被如何质疑（域选择、GT质量、指标选择）？
5. 方法部分被批「TTA/ensemble 不新」的防御策略是否成立？
6. 写一份 mock CVPR review（Summary/Strengths/Weaknesses/Score/Confidence）。
