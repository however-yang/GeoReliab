---
status: superseded_historical
authority: GEORELIAB_TASKBOOK.md
execution_entry: false
superseded_by: GEORELIAB_TASKBOOK.md#v2.2
---

> Historical document retained for provenance. It is not an execution entry; the v2.2 mainline and all current phase gates are defined by GEORELIAB_TASKBOOK.md.
# Novelty Check Report v3 — 第二份路线报告（主线复核 + 两条新备选路线）

**日期**: 2026-07-26
**对象**: 外部路线报告 #2（"Ranking Is Not Warning" 主线 + 备选一 risk transport + 备选二 admission control）
⚠️ Codex 跨模型验证不可用（同前，单模型判断已标注）

## 主线部分：与 NOVELTY_REPORT_V2 一致，无需重查
该文档对主线三条新颖点（paired counterfactual / ranking-warning gap / task transfer）的定位与 V2 结论一致（8/10 PROCEED），且正确采纳了"不写'首个'、尊重 Ovadia 先例"的措辞建议 ✅。

## 新引用验证

| 引用 | 结果 |
|------|------|
| MASt3R-SLAM | ✅ [2412.12392](https://arxiv.org/abs/2412.12392)，真实 |
| CVPR 2026 CFP 主题列表 | 合理，未逐字验证（低风险） |
| 文中 CVPR 概率数字（40-55% 等） | ⚠️ **主观估计，非文献事实**——文档自己也声明了，但使用时勿当校准概率 |

## 备选路线一：Task-Conditioned Risk Transport — **7/10 PROCEED WITH CAUTION**

| 近邻（均验证 ✅） | 关系 |
|------|------|
| **Introspective Perception** ([1607.08665](https://arxiv.org/abs/1607.08665), 2016; [2306.16698](https://arxiv.org/abs/2306.16698), 2023) | **直接概念祖先**："从内部信号学习预测视觉系统失败"是机器人学十年老题。**文档完全没提这条文献线——这是它的盲区**，robotics 背景审稿人必引 |
| ConfidNet ([1910.04851](https://arxiv.org/abs/1910.04851)) | 分类域"学习置信度预测失败"的经典，同为祖先 |
| Trust3R | 文档已正确划界（point-level UQ head vs task-level transport）✅ |

- **Delta 真实存在**: 冻结 GFM backbone + paired-shift 监督 + pose/fusion 任务条件化——introspective perception 从未在 3D GFM 生态做过
- **判定**: 机制新颖性 MEDIUM（transport 本身是 isotonic/GBDT 老工具），设定新颖性 HIGH。**成立前提：related work 必须正面引用 introspective perception + ConfidNet 并划界**，否则"这就是 introspective perception for GFMs"一句话即可杀
- **额外风险**（文档已自知）: 若 clean-calibrated confidence 已足够，transport 增量不足

## 备选路线二：View-Set Admission Control — **7/10 PROCEED WITH CAUTION**

| 近邻（均验证 ✅） | 关系 |
|------|------|
| **Learning-to-Defer 文献族**（[2006.01862](https://arxiv.org/abs/2006.01862) 一致估计器；[2304.07306](https://arxiv.org/abs/2304.07306)；training-free conformal deferral [2509.12573](https://arxiv.org/abs/2509.12573)） | **正确的形式化框架**："fallback 到经典几何后端"= defer to expert algorithm。文档未提——既是威胁也是机会：主动引用可让论文获得理论支点 |
| Can These Views Be One Scene? (2605.18754) | 病态 view-set 幻觉发现——admission 的动机来源，文档已正确划界 ✅ |
| Error-guided view selection ([2412.11428](https://arxiv.org/abs/2412.11428)) | 视图选择为重建质量服务，非 admission gating，可划界 |
| UAVFF3D | 文档已正确划界 ✅ |

- **Delta**: 前馈 3D 管线的 admit/refuse/fallback 决策层检索为空白；且与用户 AAAI SurgRefuse 的 refusal 学术标签**天然连续**（四条路线中身份契合度最高）
- **判定**: 设定 HIGH、机制 MEDIUM。**成立前提**: ① 以 learning-to-defer 语言形式化（获得理论纵深）；② 必须有下游闭环证据（嵌入 MASt3R-SLAM 或 pseudo-fusion 管线展示 failure reduction），否则如文档自知——被读作 useful engineering

## 对文档整体的三点补充意见

1. **文献盲区**: 文档的近邻分析完全以 3D 社区为圆心；A1/A2 的真正审稿威胁来自 robotics（introspective perception）与 ML（learning-to-defer/ConfidNet）两条它没扫到的线。已在上表补齐，须并入 related work 义务清单
2. **时间线零冗余**: 其 gantt 从 8/3 推到 11/16 恰好压死 CVPR 截稿，**没有 arXiv 占坑节点**，且未计入 SR 论文 Stage B/C 的时间竞争。建议恢复 v1 计划的 9 月中旬 arXiv 门（哪怕只有 probe+DTU 主结果），并把 UAVLight/第三模型明确标为 rebuttal-ready 而非主文依赖
3. **工程治理建议**（存储三层/protocol冻结/IMPL_ONLY）与新颖性无关，作为执行纪律采纳无碍

## 路线排序裁定（综合 V2+V3）

1. **主线**（8/10）——先跑三周 probe
2. **A2 admission control**（7/10）——**建议从"备选二"升为"备选一"**：与用户 refusal 学术标签连续性最强、learning-to-defer 形式化有理论纵深、且若主线 gap 成立，admission 就是其最自然的"修复"章节（比 A1 的 GBDT transport 更有故事）
3. **A1 risk transport**（7/10）——机制最平淡（isotonic/GBDT），作为 A2 的实现组件比作为独立论文更合适
