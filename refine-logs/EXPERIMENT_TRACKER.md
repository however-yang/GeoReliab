# Experiment Tracker — GeoReliab

> **当前权威阶段**：Dual-MVE decision gate。下方原 R001–R042 是 GeoReliab 被选中后才执行的历史/备选队列；不得把本地 fixture 记为科学 run。

## Dual-MVE 当前状态（2026-07-26）

| Run ID | Lane | Purpose | Required evidence | Status | Notes |
|--------|------|---------|-------------------|--------|-------|
| I001 | Shared | Artifact contracts、统计、gate、CLI 与 fixture dry-run | pytest + py_compile | **DONE / NON-SCIENTIFIC** | 仅基础设施验证，不进入论文结果 |
| G001 | Geometry | 三候选任意两个 checkpoint 复现与 fusion hook | Spatial-MLLM/SpatialStack/GUIDE ≥2，固定 upstream commit/hash | **BLOCKED: EXTERNAL** | 本仓库无模型代码/权重 |
| G002 | Geometry | VSI-Bench/CVT-Bench split 冻结 | dev/reference/calibration/test scene-disjoint | **BLOCKED: EXTERNAL** | 本仓库无数据 |
| G003 | Geometry | matched interventions + patching | 两模型×两 strata，scene bootstrap/TOST | **BLOCKED: G001/G002** | 禁止以 zeroing-only 判定 |
| GR001 | GeoReliab | VGGT/MASt3R clean baseline | DTU 20 scenes + checkpoint hash | **BLOCKED: EXTERNAL** | CUT3R 不阻塞 MVE |
| GR002 | GeoReliab | 三退化 ×3 severity + downstream random control | CI-supported confidence、harm 与 zero-update evidence | **BLOCKED: GR001** | TartanAir native fog 作 sanity check |
| D001 | Decision | 冻结唯一主线 | G003/GR002 真实 gate JSON | **BLOCKED** | fixture 选择固定为 NON-SCIENTIFIC |

执行入口：`python -m georeliab_mve readiness` → 资源 ready 后运行 adapter → `python -m georeliab_mve evaluate-gates`。

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | 数据下载+格式统一 | — | DTU/Hypersim/TartanAir | — | MUST | TODO | nuScenes 注册、SCARED 协议同步启动 |
| R002 | M0 | 退化渲染管线 | Koschmieder雾/低照度/运动模糊/散焦/JPEG ×3 severity | 各数据集20场景 | 与TartanAir原生雾变体交叉验证 | MUST | TODO | 唯一严肃工程投入 |
| R003 | M0 | **MVE 决策门** | VGGT frozen | DTU 20场景 × 雾+低照度 ×3 severity | Spearman ρ, AUSE | MUST | TODO | ρ降>50%→主framing GO；稳健→切换framing |
| R004 | M0 | 指标实现验证 | — | NYUv2 小子集 | AUSE/ECE 对照 2501.08188 惯例 | MUST | TODO | |
| R010 | M1 | 主网格 VGGT | VGGT frozen | 3数据集×16条件×~150场景 | ρ/AUSE/AUROC/ECE/risk@cov | MUST | TODO | 缓存 conf+depth+err 图供 B3/B4 复用 |
| R011 | M1 | 主网格 MASt3R | MASt3R frozen | 同上 | 同上 | MUST | TODO | |
| R012 | M1 | 主网格 CUT3R | CUT3R frozen | 同上 | 同上 | MUST | TODO | 接口适配风险；失败不阻塞成文 |
| R020 | M2 | 真实域 夜/雨 | 3模型 | nuScenes night/rain 序列 | 同B1+GT密度报告 | MUST | TODO | LiDAR投影GT |
| R021 | M2 | 真实域 动态 | 3模型 | TUM/Bonn dynamic | 同B1 | MUST | TODO | |
| R022 | M2 | 真实域 低纹理 | 3模型 | ETH3D 室内子集 | 同B1 | MUST | TODO | |
| R025 | M2 | 下游过滤 | conf/random/gradient/photometric 排序 ×4 kept-fraction | DTU官方协议 + 退化sev-2 | acc/comp/F-score/Chamfer | MUST | TODO | 等稀疏度对照是 A4 防御核心 |
| R026 | M2 | 下游加权融合 | conf加权 vs 等权 vs oracle | Hypersim/TartanAir | 融合深度误差 | MUST | TODO | |
| R027 | M2 | SCARED 附录域 | VGGT/MASt3R | SCARED keyframes | 同B1 | NICE | TODO | 协议到位才跑 |
| R030 | M3 | cheap 信号 | kNN特征距离+注意力熵 | B1/B2 全条件（复用缓存需补特征） | AUROC/AUSE/risk@cov | MUST | TODO | 参考库: ScanNet/CO3D 50k tokens |
| R031 | M3 | full 信号 | K=8 视图重采样分歧 | B1 代表性子集+B2 | 同上+延迟 | MUST | TODO | 等算力对照设计 |
| R032 | M3 | baseline 启发式 | 光度重投影/PKRN移植/跨视图一致性 | 同 R030 | 同上 | MUST | TODO | W3 stereo-confidence 文献义务 |
| R033 | M3 | baseline GNLL | GNLL head fine-tune (冻结backbone) | Hypersim train→全条件测 | 同上 | MUST | TODO | 唯一训练环节 ~40 GPU·h |
| R034 | M3 | baseline MC-Dropout | 推理期注入 dropout ×8 | B1 子集 | 同上 | MUST | TODO | 架构无dropout则文中说明 |
| R035 | M3 | Trust3R 对比 | Trust3R (若代码可得) | 交集条件 | 同上 | NICE | TODO | 不可得→文字对比 |
| R036 | M3 | selective 协议 | split-conformal image-level | 全方法输出 | risk@coverage 违反率 | NICE | TODO | 措辞: protocol 非 guarantees |
| R037 | M3 | 效率 Pareto | cheap/full/全baseline | — | AUROC vs FLOPs/延迟 | MUST | TODO | Figure 6 |
| R040 | M5 | 失效分类学 | 逐退化 top-losing 案例 | B1/B2 缓存 | 定性+rank-vs-calibration | NICE | TODO | B5 |
| R041 | M5 | 单目对照 | MoGe/DAv2 | B1 子集 | 同B1 | NICE | TODO | 附录（W5 防御） |
| R042 | M5 | 规模效应 | VGGT/DAv2 可得规模档 | B1 子集 | 同B1 | NICE | TODO | 附录 |
