# 项目总览

更新时间：2026-06-27

## 项目目标

本仓库是 OSX 的研究分支，目标是在尽量不破坏原始 OSX 全身能力的前提下，探索手部与面部微调：

- InterHand26M：提供高质量手部监督。
- BEDLAM：原计划提供全身 SMPL-X 正则与防遗忘，但在当前 frozen backbone/body 设置下实际价值有限。
- UBody：提供自然图里的手/脸上下文，是当前判断 full-body natural-hand pipeline 的关键数据。
- HInt：提供 held-out hand-interaction 2D 手关键点评测；当前 loader 是 hand-crop/local-hand 口径。

当前战略（2026-06-27 修订）：**脸不是贡献点；编码器移植 bug 已修复（pytorch 路径现与 OSX 逐比特一致）；手头对修复鲁棒、无需重训。修复后 UBody `[abs]` 与 EHF Face 免费追平 OSX，InterHand/HInt 仍赢；唯一干净未过的是 UBody natural-hand `[wa]` 落后 OSX 0.010（已隔离为纯手头自然图手指 articulation）。**

## 当前模型阶段

| 阶段 | 实验 | 状态 | 结论 |
|---|---|---|---|
| **编码器修复** | `vit.py` PatchEmbed padding | **已完成 2026-06-27** | StandardViT 移植 bug(padding 0 vs OSX 2)修复 → pytorch 编码器与 OSX 逐比特一致；手头鲁棒、无需重训；UBody `[abs]`/EHF Face 免费追平 |
| 手部微调 | `interhand_bedlam_c` | 已完成 | InterHand-test 全量 OSX 19.58mm → `_c` 16.82mm |
| 面部微调 | `face_ubody_e` | 已完成 | EHF Face 回到约 6.20mm，但未超过原始 OSX |
| 自然手评测 | `output/eval_ubody` | 三档 per-finger 完成 | 微调后自然手 2D 低于 OSX；退化沿指节递增、集中在 tip/j3，接近 `pretrained` 随机-ish 手头参照 |
| HInt held-out | `output/eval_hint` | 全链补测完成 | HInt win 从 `_c snapshot_8` 已出现：OSX `wa` 0.487 → `_c`/Stage3 0.480；Stage3/COCO 对 HInt 基本无新增 |
| Stage 3 | `joint_polish_f` | 已收口，终点 `snapshot_2` | InterHand PA 16.76（-14% vs OSX），HInt hand-crop 小幅赢 OSX，UBody `[wa]` 0.250→0.229（接近 OSX 0.219、未追平、确认平台），EHF Face 6.25；epoch 2(Phase 2) 本质 wash，已停训不跑 epoch 3 |
| COCO pilot | `coco_pilot` | 已跑 2 epochs/中途评测 | MSCOCO no-gate 稳定；HInt 基本不动，InterHand/UBody 小幅 polish；MSCOCO hard ROI gate 曾导致 loss 失稳，暂不采用 |
| 身体定性（shape） | `body_shape_t1` | 代码已实现·待跑 | 诊断＝betas mean-reversion（投影/朝向 OK）；T1 解冻 `shape_out`+`cam_out`，gated 默认关闭、对手/脸零影响，详见 `experiment_log.md` 2026-06-25 |

## 关键结果

### 当前干净基线（2026-06-27，snapshot_2 @ 修复编码器）

这是现在的 ground truth。手头未重训（仍是旧编码器训出），但在修复编码器上评测；body 与 OSX 逐比特一致，对比无 confound。

| 指标 | OSX | snap2 @修复 | 判定 |
|---|---:|---:|---|
| InterHand PA-MPJPE | 19.58 | **16.78** | 赢 −14%（in-domain caveat：OSX 未训 InterHand）|
| InterHand wrist-rel | 86.32 | 84.13 | 略好；绝对手系仍未解 |
| UBody [abs] NME | 0.311 | 0.316 | ~追平(免费) |
| UBody [wa] NME | 0.219 | 0.229 | 输 0.010（干净=手头自然图手指）|
| EHF Face PA-MPVPE | 6.09 | 6.15 | ~追平(免费) |
| HInt [all wa] NME | 0.487 | 0.480 | 赢(held-out hand-crop) |

> ⚠️ 下面各分项小节的数字是**旧（bug）编码器**上的历史记录，保留作诊断参照。编码器修复后 `[wa]`/InterHand PA 基本不变、`[abs]`/Face 改善，详见 `experiment_log.md` 2026-06-27。

### InterHand

- `interhand_bedlam_c` best 约 `snapshot_8/10`。
- InterHand-test 全量 52033 hands：

| 模型 | PA-MPJPE | wrist-rel MPJPE |
|---|---:|---:|
| 原始 OSX `normal` | 19.58mm | 86.32mm |
| `_c` snapshot_8 | 16.82mm | 84.58mm |
| `_c` snapshot_10 | 16.82mm | 84.11mm |
| `joint_polish_f` snapshot_0 | 16.85mm | 84.57mm |
| `joint_polish_f` snapshot_1 | 16.83mm | 84.29mm |
| `joint_polish_f` snapshot_2（终点） | **16.76mm** | 84.32mm |

- 手部增益：19.58 → 16.82mm，**-2.76mm / -14%**。旧 18.47 partial 口径已废弃。
- wrist-rel 仍在 84-87mm 高位，说明绝对手腕朝向/全局手系问题没有被这阶段训练彻底解决。

### HInt Hand-Crop Held-Out

HInt 当前只使用 partial 包中有图片的 Epic-Kitchens/NewDays 子集，共 3656 hands；Ego4D 标注因缺图片被跳过。当前 loader 以手 bbox 构造 hand-only crop，因此它更适合看 local hand-crop 泛化，不等价于 UBody 的 full-body natural-hand pipeline。

| 模型 | abs PCK@0.2 | abs NME | wa PCK@0.2 | wa NME |
|---|---:|---:|---:|---:|
| 原始 OSX | 0.132 | 0.451 | 0.121 | 0.487 |
| `_c snapshot_8` | **0.140** | **0.436** | **0.125** | **0.480** |
| `face_ubody_e snapshot_4` | **0.140** | **0.436** | **0.125** | **0.480** |
| `joint_polish_f snapshot_2` | 0.139 | 0.436 | 0.125 | 0.480 |
| `coco_pilot snapshot_1_itr3000` | 0.139 | 0.435 | 0.125 | 0.480 |

结论：HInt 相对 OSX 的小幅胜利在 `_c snapshot_8` 已经出现，后续 Stage3/COCO 基本不再改变。HInt 证明局部 hand-crop 能力不是纯 InterHand 过拟合，但它没有反映 Stage3 在 UBody 上的主要修复。

### EHF

- 原始 OSX Face PA-MPVPE 约 **6.09mm**。
- `face_ubody_e` snapshot4+ Face PA-MPVPE 约 **6.20mm**。
- 结论：面部阶段恢复了 pytorch face head，但没有形成对 OSX 的提升。

### UBody Natural Hand 2D

`test_sample_interval=10`，评测手数 3446：

| 指标 | 原始 OSX | `face_ubody_e` | `joint_polish_f` snap0 | `joint_polish_f` snap1 |
|---|---:|---:|---:|---:|
| `[abs]` PCK@0.2 | 0.422 | 0.383 | 0.405 | 0.406 |
| `[abs]` NME | 0.311 | 0.337 | 0.324 | 0.324 |
| `[wa]` PCK@0.2 | 0.587 | 0.515 | 0.560 | 0.560 |
| `[wa]` NME | 0.219 | 0.250 | 0.229 | 0.229 |

结论：伪 GT 手部 PA-MPJPE 看不出问题，但真实 2D 手关键点显示自然手退化。Stage 3 已大幅回收旧退化，但仍略差于原始 OSX，且 snapshot_0→snapshot_1 的 UBody 指标基本平台。

per-level `[wa]` NME 三档（`pretrained`=pytorch 映射预训练分支/随机-ish 手头参照）：

| level | OSX | 我们 | `pretrained` | 我们-OSX | 区间位置 |
|---|---:|---:|---:|---:|---:|
| j1 指根 | 0.205 | 0.208 | 0.207 | +0.003 | ~噪声 |
| j2 | 0.221 | 0.236 | 0.241 | +0.015 | 75% |
| j3 | 0.231 | 0.270 | 0.286 | +0.039 | 71% |
| tip 指尖 | 0.263 | 0.337 | 0.367 | +0.074 | 71% |

整手 `[wa]` NME 区间位置 72%（OSX 0.219 → 我们 0.250 → `pretrained` 0.262）。退化沿指节深度递增、指根不动、tip 最差，且接近随机-ish 手头的末端优先曲线。详见 `docs/experiment_log.md` 2026-06-24 条目。

Stage 3 snapshot_1 的 per-level `[wa]` NME 为 j1 0.206 / j2 0.225 / j3 0.242 / tip 0.287，tip/j1 放大比 1.39；相比旧 `face_ubody_e` 的 tip 0.337 与 tip/j1 1.62，主要修复确实发生在末端。

## 当前判断

- **编码器移植 bug（2026-06-27 修复）改写了多条旧结论**：`StandardViT` PatchEmbed padding 错（0 vs OSX 2），冻结 body/box/ROI 一直在吃 24% 错误特征。修复后：① 手头对此鲁棒、无需重训；② UBody `[abs]`、EHF Face 之前被归为”自然手退化”的，有一块其实是这个 bug，已免费回收；③ UBody `[wa]` 那 0.010 在 body 与 OSX 逐比特一致后仍在 → 被干净隔离为纯手头问题。
- UBody 上的脸不是贡献点，因为 OSX 本来就在 UBody 域内，微调最多恢复而非超越。
- BEDLAM 在当前 frozen encoder/body/box 设置下没有发挥原本的全身 GT mesh 价值；还引入了小手/遮挡噪声工程成本。
- 手部不是”野外不泛化”：HInt hand-crop held-out 小幅赢 OSX、UBody `[abs]` 追平、`[wa]` 落后 0.010。InterHand −14% 是 in-domain 特化，且 OSX 未训 InterHand（fine-tune vs zero-shot），论文需补公平基线。
- `coco_pilot` no-gate 只轻微 polish；MSCOCO hard ROI gate 因 pass 率低失稳，暂不作为主线（结论在旧编码器上得出，可在干净地基重评）。

## 推荐下一步

地基已修正（编码器与 OSX 逐比特一致），手头无需重训，`snapshot_2 @ 修复编码器`是干净工作基线。下一步：

1. **干净地打 UBody `[wa]` 那 0.010**（现在无 confound，可信测量）：
   - **路线 C**（`--train_hand_roi`，已实现、默认关闭）：warm-start snap2，修复编码器上解冻 `hand_roi_net` + decoder/regressor 小 LR 共适应，不动 `box_net`，不用 hard MSCOCO gate。主判据 UBody `[wa]`，kill 线 `[wa]`≤0.224，guardrail InterHand≤17.0 / HInt 不退 / EHF~6.2。
   - 备选/可叠加：末端 2D loss 加权（差距集中 tip/j3）、更多自然手监督。
2. **InterHand 公平基线**：补”在 InterHand 上同数据微调的 OSX”，或把 headline 让位给 held-out HInt。不补的话 −14% 在审稿里站不住。
3. 次要/附带：**`body_shape_t1`**（`--train_body_shape`，默认关闭、对手/脸零影响）——配图级，不当主线。wrist-rel（仍 84mm）需架构改动、高风险 stretch；不碰解冻 body/encoder（encoder 现已忠实，更没必要动）。

## 身体定性观感（独立小实验，已立项·代码已实现）

目标是真实图上身体 mesh 定性观感优于 OSX。诊断（demo 定性核对）：**投影/全局朝向基本 OK，真问题是体型 betas 系统性偏「软/厚」、抓不住精瘦体格（mean-reversion），即「轮廓缺陷」的真身**；复杂姿势（深蹲换胎）的大误差属 articulated pose，被冻结 backbone 卡死、数据不覆盖、且要在 OSX 最强项以少胜多，**不作为入口**。

采用 **T1**：解冻 `body_regressor.shape_out` ＋ `cam_out`（两个解耦线性头），BEDLAM-only shape 监督（屏蔽 UBody 伪 GT betas），极小 LR；双口径验证（BEDLAM/AGORA shape 量化 ＋ demo 重渲染）。注意 `cam_out` 同时利好绝对手位 2D（wrist-rel / UBody `[abs]` 的瓶颈本就在冻结身体 cam+pose）。需子模块级解冻 ＋ 更新 `_verify_freeze_status`。完整配方/架构依据/预期管理见 `docs/experiment_log.md` 2026-06-25 条目。

预期：动得了合成口径 shape，真实图轮廓改善不确定、可能有限；定性目标软，务必挂量化兜底。

## 文档维护规则

- `docs/post_stage3_roadmap.md`：**Stage 3 之后的前瞻路线/决策**（in-the-wild hand 主线、HInt/UBody 口径修正、COCO pilot 复盘、路线 C 条件）。规划方向时更新这里。
- `docs/continue.txt`：只放重开会话需要的短 handoff。
- `docs/project_overview.md`：每次战略判断改变时更新。
- `docs/experiment_log.md`：每次训练/评估完成后追加一条。
- 长推理、旧 bug 过程、历史命令放入 `docs/archive/`，不要继续堆进 `continue.txt`。
