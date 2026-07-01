# Stage 3 之后的路线与决策（in-the-wild hand 主线）

更新时间：2026-07-01

本文件是**前瞻性路线/决策文档**，承接 Stage 3 收口之后的方向选择。
- 按时间的结果记录见 `docs/experiment_log.md`；当前状态快照见 `docs/project_overview.md`。
- 本文件回答的是：**手部微调收口后，往哪走、为什么、风险多大、怎么判定生死。**

---

## 0⁗. 2026-07-01 更新（四）：H4W++ 竞品 + 目标下沉「全绿 vs stock OSX」+ 结构性 decoder（已实现，gate 训练中）

### 关键决策（本会话与用户敲定）

- **目标定位下沉**：不追顶会。目标 = **相对 stock OSX 全面提升（全绿，无论多小）+ 一个能写成一句话的方法贡献**。可发性判断 = **够中低档**（workshop 近必中；TCSVT/TMM/PR/CVIU/IVC/Neurocomputing/The Visual Computer 现实可期；3DV/WACV/BMVC 够一够），前提是下面三件事到位。
- **结构性 decoder 大改：已实现+提交（commit `0e30614`）、smoke 过、gate 训练中。** 本轮只做 §0‴ 根因 **(1)+(2)**：每层坐标头 + 迭代参考点 + 每层 aux 监督（`HandDecoder` 特征精修器→坐标精修器）+ detach `query_init` 封 DCNv4 NaN。**层数 3→6、去 topo/occlusion、soft-argmax 量化刻意留作后续单独 ablation**（保持单变量）。它既是方法贡献来源，也是论文活过审稿的闸。实现/smoke/配方细节见 `experiment_log.md` 7-01。
- **生死闸（每天盯的唯一数字）**：新 decoder 的 InterHand PA 必须钻到**公平基线 16.29↓**（OSX decoder 同条件微调、且仍在降），**不是只打 stock 19.58**。
  - 过闸 → 方法论文成立，`paper_table_plan.md` 的 Table 1/3/5 顺手填绿。
  - 不过闸 → 退「高效可复现配方 + 冻结骨干分析」分析型论文（更低档但仍可发），**不得声称方法优越性**。

### gate 配方与起点纪律（务必同配方，否则 16.29 不可比）

- **配方 = snap2(`joint_polish_f`) 原样**：`lr 5e-5 / batch 64 / end_epoch 4 / phase1_epochs 2 / posnet_lr_mult 0.5 / lr_mult 0.1`。（初版误抄 CLAUDE.md 旧例 lr 2e-4/batch48/end14/phase1 10，用户 catch 已纠正。）
- **起点 = `osx_l` 冷启、4 epoch**：snap2 与公平基线都这么做 → 新 decoder 也同起点同预算才是干净单变量；暖启 snap2 会给 head start + 8-epoch 预算失衡，那是"在 snap2 上再加精修头"的另一实验、非本闸。
- **配方一致性 caveat**：本地 `osx_fairbase_smoke` 用 lr 1e-5（仅 snap2/poseur_iso 是 5e-5）；须确认产出 16.29 的 OSX-ft 与 snap2 同配方，否则基线自身就不公平。用户确认本地无正式 osx_fairbase、已自行核对参照无误。

### 新竞品：Hand4Whole++（CVPR 2026，Moon，必须对标）

`E:\论文\Hand4Whole-plus-plus_RELEASE` = "Enhancing Hands in 3D Whole-Body Pose Estimation with **Conditional Hands Modulator**"，OSX/Hand4Whole/SMPLer-X 同源作者。

- **架构**：冻结 SMPLer-X 身体 + Hand4Whole 脸 + **WiLoR**（手专家，fp16）+ DWPose（2D→手框）；唯一可训 = 一个**零初始化卷积 ControlNet**（24 层、每 ViT block 一个），把 WiLoR 手特征逐层注入身体 ViT 的 patch token。最终手 mesh = **WiLoR MANO 经 Procrustes 刚性 graft 到 SMPL-X 腕** + 边界平滑。零初始化 ⇒ step 0 逐比特等于冻结 baseline（与我们 freeze flag 同哲学）。
- **战略含义**：**"提升 whole-body 手部"的新意轴 = 融合冻结专家 / 条件调制，不是 decoder**——headline 手部增益来自专家（WiLoR），学到的只是轻量整合器。**坐实 §0‴ 的 decoder 根因结论**；"我们 decoder 打败 OSX decoder"这条卖点双重失效。
- **我们的差异化（写进 related work）**：单模型、**不挂外部专家/检测器**（无 WiLoR/YOLO/DWPose）、冻结骨干诚实分析。不对标 → 好一点的地方 desk-reject。
- 详见记忆 `h4wpp-competitor.md`。

### 可借鉴的"免费"零件（headline-agnostic，无论闸过不过都该加）

- **多参考系关键点 loss**（直击 §0‴ 根因 #1 的贫乏监督）：part-relative（腕/颈相对）+ inter-hand-relative + inter-hand 成对向量 + **显式监督 MANO root_pose**。
- **Procrustes graft + 边界平滑**：让"更好的手"真正进入最终 mesh 指标。⚠️ graft 只在"贴进去的手 > 基线手"时才涨点——graft 我们自己（弱于公平基线的）手不制造胜利，只是干净缝进去。

### 论文表格 / 实验规划

→ **`docs/paper_table_plan.md`**：图例（🟢必赢 stock / 🔑必赢公平基线 / ⚪同台 / 🔍分析 / ⚠️风险）+ 五张表（全身 / 手专项含生死闸 / held-out 泛化 / 消融 / 效率）+ 最小可发子集 + 档位预期。修 decoder 时照它对表。

---

## 0‴. 2026-06-30 更新(三)：公平基线结果 + decoder 根因 + 能否超过 OSX 的概率判断

### 公平基线结果：OSX decoder 同等条件略胜、且更 sample-efficient

同一 400-iter 子集(3200 手，L:680/R:2520)：**OSX-ft(itr1500，1/3 epoch) PA 15.53 vs 我们 snap2 15.85 → OSX-ft 领先 0.32mm，且未收敛**(wrist-rel 80.80 vs 81.11 同向)。子集偏易(我们全量 16.78)，但**同口径下 OSX 仍在前**。backbone 冻结、coord loss 相同 → 差距是 **decoder 架构/实现**，非 loss/数据。**"−14% 是我们 decoder 的方法贡献"守不住**；待 OSX-ft 收敛 + 全量复评再定幅度。

### Decoder 根因（代码审查，杠杆排名）

1. **我们的 decoder 是"特征精修器"不是"坐标精修器"**：被监督的 2D 手坐标全来自 `hand_position_net` soft-argmax(`model_core.py:721`)；`hand_decoder`(725) 只输出**特征**喂 regressor，**无坐标输出、无坐标监督**，唯一梯度是 pose→FK 间接。OSX Poseur **每层回归坐标 + 每层 RLE/坐标 aux 监督**(`hand_decoder.py:86-88,118-144`)。→ 稀疏间接 vs 密集直接监督，解释 sample-efficiency 差。
2. **参考点全程不迭代**(`my_decoder.py:661-672`)：从 `coord_init` 算一次、所有层共用、layer 只返回特征；OSX 每层用 offset 更新参考点。3 层都盯同一 soft-argmax 初始点，深层纠不了坏 init（**末端最吃亏**=distal gap）。
3. **soft-argmax 16³ 量化瓶颈**(`output_hand_hm_shape=16×16×16`)：指尖精度受限；OSX 回归连续坐标。
4. **3 层 vs OSX 6 层**(`model_core.py:1118` vs `hand_decoder.py:120`)。
5. **over-engineered topo/occlusion 模块**(`my_decoder.py:454-458`)：OSX 层是干净 self-attn+deformable+FFN；我们多了 `HandTopologyAttention`+`ImplicitOcclusionModule`，decoder 还输 → **高度怀疑帮倒忙，头号 ablation**。

最高杠杆单点：**把 decoder 改成迭代坐标精修 + 每层 aux 坐标监督(deformable-DETR/Poseur 式)**，一次解决 1/2/3、直接打 distal。捷径待查：`model_core.py:1099` 有现成 `PoseurDecoder`，若是 OSX decoder 的 port 可直接切/对比。

### 能否超过 OSX 的概率判断

**结构天花板**：backbone 与 OSX 逐比特一致 → 同一份特征，**OSX 性能 ≈ 我们在其域上的天花板**；系统性超过需 decoder 严格更强，否则高概率结局是**追平**而非超过。

| 域 | 追平 OSX | 明显超过 | 仍输 |
|---|---:|---:|---:|
| 实验室手 InterHand | ~50% | ~30%（叠 OSX 没用的 RLE 有上行）| ~20% |
| 自然手 UBody `[wa]` | ~40-45% | **~15%**（OSX in-domain + 共享冻结特征，主场难）| ~40-45% |

- **两域同时明显超过 ≈ 10-15%**；**两域追平(或一超一平) ≈ 40-50%，是高概率结局。**
- **不要押"两域都超 OSX"(~12%)。** 高概率且可发表的故事 = **in-domain 追平 OSX + held-out(HInt) 赢 + 干净 from-scratch 复现 + 诚实分析**。HInt 是我们真有头寸、OSX 没利用的地方。
- 抬概率三件事(性价比序)：① decoder 迭代坐标精修 + aux 监督；② **叠 RLE loss**(OSX 看家武器、我们没用，最可能让 InterHand 真反超)；③ ablate topo/occlusion(免费验证是否帮倒忙)。

---

## 0″. 2026-06-30 更新：WA2D 收口 → 主线转 InterHand 公平基线

**UBody `[wa]` 那 0.010 的 direct-loss 攻坚（tipw + WA2D，6-28~30）已收口为负结果**（完整记录见 `experiment_log.md` 2026-06-30）：三堵墙——指标天花板（最优 frz500 仍 +0.0068、88% distal）、数值 NaN（根在 DCNv4 反向 kernel、与 weight 无关）、InterHand 侵蚀（frz500 PA=17.00 踩 kill 线）。**§3-§4 的"路线 C / direct 2D 监督"主线到此为止；下面 §1-§7 作历史保留。** 新主线 = **把 InterHand −14% 从"不公平"变"公平/诚实"**，这是论文真正的成败点。

### 背景：为什么 −14% 现在站不住

OSX 训练集（MSCOCO/H36M/MPII/UBody/AGORA）**不含 InterHand26M**，所以我们 InterHand-test PA 16.78 vs OSX 19.58（−14%）本质是"我们微调 vs OSX 零样本"，审稿人会当 domain adaptation 打折（见 memory `interhand-not-in-osx-training`）。

### 代码约束（必须先知道）

`main/OSX.py`（normal 路径）**没有**冻结脚手架（`frozen_modules`/`freeze_modules()`/`trainable_module_names`），而 `common/base.py` 训练流程无条件调用 `model.module.freeze_modules()`（602）等。→ **`--decoder_setting normal` 跑训练会直接 AttributeError 崩**；normal 在本 fork 是纯 eval（19.58 基线来源）。所以"微调 OSX"不是加 flag，需要先给 OSX.py 补脚手架。

### 轨 A（零训练，立刻让论文站住）—— 换 headline

- **headline 让给 held-out HInt**：HInt hand-crop held-out，两个模型都没训过 → 公平对比，我们 0.487→0.480 赢。
- **InterHand 降级为诚实 in-domain 结果**：明确标注 fine-tune-vs-zero-shot caveat，不当主卖点。
- 成本 0 训练，纯 framing；先化解审稿人质疑。

### 轨 B（受控实验，需代码）—— 架构受控公平基线

只让"手 decoder 架构"一个变量不同，回答"我们的 decoder 是否真比 OSX 强（而非只是喂了数据）"：

| 维度 | 我们 | 公平基线 |
|---|---|---|
| backbone | 冻结(encoder/body/box/ROI) | **同样冻结** |
| 训练数据 | IH 0.35 / UBody 0.25 / MSCOCO 0.40 | **完全相同** |
| 可训部分 | pytorch DCNv4 手头 + my_decoder + 手 regressor | **OSX 原生 MMCV 手 decoder + 头** |
| warm-start | osx_l.pth.tar（未见 InterHand）| **同起点** |
| schedule/LR/epoch | 我们的配方 | **相同** |

**判定矩阵：**

| OSX-finetuned InterHand PA | 解读 |
|---|---|
| ≈ 16-17（追上我们）| −14% 是数据带来的，decoder ≈ OSX → 方法贡献弱，走轨 A 的 system/held-out 定位 |
| 明显 > 我们（~18）| 我们 decoder 真更强 → 硬方法贡献 |
| ≈ 19.58（没动）| OSX decoder 吃不动 InterHand → 强赢（需排除）|

**无论结果如何都必须做**——它决定论文方法 claim 的强度。

**需要的代码（轨 B 实施中，2026-06-30 起）：**
1. 给 `OSX.py` 加 `frozen_modules`/`freeze_modules()`/`trainable_module_names`，照 `model_core.py:69-149` 模式：冻 encoder/body_position_net/body_regressor/box_net/hand_roi_net/face 相关，只训 hand_decoder + 手头/regressor。
2. 两步 save/load（frozen backbone + lightweight trained），key 映射对上 MMCV decoder。
3. 跑通 `freeze_modules()` + `_verify_freeze_status`，smoke 训练。
- 风险：MMCV decoder 在本 harness 下训练未测过，可能需调几处 key/接口。

### 建议

轨 A 现在就定（免费、立即生效）；轨 B 作为决定方法强度的实验并行推进。两者不互斥：A 守底（held-out 故事一定成立），B 争上限（若证明 decoder 更强则方法贡献坐实）。

---

## 0′. 2026-06-27 更新：编码器 bug 已修复，UBody confound 解决

**本文件 §0–§3 围绕"UBody natural-hand 仍输 OSX、可能有 confound"展开。该 confound 现已坐实并解决：**

- 根因是 `StandardViT` PatchEmbed padding 错（0 vs OSX 2），冻结 body/cam 一直漂（cross-path 腕 1.4°）。修复后 pytorch 编码器与 OSX **逐比特一致**，cross-path 腕漂移归零。详见 `experiment_log.md` 2026-06-27。
- 修复后的干净结论：**手头无需重训**；UBody `[abs]` 与 EHF Face **免费追平 OSX**；UBody `[wa]` 那 0.010 **仍在、且现在 100% 是手头**（不再有 body confound）。
- **所以 §0 那张"现状表"里的 `[abs]` 退化、wrist-rel 部分归因要更新**：`[abs]` 已基本追平；`[wa]` 0.010 是唯一干净未过项。下面 §1–§7 的路线逻辑（路线 C 优先、wrist-rel 需架构改动、兜底判据）**基本仍成立**，只是现在都在无 confound 的干净地基上测量。

---

## 0. 起点：我们现在手里有什么（审稿人视角）

Stage 3 `joint_polish_f` 已收口，终点 `snapshot_2`。把结果放到审稿人视角：

| 部分 | 现状 | 性质 |
|---|---|---|
| 手 InterHand PA | 19.58→**16.76**（-14%）✅ | 主要量化亮点，属 **in-domain 特化**（在 InterHand 上重训得比 OSX 狠）|
| 手 HInt hand-crop | 0.487→**0.480**(`[wa]` NME) ✅ | held-out hand-interaction 小幅赢 OSX，但 `_c snapshot_8` 起已达成 |
| 手 wrist-rel | 84-86mm，没动 ❌ | 绝对手系/全局朝向未解决 |
| 手 自然图(UBody `[wa]`) | 0.219→**0.229，仍差于 OSX** ❌ | full-body natural-hand pipeline 仍是当前最大硬伤 |
| 脸 EHF Face | 6.09→6.25，略差 ❌ | 无贡献 |
| 身体 | 冻结=与 OSX 相同 | 结构上动不了 |

**核心矛盾（2026-06-26 修正）**：手部并非“完全不泛化”。`_c snapshot_8` 起已经在 HInt hand-crop held-out 上小幅超过 OSX，但 UBody full-body natural-hand 仍低于 OSX。也就是说，局部手能力成立，完整自然图里的 ROI/上下文/小手 pipeline 仍没追平。

**因此下一阶段的目标**（也是论文成败点）：
> **把手的胜利从 hand-crop/local-hand 搬到 full-body natural-hand pipeline——在 InterHand 不退（≤~16.8）、HInt 不退的同时，让 UBody natural-hand 追平/超过 OSX。**

### 0.1 HInt 补测后的口径修正

HInt 全链补测见 `docs/experiment_log.md` 2026-06-26。关键发现：

| 模型 | HInt abs NME | HInt `[wa]` NME |
|---|---:|---:|
| OSX baseline | 0.451 | 0.487 |
| `_c snapshot_8` | 0.436 | 0.480 |
| `face_ubody_e snapshot_4` | 0.436 | 0.480 |
| `joint_polish_f snapshot_2` | 0.436 | 0.480 |
| `coco_pilot snapshot_1_itr3000` | 0.435 | 0.480 |

判读：
- HInt win 在 `_c snapshot_8` 已出现，后续 Stage 3 / COCO pilot 基本不再改变。
- HInt 当前 loader 是 hand-only crop 口径，更接近 InterHand；它能证明 local hand-crop 能力有 held-out 泛化，但不能单独代表完整 OSX full-body 场景。
- Stage 3 的主要贡献是 UBody natural-hand `0.250→0.229`，HInt 对这件事不敏感。
- 因此 HInt 从“唯一主判据”降级为 **local hand-crop held-out guardrail**；full-body natural-hand 主判据应看 UBody `[wa]/[abs]` 与 demo。

---

## 1. 候选路线总览

| 路线 | 机制 | 价值 | 风险/代价 | 判定 |
|---|---|---|---|---|
| **A. 冻结框架内调参/换 decoder** | LR/schedule/解码器结构 | 仅 in-domain 边际 | 低 | ❌ 不做：已平台，且碰不到 wrist-rel / 泛化 |
| **B. 加 COCO-WholeBody 真实 2D（冻结）** | 补野外真实 2D 手监督 | 轻微 polish；HInt 无新增，UBody/InterHand 小幅改善 | 低到中（hard MSCOCO gate 已证实不稳） | 🔶 已跑 pilot：不建议同配方长训 |
| **C. 解冻 `hand_roi_net`（+decoder 共适应）** | 给 full-body natural-hand pipeline 新容量 | 针对 UBody 仍输 OSX 的真实瓶颈 | 中 | ✅ 当前更有价值的下一步候选 |
| **D. wrist-rel 架构改动（手腕全局朝向 DOF）** | 给手分支补全局朝向自由度 | 最 novel | 高、不确定（2D 朝向歧义是全领域硬上限）| 🔻 stretch goal，非主线 |
| **E. 身体 shape T1（已实现）** | 解冻 `body_regressor.shape_out`+`cam_out` | 次要/配图 | 低 | ⚪ 附带结果，不当论文主线 |
| **F. 兜底** | 转 workshop/技术报告，或承认到顶 | — | — | 决策门触发时启用 |

详细分析见各路线小节（§3-§6）。

---

## 2. 路线 B 复盘：COCO pilot 是轻微 polish，不是 HInt 突破

### 已跑配方

- 暖启 `joint_polish_f/snapshot_2`。
- 加权三源：`InterHand26M=0.35 / UBody=0.25 / MSCOCO=0.40`，砍掉 BEDLAM。
- 冻结 backbone/body/box/ROI，只训练 hand/face position、decoder、regressor。
- 稳定版本只开 `--ubody_use_hand_roi_quality`；**未开 MSCOCO hard gate**。

### 结果复盘

| 指标 | Stage3 snap2 | COCO pilot | 判读 |
|---|---:|---:|---|
| HInt `[wa]` NME | 0.480 | 0.480-0.481 | 无新增 |
| HInt abs NME | 0.436 | 0.435-0.436 | 无实质新增 |
| UBody `[wa]` NME | 0.229 | 0.227-0.228 | 小幅 polish |
| InterHand PA | 16.76mm | 16.67-16.69mm | 小幅 polish |
| EHF Face | 6.25mm | 约 6.12-6.21mm | 波动内回收 |

结论：
- COCO pilot 没有推动 HInt，因为 HInt 的胜利早在 `_c snapshot_8` 就存在。
- COCO 对 UBody/InterHand 有轻微正向，但幅度太小，不足以构成主突破。
- 当前 no-gate COCO 结果可作为一个较稳的 polish checkpoint；不建议继续同配方长训。

### MSCOCO hard gate 复盘

曾尝试 `--mscoco_use_hand_roi_quality`，日志显示：
- MSCOCO gate pass 常仅 `0.12-0.37`，`n_valid` 经常只有 `2-8`。
- 初期 loss 正常，约 1.3k iter 后 `joint_proj / joint_img / smplx_joint_img` 整体飙高。
- 飙高不只发生在 MSCOCO，InterHand/UBody 也被共享 hand head 带坏。

判读：当前 hard gate 用冻结 box/ROI 结果做二值筛选，但 box/ROI 本身不更新；在 MSCOCO 上它主要是在大量删除手部 2D 监督，导致梯度分布变尖和训练动态失稳。**不要再直接开启 hard MSCOCO gate。**

若未来还想利用 MSCOCO ROI 质量，必须改成小实验：
- soft weighting，而非整只手二值置零；
- 或 MSCOCO 专属低阈值（如 coverage 0.2-0.3 / min_joints 4）短跑验证；
- 或等路线 C 解冻 ROI 后再重新评估 gate。

---

## 3. 后续手部主线：优先修 full-body natural-hand pipeline

当前真正没过的指标是 UBody full-body natural-hand：`[wa]` 仍为 0.227-0.229，高于 OSX 0.219；HInt hand-crop 已经平台，不能再作为区分路线的主指标。

建议的执行顺序：

1. **停止同配方 COCO 长训**：HInt 无新增，UBody/InterHand 只微动。
2. **以 UBody natural-hand 为主判据**：看 `[wa]`、`[abs]`、per-level tip/j3；HInt 只看是否退化。
3. **进入小心版路线 C**：解冻 `hand_roi_net`，并与 hand decoder/regressor 小 LR 共适应；先不动 `box_net`，不用 hard MSCOCO gate。
4. **若 C 仍不动 UBody**：再考虑更高风险的 `box_net` 或 wrist/global pose 架构改动；否则及时转向。

推荐 guardrail：
- InterHand PA：维持 ≤~16.8，最多不破 17.0。
- HInt hand-crop：不低于 Stage3 水平（`wa` NME ~0.480）。
- EHF Face：约 6.2 附近，避免脸明显回退。
- UBody：主目标是 `[wa]` 从 0.227-0.229 往 0.219 靠，且 `[abs]` 不恶化。

---

## 4. 路线 C：解冻 `hand_roi_net`（条件性下一步）

**何时上**：当前即可作为下一步候选。理由不是“B 没让 HInt 超 OSX”（HInt 早已小幅超），而是 **UBody full-body natural-hand 仍差于 OSX，且冻结 head/decoder 继续训练已平台**。

**机制**：给 full-body natural-hand pipeline 新容量，让 hand ROI 特征与 decoder/regressor 共适应。**参数高效**（先不碰 encoder/body/box）。

**关键约束（重要）**：
- **decoder 不从头训**——用现在的当初始化、**和 ROI 一起小 LR 共适应**。
- **不能只解冻 ROI 而冻着 decoder**：ROI 特征一动、decoder 不变 → 输入分布 mismatch、无梯度通路跟上 → 大概率冲掉 InterHand 增益。ROI 进优化器，decoder（+regressor）必须一起进。
- **先 `hand_roi_net`、后 `box_net`**：`hand_roi_net`（crop+upsample）是渐进扰动；`box_net`（挪动 crop 位置）扰动剧烈，destabilize 风险大，分阶段。
- 机制上是**模块级**改动（把 `hand_roi_net` 移出 `frozen_modules`、加进 `trainable_module_names`），比 body T1 的子模块白名单还简单；两步加载仍成立。

**风险**：ROI 特征分布一动，把 decoder 输入带飘、冲掉 InterHand/HInt → 靠小 LR + 联合训练 + InterHand/HInt guardrail 控。主判据看 UBody，而不是 HInt。

---

## 5. 路线 D：wrist-rel（stretch goal，非主线）

**根因（已在代码确认）**：`hand_regressor` 只输出 **15 个手指关节**（喂 SMPL-X `*_hand_pose`）；**手腕全局朝向是 body 关节、来自冻结的 `body_pose`**（`model_core.py:493,546-558`）。wrist-rel 的 3D loss（`joint_cam` 本就是 wrist-relative）**早已在监督**，但**唯一可训 DOF 是手指，手指无法表示整只手的全局旋转** → loss 有、却无 DOF 可作用 → 卡 84mm。

**结论**：**loss 设计单独无解**，必须**架构改动**——给手分支加一个"手腕全局朝向"头（预测腕 6D 旋转），splice 进运动链（override/residual 掉 body 链的腕旋转），已有的 wrist-rel loss 才有东西可优化。

**风险/代价（最高）**：
1. 运动链接缝：手预测的腕 ≠ 身体链前臂 → mesh 折断；需一致性/blend loss。
2. **2D→3D 朝向本质歧义**：同一 2D 投影对应无数 3D 朝向——全领域绝对手系都差的根因，**即使加 DOF+loss 上限也可能有限**。
3. InterHand（纯手 crop、无身体锚点）上更难；折中是预测**相对前臂**的腕旋转减歧义，但又被冻结前臂限制。

**定位**：高风险低确定性的 stretch goal。只在 B/C 稳了、且愿担架构+歧义风险时再上。**别当主线。**

---

## 6. 兜底（决策门触发时）

- 若 C（以及可选的 soft ROI weighting / 低风险 COCO 变体）仍推不动 UBody → OSX 在这个冻结家族的 full-body natural-hand pipeline 已接近天花板，"完整自然图手超 OSX"不是可行卖点 → **及时转向**。
- 退路：① 现有 InterHand 特化结果做 **workshop/技术报告**（in-domain 贡献 + 诚实的泛化分析）；② 把重心移到别的可量化角度。
- 决策点很清晰：**能不能在 UBody/full-body natural-hand 上追平或超过 OSX，同时 InterHand 与 HInt hand-crop 不退。**

---

## 7. 评测口径（统一）

| 基准 | 角色 |
|---|---|
| **UBody natural-hand `[wa]/[abs]`** | full-body natural-hand 主判据；当前唯一仍稳定低于 OSX 的手部口径 |
| **HInt**（held-out hand-crop 2D）| local hand-crop held-out guardrail；证明非纯 InterHand 过拟合，但不单独代表 full-body pipeline |
| InterHand26M test（全量）| in-domain guardrail（≤~16.8，盯 Pareto 代价）|
| EHF Face | 脸 guardrail（~6.2）|
| demo 重渲染 | 定性配图 |

---

## 8. 一句话总结

Stage 3 拿到了**手部基准硬胜（InterHand -14%）**，并且 `_c` 起已经在 **HInt hand-crop held-out** 上小幅超过 OSX；真正没过的是 **UBody full-body natural-hand**。COCO pilot no-gate 只带来轻微 polish，hard MSCOCO gate 不稳。下一阶段若继续手部主线，应把 HInt 作为 guardrail，把 UBody natural-hand 作为主判据，优先试小心版 **路线 C：解冻 `hand_roi_net` + decoder/regressor 共适应**；wrist-rel 架构改动仍是高风险 stretch，身体 shape T1 是独立附带线。
