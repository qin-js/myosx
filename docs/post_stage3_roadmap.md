# Stage 3 之后的路线与决策（in-the-wild hand 主线）

更新时间：2026-06-25

本文件是**前瞻性路线/决策文档**，承接 Stage 3 收口之后的方向选择。
- 按时间的结果记录见 `docs/experiment_log.md`；当前状态快照见 `docs/project_overview.md`。
- 本文件回答的是：**手部微调收口后，往哪走、为什么、风险多大、怎么判定生死。**

---

## 0. 起点：我们现在手里有什么（审稿人视角）

Stage 3 `joint_polish_f` 已收口，终点 `snapshot_2`。把结果放到审稿人视角：

| 部分 | 现状 | 性质 |
|---|---|---|
| 手 InterHand PA | 19.58→**16.76**（-14%）✅ | 唯一量化亮点，但属 **in-domain 特化**（在 InterHand 上重训得比 OSX 狠）|
| 手 wrist-rel | 84-86mm，没动 ❌ | 绝对手系/全局朝向未解决 |
| 手 自然图(UBody `[wa]`) | 0.219→**0.229，仍差于 OSX** ❌ | **手的提升不泛化、野外反退** = 当前最大硬伤 |
| 脸 EHF Face | 6.09→6.25，略差 ❌ | 无贡献 |
| 身体 | 冻结=与 OSX 相同 | 结构上动不了 |

**核心矛盾**：唯一的量化亮点（InterHand -14%）伴随一个会被直接攻击的弱点——**手不泛化、野外还退了**。任何"我们手更好"的论文，只要 in-the-wild 比 baseline 差，基本会被拒。

**因此下一阶段的目标**（也是论文成败点）：
> **把手的胜利从 in-domain 搬到 in-the-wild——在 InterHand 不退（≤~16.8）的同时，野外手（held-out）超过 OSX。** 两者都要，才是论文。

---

## 1. 候选路线总览

| 路线 | 机制 | 价值 | 风险/代价 | 判定 |
|---|---|---|---|---|
| **A. 冻结框架内调参/换 decoder** | LR/schedule/解码器结构 | 仅 in-domain 边际 | 低 | ❌ 不做：已平台，且碰不到 wrist-rel / 泛化 |
| **B. 加 COCO-WholeBody 真实 2D（冻结）** | 补野外真实 2D 手监督 | 修野外泛化、最便宜的判定实验 | 低 | ✅ **当前选择（第一步）** |
| **C. 解冻 `hand_roi_net`（+decoder 共适应）** | 给容量、制造超过 OSX 的 headroom | 高（参数高效、不碰 encoder/body）| 中 | 🔶 条件性下一步（B 若只持平则必上）|
| **D. wrist-rel 架构改动（手腕全局朝向 DOF）** | 给手分支补全局朝向自由度 | 最 novel | 高、不确定（2D 朝向歧义是全领域硬上限）| 🔻 stretch goal，非主线 |
| **E. 身体 shape T1（已实现）** | 解冻 `body_regressor.shape_out`+`cam_out` | 次要/配图 | 低 | ⚪ 附带结果，不当论文主线 |
| **F. 兜底** | 转 workshop/技术报告，或承认到顶 | — | — | 决策门触发时启用 |

详细分析见各路线小节（§3-§6）。

---

## 2. 当前选择：路线 B（加 COCO-WholeBody），以及为什么

### 为什么先选 B
1. **数据现成**：`data/MSCOCO/MSCOCO.py` 已读 `coco_wholebody_train_v1.0.json`，带 `lefthand_kpts/righthand_kpts`（line 75/102-103）。接进训练成本低。
2. **直接对症当前硬伤**：野外手退化。COCO-WholeBody 是**野外真实 2D 手**，整身、能直接接 SMPL-X 手关节投影 loss。
3. **最便宜的判定实验**：它能直接测出我们处在"数据受限"还是"特征受限"（见 §2.3），决定后续是否加码。

### 必须认清的前提（决定概率）
- **OSX 上游训练时就用了 COCO-WholeBody 含手 2D**。它的 UBody `[wa]` 0.219 正是用这份数据、且 **backbone 可训** 练出来的。→ 我们是**冻结 backbone、在 OSX 用过的数据上追/超它**，而那个被冻住的 backbone 恰是 OSX 用这数据调出来的。
- **UBody 训练本来就用了真实 2D 手关键点**（`UBody.py:179-184`）。→ COCO 对我们是**"同一种监督、更大更多样"**（COCO ~10万+多样图 vs UBody 15 个视频场景），**不是新信号**。

### 2.3 核心未知：0.229 平台是"数据受限"还是"特征受限"
| 读法 | 含义 | 指向 |
|---|---|---|
| (a) 数据受限 | UBody 太窄才停在 0.229，COCO 多样性能打破 | 能超过 OSX |
| (b) 特征受限 | 冻结特征+只训头，渐近线≈0.219（OSX 把 backbone 调到这了）| 只到追平 |

证据：Stage 3 **带着自然 2D 数据**（UBody）仍平在 0.229、多 epoch 不往 0.219 漂 → **略偏 (b)**；但 UBody 太窄是混淆项，COCO 可能打破。**先验无法断定，pilot 直接测。**

### 2.4 概率评估（冻结 + 加 COCO）
| 结果 | 概率 |
|---|---|
| 追平 OSX（~0.219±噪声）| ~65-75% |
| 名义超过（`[wa]`<0.219 任意幅度）| ~35-45% |
| **可发表的稳超**（HInt held-out 有清晰 margin）| ~25-35% |

"名义超过"接近掷硬币，但**可发表的稳超仍 <50%**，因为：① 带自然数据已平在 0.229；② 冻结渐近线≈OSX 水平；③ 降一丢丢是噪声、审稿人不认。

### 2.5 Pareto 张力（容易忽略的制约）
要把自然手压过 OSX，最有效杠杆是**重配比**（加大自然数据权重、降 InterHand 主导），但这很可能**反噬 InterHand PA**（-14% 头条）。论文要**两个都赢**，而 in-domain 与 in-the-wild 常在 Pareto 前沿互斥——**这才是真正难的部分**。

---

## 3. 路线 B 执行计划（pilot + 决策门）

### Pilot 配方
- 暖启 `joint_polish_f/snapshot_2`（现成好手头当初始化）。
- 把 **MSCOCO（coco_wholebody）加进 `trainset_2d`**，调配比（自然 2D 占比上来、InterHand 别过载）。
- **先保持冻结 backbone**（只动手头/decoder），短跑。
- 保留手部既有保护项（ROI gate 等）。

### 测量陷阱（必读）⚠️
训 COCO-WholeBody 后再用 UBody 评 = **部分 in-distribution、数字偏乐观**。**可发表结论必须在训练没见过的 held-out 野外基准（HInt）上评**，否则审稿人直接打折。UBody 只作参考、看趋势。

### 决策门
1. 若 HInt/UBody `[wa]` **明显往 0.219 以下走** → 判定 **(a) 数据受限** → 加码：叠 **路线 C（解冻 hand_roi_net）** 冲更低。
2. 若**卡在 ~0.219 不动** → 判定 **(b) 特征受限** → 冻结路线到顶，要么上路线 C 给容量，要么转 §6 兜底。
3. 全程盯 **InterHand guardrail**（≤~16.8）看 Pareto 代价；EHF Face 看脸 guardrail。

### 实现成本
- MSCOCO loader 现成；主要是把它加进 `config.py` 的加权 `trainset_3d`、设采样配比、校验 hand kpt 顺序/valid 与现有 joint_set 一致。
- 中等偏低工作量。

### 准备状态（2026-06-25）
**代码侧已就绪**：
- ✅ `data/MSCOCO/MSCOCO.py` train 分支 meta_info 已补齐 fork 的 6 个键（`bb2img_trans / dataset_id=3 / is_interhand=False / is_bedlam=False / is_ubody=False / is_hand_only=False`），与 BEDLAM/UBody/InterHand 的 17 键 schema **完全对齐**（否则 mixed-batch `default_collate` 会崩）。targets 键、joint_num=134 本就一致。
- 改动安全：MSCOCO 不在 trainset 时**零影响**。

**数据侧是 blocker（需你获取，我下不了）**：
- `dataset/MSCOCO/images/`（COCO2017 train images, ~19GB）
- `dataset/MSCOCO/annotations/coco_wholebody_train_v1.0.json`（COCO-WholeBody 真实 2D，含手）
- `dataset/MSCOCO/annotations/MSCOCO_train_SMPLX.json`（OSX/Hand4Whole 出的伪 GT SMPL-X 拟合）
- loader 路径优先 `dataset/MSCOCO/{images,annotations}`，否则回退 `dataset/coco/`。`cfg.data_dir = <root>/dataset`。
- 评测 held-out 的 **HInt**（HaMeR）也需另外获取（eval 时用）。

**config 改法（pilot 启动时）**：`main/config.py` 改为**三源加权（砍掉 BEDLAM）**——把 MSCOCO 放进加权 3d 组而非 trainset_2d，避免 `make_same_len=True` 强制的 50/50（那会把 InterHand 腰斩到 17.5%）：
```python
trainset_3d = ['InterHand26M', 'UBody', 'MSCOCO']
trainset_2d = []
trainset_3d_sample_prob = {'InterHand26M': 0.35, 'UBody': 0.25, 'MSCOCO': 0.40}
```
- **比例理由**：InterHand 0.35 维持原值当 guardrail 锚（实验干净 + 加 MSCOCO 后绝对曝光反升）；MSCOCO 0.40 当被测主力（保证统计功效）；UBody 0.25 保域连续性；**砍 BEDLAM**（冻结 backbone 下身体 loss 不训练、`--bedlam_no_hand_img_loss` 又切了手 2D、只剩合成手姿态，价值最低）。唯一代价：3D 手监督只剩 InterHand（lab），略增 in-domain 拉力，但合成手姿态本就帮不上自然手，可接受。
- **比例是起点不是定值**：跑起来按前 ~1280 itr 调——PA 稳就更激进（`IH 0.30 / MSCOCO 0.40+`）；PA 在 IH 0.35 下仍掉则说明 MSCOCO 在干扰手姿态，需降 MSCOCO/加 IH。
- 机制：从 `trainset_3d` 和 `trainset_3d_sample_prob` **同时删掉 `'BEDLAM'`**。`itr/epoch ≈ (len_IH+len_UB+len_MSCOCO)/64`，每源曝光 = `itr × prob`。

**启动命令（数据就位后）**：暖启 snapshot_2、冻结 backbone：
```bash
cd /workspace/myosx/main
export UBODY_ANNOTATION_DIR=/workspace/myosx/dataset/UBody/annotations_filtered
python train.py --gpu_ids 0 --lr 5e-5 --lr_mult 0.1 \
  --train_batch_size 64 --num_thread 8 --end_epoch 4 --phase1_epochs 2 --save_iters 1280 \
  --exp_name output/coco_pilot --decoder_setting pytorch --encoder_setting osx_l --grad_clip 1.0 \
  --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
  --init_trained_path ../output/joint_polish_f/model_dump/snapshot_2.pth.tar \
  --train_face_modules --posnet_lr_mult 0.5 --ubody_use_hand_roi_quality
```
（BEDLAM 已砍，`--bedlam_*` flags 已移除。`--save_iters 1280` 便于早停判定。）

**启动后自检**：日志应出现 MSCOCO 被加载、batch 里有 MSCOCO 样本、`✅ 梯度流正常`、无 collate 报错；早期看 UBody/InterHand 趋势。

---

## 4. 路线 C：解冻 `hand_roi_net`（条件性下一步）

**何时上**：路线 B 只到追平、判定为特征受限时；或想主动制造超过 OSX 的 headroom 时。

**机制**：给模型新容量，在同一份数据上抽出比 OSX 冻结手头更好的自然手特征。**参数高效**（不碰 encoder/body）。

**关键约束（重要）**：
- **decoder 不从头训**——用现在的当初始化、**和 ROI 一起小 LR 共适应**。
- **不能只解冻 ROI 而冻着 decoder**：ROI 特征一动、decoder 不变 → 输入分布 mismatch、无梯度通路跟上 → 大概率冲掉 InterHand 增益。ROI 进优化器，decoder（+regressor）必须一起进。
- **先 `hand_roi_net`、后 `box_net`**：`hand_roi_net`（crop+upsample）是渐进扰动；`box_net`（挪动 crop 位置）扰动剧烈，destabilize 风险大，分阶段。
- 机制上是**模块级**改动（把 `hand_roi_net` 移出 `frozen_modules`、加进 `trainable_module_names`），比 body T1 的子模块白名单还简单；两步加载仍成立。

**风险**：ROI 漂移把 decoder 输入带飘、冲掉 InterHand → 靠小 LR + 联合训练 + InterHand guardrail 控。

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

- 若 B（持平）+ C（仍只持平）都推不动 → OSX 在这个冻结家族已到天花板，"野外手超 OSX"不是可行卖点 → **及时转向**。
- 退路：① 现有 InterHand 特化结果做 **workshop/技术报告**（in-domain 贡献 + 诚实的泛化分析）；② 把重心移到别的可量化角度。
- 决策点很清晰：**能不能拿到一个干净的、held-out 上"in-the-wild 手 > OSX"的数字，同时 InterHand 不退。**

---

## 7. 评测口径（统一）

| 基准 | 角色 |
|---|---|
| **HInt**（held-out 野外手 2D）| **主判据**：声称"野外手 > OSX" 必须在此 |
| InterHand26M test（全量）| in-domain guardrail（≤~16.8，盯 Pareto 代价）|
| UBody natural-hand `[wa]/[abs]` | 参考/趋势（训 COCO 后部分 in-distribution，不单独定论）|
| EHF Face | 脸 guardrail（~6.2）|
| demo 重渲染 | 定性配图 |

---

## 8. 一句话总结

Stage 3 拿到了**手部基准的硬胜（InterHand -14%）**，但泛化是硬伤。下一阶段唯一高价值的主线是**把这个胜利搬到野外**：**先加 COCO-WholeBody（路线 B）跑 pilot，用 HInt held-out 测"数据受限 vs 特征受限"** → 数据受限就叠**解冻 hand_roi_net（路线 C）**冲过 OSX。wrist-rel（D）需架构改动、是高风险 stretch；身体 shape（E）是附带配图；都不当主线。**可发表的稳超 OSX 概率约 25-35%，且必须不掉 InterHand——值得用低成本 pilot 赌一把，但要设清楚转向门。**
