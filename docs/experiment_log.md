# 实验日志

本文件按时间追加实验结果和决策。只写结论、关键数字、下一步，不放长篇排查过程。

## 2026-07-01

### PoseurDecoder 隔离实验（itr1500，inconclusive）+ DCNv4 NaN 因果模型【修正】

**poseur_iso 结果**（`--use_poseur_hand_decoder`，warm-start snap2 但 PoseurDecoder 结构不符 → decoder 从**随机初始化**；WA2D 关，日志确认 `set hand_wa_2d_loss_weight to 0.0`）：itr1500 同子集(3200 手) InterHand PA **29.14** / wrist-rel 70.05，vs OSX-ft itr1500 15.53、snap2 15.85。**这不是判决**：PoseurDecoder 冷启、仅 1/3 epoch，对暖/全训 decoder 不公平，信息量≈0；loss 在降。wrist-rel 70 反低于 80-81 有点意思（全局好、局部 PA 差），但欠训 checkpoint 不可读。itr1710 在 `hand_position_net.dcnv4_blocks.0` NaN（未带 `--skip_nonfinite_grad` → abort）。

**【修正 6-30 的过度说法 "DCNv4 broadly fragile"】**：DCNv4 头历史上（interhand_bedlam_c / joint_polish_f，暖 3 层 HandDecoder、无 WA2D）训 **10+ epoch 从不炸**。它**只在大上游梯度经未-detach 的 `query_init`（`model_core.py:725` 传 `hand_img_feat_joints`）回流进 DCNv4 反向时溢出**。两类 NaN 共因 = "大梯度进 DCNv4"：① WA2D 的 `1/diag` 尖峰经 decoder 回传；② 冷启随机 6 层 PoseurDecoder 的早期大梯度。→ **poseur_iso 的 NaN 是冷启瞬态，非 WA2D（已确认 weight=0.0）、非 DCNv4 固有缺陷**；decoder 暖起来梯度变小、预期 NaN 自停。准确表述：DCNv4 反向在常规区间（暖浅 decoder、无 WA2D）稳，仅大上游梯度下溢出。

**下一步**：带 `--skip_nonfinite_grad` 重跑（poseur_iso2）熬过冷启、训到收敛（4 epoch）再对 OSX-ft 16.29 / snap2 16.78。看 `skipped=N`：集中早期后停 = 坐实冷启瞬态、能跑到收敛拿真数；持续到晚 = DCNv4×深 decoder 有更深不稳，再议。公平性 caveat：PoseurDecoder 无法暖启（无兼容预训练权重），只能"从零训到收敛"对暖启基线——追平=架构强，没追平=冷启 handicap 与架构差分不开。备选解法（未采用）：detach `query_init` / LR warmup 压早期梯度。

### 战略结论（本会话）：定位下沉 + H4W++ 竞品 + 决定上结构性 decoder + 表格规划

- **目标定位下沉（用户确认）**：不追顶会，目标 = **相对 stock OSX 全面提升（全绿，无论多小）+ 一句话方法贡献**。结论：**决定上结构性 decoder 大改**（每层出坐标 + 每层 aux 监督 + 迭代参考点 + 层数↑，即 §0‴ 最高杠杆项），它既是方法贡献来源、也是论文活过审稿的闸。
- **生死闸**：新 decoder 的 InterHand PA 必须钻到**公平基线 16.29↓**（不只是 stock 19.58）。过闸→方法论文成立；不过→退「高效可复现配方 + 冻结骨干分析」分析型论文、不得声称方法优越性。
- **新竞品 Hand4Whole++（CVPR2026，Moon，OSX 同源）**：冻结 SMPLer-X 身体 + WiLoR 手专家 + DWPose，只训一个零卷积 ControlNet（"Conditional Hands Modulator"），最终手 mesh = WiLoR MANO 经 Procrustes graft 到 SMPL-X 腕。**坐实"提升 whole-body 手部"的新意轴是融合冻结专家、不是 decoder**，必须 cite/对标（差异化=单模型、不挂外部专家/检测器）。详见 `post_stage3_roadmap.md` §0⁗、记忆 `h4wpp-competitor.md`。
- **表格/实验规划已固化** → `docs/paper_table_plan.md`。

### 结构性 decoder 实现 + smoke 通过 + gate 配方锁定（gate 训练中）

**已实现并提交**（commit `0e30614`）：HandDecoder 从"特征精修器"改为"坐标精修器"——每层 `bbox_embed` 坐标头（末层零初始化）+ 迭代参考点（logit/sigmoid，镜像 `vit.py` PoseurDecoder）+ 返回 `(features, per_layer_coords)`；`model_core.py:725` detach `query_init`（切 decoder→DCNv4 NaN 路径，`cfg.detach_hand_decoder_query` 默认 True）+ 新增 `loss['hand_aux_coord']`（每层坐标 vs 重映射手部 GT + BEDLAM 掩码 trunc，`cfg.hand_aux_coord_loss_weight=1.0`）。PoseurDecoder 同步改 2-tuple。**只动根因 (1)+(2)**，topo/occlusion + 3 层 + 同 loss 保持不变（单变量）。

**smoke 通过**（`output/smoke_coordrefiner`，batch4 ~2100 iter，暖启 snap2）：冻结骨干全加载、warm-start 覆盖 471 张量、freeze 检查过；`loss_hand_aux_coord` 有限（~0.44，贴 soft-argmax 基线，因 bbox_embed 零初始化）；**全程 `skipped=0`、零 NaN → detach 实测封住 NaN 路径**（detach vs skip 之争，数据判 detach）。

**gate 配方锁定（用户 catch 修正）**：初版误抄 CLAUDE.md 旧例（lr 2e-4/batch48/end14/phase1 10，全错）。snap2(`joint_polish_f`) 真实配方 = **lr 5e-5 / batch 64 / end_epoch 4 / phase1_epochs 2 / posnet_lr_mult 0.5 / lr_mult 0.1**；gate 必须同配方否则与 16.29 不可比。
- **起点纪律**：snap2 与公平基线都是 `osx_l` 冷启训 4 epoch → 干净 gate 应**同样 osx_l 冷启 4 epoch**（只 decoder 架构不同=单变量），非暖启 snap2（head start + 预算失衡）。暖启 snap2 版是"在 snap2 上再加精修头"的另一实验。
- **配方一致性 caveat**：本地 `osx_fairbase_smoke` 用 lr **1e-5**（非 5e-5），仅 snap2/poseur_iso 是 5e-5；用户确认本地无正式 osx_fairbase、已自行核对 16.29 参照无误。

**下一步**：gate 训练中；训完评 InterHand 全量（中间 epoch 都评取最低）对 **16.29↓**。盯 `loss_hand_aux_coord` 是否钻到 0.44 以下（钻=坐标头真精修；死贴=精修没到输出/`model_core.py:728` 那条线，按诊断表走 → 精修坐标喂 regressor）。

## 2026-06-30

### WA2D 触顶 + NaN 高置信根因定位 + 梯度稳定补丁

**Bootstrap（配对，1万次重采样，逐手对齐 n=3432；6-29 文档要求但当时未做的那个）**：

| 对比 | `[wa]` | tip | 备注 |
|---|---|---|---|
| frz500 vs tipw3000 | −0.0018 CI[−0.0019,−0.0017] SIG | −0.0042 SIG | freeze 真比 tipw 好，但极小 |
| frz500 vs snap2 | −0.0031 SIG | — | 整条 WA2D 线净收益 |
| frz500 vs frz1000 | −0.0001 | — | 平台坐实 |
| frz500 vs OSX | **+0.0068** CI[+0.0060,+0.0076] SIG | +0.0174 CI[+0.0156,+0.0191] SIG | 仍显著输；gap **88% 在 tip+j3** |
| frz500 vs tipw3000（3D PA手）| +0.087mm SIG | — | WA2D 拿一丝 3D 换 2D（仍 ~OSX 水平）|

判读：配对 CI 窄到 0.0001 也判"显著"（见 frz500 vs frz1000），说明**统计显著不是该看的标尺**；WA2D 把 gap 均匀压低约 30% 但 distal 形状不变（仍 88% tip+j3）→ 命中 `wa2d_group_attribution.md` 预注册的**天花板支**。frz500 是当前最好的评测产物，但收益微小且已渐近。

**frz500 guardrail（2026-06-30 补测，`output/hand_wa2d_distal_guard_freeze_hand_position_net/result/snapshot_0_itr500/`）**：

| 指标 | snap2 | tipw3000 | frz500 | 判定 |
|---|---:|---:|---:|---|
| InterHand PA | 16.78 | 16.95 | **17.00** | 正好踩 17.0 kill 线，lineage 最差 |
| InterHand wrist-rel | 84.13 | — | **83.89** | 微好 |
| HInt `[all wa]` | 0.480 | 0.480 | **0.480** | 守住 |
| HInt `[all abs]` | 0.445 | — | **0.445** | 守住 |
| EHF Face PA-MPVPE | 6.15 | 6.15 | **6.15** | 守住 |
| EHF Hands PA-MPVPE | 15.67 | 15.62 | **15.72** | 轻微回退但可接受 |
| UBody `[wa]` | 0.229 | 0.227 | **0.226** | 仍输 OSX 0.0068 |
| UBody tip | 0.288 | 0.284 | **0.280** | 仍输 OSX |

判读：**严格说 guardrail 基本过，但 frz500 没有报告余量**。InterHand PA=17.00 刚好压线，且从 snap2→tipw→frz500 单调变差；HInt/EHF 守住，UBody `[wa]` 小幅最好，但仍显著输 OSX。结合 bootstrap 和 NaN 根因，frz500 应定位为"WA2D 触顶诊断/崩溃前快照"，不宜作为主报告点，也不该把 freeze 当稳定 recipe 推进。

**NaN 高置信根因（关键，推翻 6-29 "稳定 freeze 路线"）**：freeze 没修 NaN，只把崩溃从 `module.hand_position_net.dcnv4_blocks.0`（guarded run，itr **820**，lr=1e-7，offset_mask 部分坏 55296/57344）挪到 `module.hand_decoder`（freeze run，itr **1295**，level_embed/input_proj 全坏）。两处都是可变形注意力反向。机制解释：`err = sqrt(Σdiff²+eps) / diag`，`diag` 由 GT 算、**无梯度** → 反向幅度随 `1/diag` 放大；旧 `diag.clamp(min=1e-4)` 让极小 GT 手把它推到 **1e4** 量级，wrist 对多指误差还会累积，有限但可能饱和 DCNv4/decoder 反向成 NaN。`hand_tipw`（无此 loss、梯度 O(1)）从不炸，是反证。待 6-30 补丁在同配置下跑过 itr1295 后闭环。**freeze itr500/itr1000 是崩溃前快照，可评测但配方不可复现、不稳定。**

**补丁（2026-06-30，3 文件 `config.py`/`model_core.py`/`train.py`）**：新增 `hand_wa_2d_min_diag=1.0`（每指梯度 ≤1，与其它 2D loss 同量级）+ `hand_wa_2d_err_clip=5.0`（hard clamp 单关节归一化误差）。WA2D 为 opt-in，`weight=0` 时字节不变；旧 NaN 行为可复现 `--hand_wa_2d_min_diag 1e-4 --hand_wa_2d_err_clip 0`。`py_compile` + `git diff --check` 通过；未跑 smoke（交互环境 DCNv4 CUDA 枚举失败）。

**补丁实测（`output/hand_wa2d_stable`，weight 0.1、posnet 不冻、安全默认）——只缩小未根治**：补丁确实生效（code 快照含 min_diag/err_clip，日志确认设上），但仍在 `hand_position_net.dcnv4_blocks.0` 于 itr **814**（前 820，几乎同批数据）炸，只是**爆炸半径缩约 20×**：offset_mask 坏元素 55296→**3072**、value_proj 满坏→**98304/262144**，且 **output_proj/norm2 这次干净**。崩前 `loss_hand_wa_2d_raw=0.088` 正常。坏元素位置（output_proj/norm2 干净、value_proj/offset_mask 部分坏、norm1 全坏）说明 **NaN 生于 DCNv4 可变形采样的反向 kernel 内部**；loss 端只能压上游梯度量级（已压 20×）、压不掉 kernel 内溢出。`hand_tipw`（无 WA2D）从不炸是反证。**根因层级从 loss 上移到 DCNv4 kernel。**

**新增 `--skip_nonfinite_grad`（train.py，默认关闭=旧严格 abort）**：backward 后检测到非有限梯度则 `zero_grad`+跳过该 iter（不 step）、记日志计数 `skipped=N`，而非中止整个 run；scheduler/timer 照走。用于绕过 ~1/800 病态 batch 拿完整 epoch 测量。看 `skipped=N`：个位数=可用；几十+=大比例手 batch 病态、结果有偏、应收掉 WA2D 线。实现把 `_raise_nonfinite_trainable_grads` 拆出非抛异常的 `_collect_nonfinite_trainable_grads`，clip/step 包进 `if not skip_iter:`。`py_compile` 过，flag 关时字节不变。

**下一步**：① WA2D loss 补丁只缩小未根治 → 用 `--skip_nonfinite_grad`（或更低 weight 0.05，user 正测）拿一个完整 epoch 的干净测量，与 frz500/tipw 做 bootstrap——这是 WA2D 线最终 go/no-go，不是要救成主线；② 做干净 MSCOCO source ablation（`sources=ubody` vs `ubody,mscoco`，除 source 外同配置、同 warm-start）——`hand_wa2d_distal` 的 ubody-only run 不算，因为它是 weight=1.0 激进+无守卫的崩溃 run；③ 不再把 frz500/freeze 当主线，若写入论文只能作为 WA2D 缩小 gap 但触顶的负结果/诊断；④ 回到 InterHand 公平基线 + UBody 诚实定位（teacher distillation 只能追平 OSX、不可能超过 teacher）。

### WA2D 线收口（weight=0.05 是封口点）

**weight=0.05 实测（`output/hand_wa2d_stable_005`，posnet 不冻）——两个轴都更差且仍炸**：指标 itr500 group `[wa]`=0.2263（Δ vs OSX **+0.0081**）、tip Δ +0.0203——**比 frz500 还差**（+0.0068 / tip +0.0174），符合"weight 减半=polish 减半"。NaN 在 itr**767**（比 0.1 的 814/820 **更早**），爆炸半径与 0.1 几乎一样（offset_mask 4096/57344、value_proj 131072/262144）。→ **在 0.05–0.1 区间 NaN 基本与 weight 无关**：降权既换不来更好指标、也逃不掉 DCNv4 kernel NaN。（此 run 未带 `--skip_nonfinite_grad`，走旧严格 abort。）

**完整 sweep（group attribution 口径）**：

| weight / 变体 | UBody `[wa]` | Δ vs OSX | tip Δ | NaN |
|---|---:|---:|---:|---|
| snap2（基线）| 0.2281 | +0.0099 | +0.0252 | — |
| 0.05 itr500 | 0.2263 | +0.0081 | +0.0203 | @767 |
| 0.1 guarded itr500 | 0.2256 | +0.0075 | +0.0188 | @820 |
| 0.1 freeze itr500 | 0.2250 | +0.0068 | +0.0174 | @1295(decoder) |
| 1.0 aggressive | 0.260 | 崩 | — | — |
| OSX | 0.2182 | 0 | 0 | — |

**三堵墙全部用数据封死**：① 指标天花板（最优 frz500 仍 +0.0068、88% distal，单调但够不到 0）；② 数值（NaN 与 weight 无关、根在 DCNv4 反向 kernel，只能 skip 不能调）；③ InterHand 侵蚀（frz500 PA=17.00 踩 kill 线）。**WA2D 权重 sweep 已无信息增量 → 线收口。** `frz500≈frz1000` 已是收敛证据，不必再补 0.1+skip 长跑；frz500 仅作 ablation 负结果（"direct 2D 监督收 ~30% gap 后平台"）。

**转论文（替代上面的"下一步"①②）**：① **InterHand 公平基线**（同数据微调一个 OSX-normal 当对照）——最高性价比，守住 −14% 头条；② **rotmat pose loss**（`use_hand_rotmat_pose_loss`，已实现未试）——便宜低风险的 **InterHand 侧** polish（仅在 InterHand/BEDLAM 有可靠 hand pose GT 处 fire，UBody 伪 GT 吃不到），当 ablation；③ **OSX teacher distillation**——把 UBody 锁到 parity（追平、超不过 teacher）；④ **诚实定位**：held-out 泛化赢（InterHand/HInt）+ in-domain 追平（UBody/EHF），不在 OSX 训练域内硬刚。

### body-shape T1 启动配方（2026-06-25 立项的可执行版；代码已验证完好）

T1 = 解冻 `body_regressor.shape_out`(10) + `cam_out`(3)，BEDLAM-only 监督、极小 LR，针对诊断出的 betas mean-reversion（身体偏软/厚、抓不住精瘦体格）。**定位：配图级附带线，非主贡献。** 预期诚实：13 参数 + 冻结特征 + 全合成 betas 监督 → **大概率动合成口径 shape，真实图轮廓改善不确定、可能有限**；务必挂量化兜底，别靠肉眼自我感动。

**代码状态（2026-06-30 复核）**：T1 实现（config/model_core/base/train 四处，详见 2026-06-25 条目）经 WA2D/track-B 改动后**仍完好**，flag/prefixes/optimizer group/verify 白名单均在，launch-ready。

**前置 config.py 改动**（全局，跑完务必改回）：`trainset_3d=['BEDLAM']`、`trainset_2d=[]`（BEDLAM-only，屏蔽 UBody 伪 GT betas 污染）。

**启动命令**（pytorch 路径，冻手只训 shape/cam；`--phase1_epochs`=end_epoch 保证满 LR，避免 phase2 ×0.1 坑）：

```bash
cd /workspace/myosx/main
python train.py --gpu_ids 0 --lr 1e-5 --train_batch_size 64 --num_thread 8 \
  --end_epoch 2 --phase1_epochs 2 \
  --exp_name output/body_shape_t1 --decoder_setting pytorch --encoder_setting osx_l \
  --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
  --init_trained_path ../output/joint_polish_f/model_dump/snapshot_2.pth.tar \
  --no_train_hand_modules --train_body_shape --body_shape_lr 1e-5
```

启动应见 `🔧 body-shape T1: 解冻 ~13k 参数` + verify/grad-flow 通过（仅 shape_out/cam_out 有梯度，手全冻）。

**验证（三条全做，不能只看 demo）**：

| 口径 | 看什么 | 作用 |
|---|---|---|
| 量化锚（必须）| AGORA/BEDLAM shape PA-MPVPE / betas 误差 | 合成口径 shape 动没动、变好变坏——客观真相 |
| 定性 | 重渲染 `demo/output_face_ubody_e/compare/{24,39,69,92}` 精瘦男多姿态 vs OSX | 真实图轮廓有无肉眼改善 |
| 手 guardrail | InterHand PA / UBody `[wa]` | 手冻结→**必须纹丝不动**；动了=漏进手路径，bug，停 |
| cam_out 副作用 | UBody `[abs]` / wrist-rel | cam_out 改投影→可能白捡 `[abs]` 改善或退化，盯一下 |

**停止/判定**：2 epoch 足够（13 参数极快）；BEDLAM body type 单一，AGORA 量化开始退=过拟，停。AGORA 改善+demo 肉眼变好+手不动→留作配图；AGORA 动但 demo 无肉眼变化→诚实记"合成口径动、真实定性有限"，别 oversell。

### track-B 公平基线首个结果 + decoder 根因 + 概率判断

**同子集对比（400-iter 子集，3200 手）**：`osx_fairbase` itr1500（1/3 epoch，未收敛）InterHand PA **15.53** vs 我们 `snap2` 在**同一子集** **15.85** → OSX-ft 领先 0.32mm（wrist-rel 80.80 vs 81.11）。子集偏易（snap2 全量 16.78），但同口径下 OSX 仍在前。→ backbone 冻结 + coord loss 相同 → **差距是 decoder 架构/实现，非 loss/数据**；"−14% 是我们 decoder 方法贡献"守不住，待 OSX-ft 收敛 + 全量复评定幅度。

**Decoder 代码审查根因**（详见 `post_stage3_roadmap.md` §0‴）：① 我们 decoder 是"特征精修器"非"坐标精修器"——`hand_decoder`(model_core:725) 只输出特征、无坐标输出/监督，被监督的 2D 坐标全来自 soft-argmax(721)，唯一梯度是 pose→FK 间接；OSX Poseur 每层回归坐标+每层 aux 监督。② 参考点全程不迭代(my_decoder:661-672)。③ soft-argmax 16³ 量化。④ 3 层 vs OSX 6 层。⑤ over-engineered topo/occlusion(my_decoder:454-458)，头号 ablation。最高杠杆：改迭代坐标精修+每层 aux 监督；捷径查 `model_core:1099` 的 `PoseurDecoder`。

**概率判断**（结构天花板：冻结 backbone 逐比特同 OSX → OSX≈我们在其域天花板）：修 decoder 后 InterHand 追平 ~50%/超过 ~30%；UBody `[wa]` 追平 ~40-45%/超过 ~15%；**两域同时超过 ~10-15%，两域追平 ~40-50%(高概率)**。建议不押"两域都超 OSX"，主打 in-domain 追平 + held-out(HInt) 赢 + 诚实复现；抬概率：迭代坐标精修 / 叠 RLE loss / ablate topo-occlusion。

## 2026-06-29

### Group attribution + direct WA2D loss 实验（注：bootstrap/NaN 已于 2026-06-30 补测并部分推翻本条，见上）

**逐关节 group attribution（per-hand dump，按 `wa2d_group_attribution.md` 方案）**：snap2 vs OSX 整体 `[wa]` delta +0.0099，按 level：j1 +0.0009 / j2 +0.0040 / j3 +0.0117 / tip +0.0252 → gap **集中在 distal（tip+j3 占 ~88%）**，不是小手/遮挡/左右手某一类。`hand_tipw itr3000` vs OSX 整体 +0.0086、tip +0.0216（tipw 缩小了一点 tip gap）。

**WA2D 实验三组**：

- **Aggressive**（`output/hand_wa2d_distal`，weight 1.0、level j1=.25/j2=.5/j3=1.5/tip=3.0、sources **ubody**）：itr500 崩，`[wa]` 0.260 / tip 0.364 / PA手 10.93。权重过强 + 当时无 non-finite guard，弃。
- **`hand_wa2d_distal_2`**：source 传中文逗号 `ubody，mscoco`，修复前导致 WA2D inactive，结果无效，弃。
- **Guarded**（`output/hand_wa2d_distal_guard`，warm-start `hand_tipw itr3000`、weight 0.1、sources ubody,mscoco、level j1=0/j2=0/j3=1/tip=1、tip_w 2.0/j3_w 1.5、posnet_lr_mult 0.25）：itr820 backward 出 non-finite（`hand_position_net.dcnv4_blocks.0`）。itr500 评测 `[wa]` 0.226 / tip 0.281；vs OSX 整体 +0.0075 / tip +0.0188。3D 手不坏、略好。

**新增 `--freeze_hand_position_net`**（默认关闭）：冻 `hand_position_net`（不进优化器、仍存进 snapshot），只训 decoder/regressor，意在绕开 DCNv4 NaN。Freeze run（warm-start guarded itr500）：itr500 vs OSX 整体 **+0.0068** / tip +0.0174；itr1000 平台（+0.0069）。**当时判为当前最佳候选**——此结论于 2026-06-30 被推翻（freeze 只把 NaN 挪到 decoder itr1295，且收益经 bootstrap 确认极小、仍显著输 OSX）。

## 2026-06-28

### Route C 收口 + hand_tipw + 新增 direct WA2D loss 代码

**Route C（`--train_hand_roi`，解冻 `hand_roi_net`）**：Clean C（IH/UBody only、`hand_roi_lr=1e-6`）UBody 无改善，`[wa]` 仍 ~0.229；C+tipw+MSCOCO（`hand_roi_lr=3e-6`）约 itr700 后失稳，itr1000 `[wa]` ~0.260 / tip ~0.364。判定：收益/风险比差，非优先路线。

**hand_tipw**（`output/hand_tipw`，warm-start snap2、mix IH 0.35/UBody 0.25/MSCOCO 0.40、frozen ROI、`--hand_tip_loss_weight 2.0 --hand_j3_loss_weight 1.5`）：

| ckpt | UBody `[wa]` | tip | `[abs]` | PA手 |
|---|---:|---:|---:|---:|
| snap2 | 0.229 | 0.288 | 0.324 | 10.15 |
| tipw itr3000 | 0.227 | 0.284 | 0.315 | 10.08 |
| tipw itr4000 | 0.227 | 0.284 | 0.315 | 10.10 |

选 **itr3000**（与 itr4000 的 UBody 持平，但 InterHand/EHF/PA 略好）。guardrail itr3000：InterHand PA 16.95 / HInt `[wa]` 0.480 / EHF Hands 15.62 / Face 6.15。

**Bootstrap（配对）**：snap2 vs tipw3000 `[wa]` delta +0.0016 CI[+0.0003,+0.0029] P=0.993（tipw 显著好但微小）、tip +0.0039、abs +0.0082；OSX vs tipw3000 `[wa]` delta −0.0086 CI[−0.0094,−0.0078]（仍显著输 OSX）。tipw 只补回原 gap 的 ~16%。

**新增 direct wrist-aligned 2D hand loss（默认关闭）**：full-body heatmap 坐标、wrist translation 对齐、GT 手 bbox 对角线归一化，对齐 UBody `[wa]` 指标；sources 默认 ubody,mscoco，支持 j1/j2/j3/tip level 权重。改 6 文件（`config.py`/`train.py`/`model_core.py`/`UBody.py`/`MSCOCO.py`/`dataset.py`），`py_compile` 通过，当时未跑训练。详细 handoff 见 `docs/archive/hand_tipw_wa2d_handoff.md`、`docs/archive/recent_experiment_summary_2026-06-29.md`；方法设计见 `docs/wa2d_group_attribution.md`。

## 2026-06-27

### 【重大】编码器移植 bug 发现并修复：StandardViT 与 OSX MMCV ViT 现逐比特一致

**根因**：`common/nets/vit.py` 的 `StandardViT.PatchEmbed` 卷积用了 `padding=0`，而 OSX 的 MMCV ViT 用 `padding = 4 + 2*(ratio//2 - 1) = 2`（`ratio=1`，见 `transformer_utils/.../vit.py:130` 与 `body_encoder_large.py`）。两者输出都是 (16,12)/192 patch，所以形状检查全过、一直隐形；但同一份卷积权重在 padding 2 vs 0 下，每个 patch 在偏移 2px 的窗口上卷积 → patch 特征系统性错。次要：`LayerNorm` eps 端口默认 1e-5，OSX 用 1e-6。

**怎么发现的**：新增编码器忠实度探针 `tool/analysis/encoder_compare.py`（配合 `test.py --dump_encoder_n N`）。在**相同输入**(body_img max|A−B|=0)上量到 `img_feat` 相对 L2 ≈ **0.24**、`task_tokens` ≈ 0.05 —— patch≫token 的不对称直接指向 PatchEmbed，再对配置确认。

**修复与验证**（fix commit `e9566c7`）：`padding=2` + `eps=1e-6`。修复后探针 `img_feat`/`task_tokens` 相对 L2 = **0.000e+00（逐比特相同）**；`crosspath_compare` 的腕漂移 **1.4° → 0**。冻结骨干(encoder→body/box/ROI)现在与 OSX-normal 完全等价。

**`snapshot_2` 在修复编码器上的干净基线**（手头未重训，仍是旧编码器训出来的）：

| 指标 | OSX | snap2 @旧(bug) | snap2 @修复 | 判定 |
|---|---:|---:|---:|---|
| InterHand PA-MPJPE | 19.58 | 16.76 | **16.78** | 赢 −14%（in-domain，见 caveat）|
| InterHand wrist-rel | 86.32 | 84.32 | 84.13 | 略好 |
| UBody [abs] NME | 0.311 | 0.324 | **0.316** | ~追平(免费) |
| UBody [wa] NME | 0.219 | 0.229 | **0.229** | 输 0.010（干净=手头）|
| UBody PA Hands | 10.29 | 10.28 | 10.15 | 略好 |
| EHF Face PA-MPVPE | 6.09 | 6.25 | **6.15** | ~追平(免费) |
| EHF Hands PA-MPVPE | 15.97 | 15.69 | 15.67 | 持平 |
| HInt [all wa] NME | 0.487 | 0.480 | 0.480 | 赢(held-out) |
| HInt [all abs] NME | 0.451 | 0.436 | 0.445 | 略退(hand-crop 噪声) |

**判读（战略级，多条旧结论被推翻）**：

1. **手头对编码器修复完全鲁棒、无需重训**：InterHand PA、UBody/HInt `[wa]` 修复前后纹丝不动（per-finger ±0.001）。`hand_position_net` soft-argmax 重定位保住了相对手形。snap2@修复是可用的工作基线，直接往前走。
2. **编码器修复是净赚**：body 依赖口径(`[abs]`、EHF Face)免费向 OSX 收敛(0.324→0.316、6.25→6.15)，之前归因为"自然手退化/[abs]退化"的，**有一块根本是这个 bug**。
3. **唯一真问题被干净隔离**：UBody `[wa]` 那 0.010 在 body 与 OSX 逐比特一致后**仍在** → 100% 是手头自然图相对手指 articulation（不是 body/port/腕漂移）。bootstrap 配对 CI 仍为 `[-0.0117, -0.0087]`、显著。
4. **InterHand −14% 的 caveat（不受本次修复影响）**：OSX 原始训练集(MSCOCO/H36M/MPII/UBody/AGORA)**不含 InterHand26M**，故 −14% 本质是"我们微调 vs OSX 零样本"。论文要么补一个"同数据微调的 OSX"做公平基线，要么 headline 让位给 held-out 泛化(HInt)。

**新增工具**：`tool/analysis/{bootstrap_ci,crosspath_compare,encoder_compare}.py`（纯 numpy）+ `test.py --dump_analysis`(逐手指标/冻结 body 参数 npz) / `--dump_encoder_n`(编码器 I/O npz)。**Route C** `--train_hand_roi`(解冻 `hand_roi_net` 小 LR 共适应，默认关闭、零影响)已实现。

**下一步**：① 干净地打那 0.010——Route C(warm-start snap2、修复编码器、kill 线 UBody `[wa]`≤0.224) 或末端 2D loss 加权(差距集中 tip/j3) 或更多自然手监督，现在都能无 confound 测量；② 处理 InterHand 公平基线。

## 2026-06-26

### HInt 全链补测：HInt win 起点在 `_c snapshot_8`，COCO pilot 对 HInt 无新增收益

补齐 HInt hand-crop held-out 评测链，结果目录：

- `_c snapshot_8`、`face_ubody_e snapshot_4`、`joint_polish_f snapshot_0/1`：`output/eval_hint/result/`
- `joint_polish_f snapshot_2`：`output/eval_joint_polish_f/result/snapshot_2/HInt_result.txt`
- OSX baseline 与 `coco_pilot`：`output/eval_coco_pilot/result/`

HInt 有效样本均为 3656 hands，且左右手标签均已从文件名 `_l/_r` 解析成功（`Known side labels used: 3656 / 3656`）。

| 模型 | abs PCK@0.2 | abs NME | wa PCK@0.2 | wa NME |
|---|---:|---:|---:|---:|
| OSX baseline | 0.132 | 0.451 | 0.121 | 0.487 |
| `_c snapshot_8` | **0.140** | **0.436** | **0.125** | **0.480** |
| `face_ubody_e snapshot_4` | **0.140** | **0.436** | **0.125** | **0.480** |
| `joint_polish_f snapshot_0` | 0.139 | 0.436 | 0.125 | 0.481 |
| `joint_polish_f snapshot_1` | 0.139 | 0.436 | 0.125 | 0.481 |
| `joint_polish_f snapshot_2` | 0.139 | 0.436 | 0.125 | 0.480 |
| `coco_pilot snapshot_1` | 0.139 | 0.436 | 0.125 | 0.481 |
| `coco_pilot snapshot_1_itr3000` | 0.139 | 0.435 | 0.125 | 0.480 |

**判读**：

- HInt 相对 OSX 的小幅胜利在 `_c snapshot_8` 已经出现：`abs NME 0.451→0.436`，`wa NME 0.487→0.480`。
- `face_ubody_e` 冻手训脸，HInt 与 `_c` 完全一致，符合预期。
- Stage 3 `joint_polish_f` 大幅修复 UBody natural-hand（`0.250→0.229`），但 HInt 基本不动；说明 HInt 当前 hand-crop 评测口径不敏感于 Stage 3 修复的 full-body natural-hand pipeline 问题。
- `coco_pilot` 对 HInt 基本无新增收益；它的边际收益只体现在 InterHand 微降（约 `16.76→16.67/16.68`）与 UBody `[wa]` 微降（`0.229→0.227`）。
- 之前尝试开启 `--mscoco_use_hand_roi_quality` 后训练 loss 在约 1.3k iter 后整体飙高；日志显示 MSCOCO gate pass 率常仅 `0.12-0.37`，且 InterHand/UBody 也被带坏。当前稳定 `coco_pilot` 结果来自 **MSCOCO hard gate 关闭** 的配方。

**结论修正**：Stage 3 不能再概括为“手只会 InterHand、不泛化”。更准确是：`_c` 起已经在 HInt hand-crop held-out 上小幅超过 OSX，Stage 3 的真实贡献是修复 UBody full-body natural-hand 退化；但 UBody 仍未追平 OSX。后续若继续手部主线，主判据应转向 UBody/full-body natural-hand pipeline，HInt 作为 local hand-crop held-out guardrail。

## 2026-06-25

### Stage 3 `joint_polish_f` 收口：epoch 2 平台确认，终点定 snapshot_2

epoch 2（Phase 2，新模块 lr 已降到 ~5e-7）跑完，评估 snapshot_2 及其 `--save_iters 1280` 中途点（itr1280 / itr2560）三件套。结果目录 `output/eval_joint_polish_f/result/snapshot_2*/`。

| 指标 | OSX | snap1 | s2_itr1280 | s2_itr2560 | **snap2(末)** |
|---|---:|---:|---:|---:|---:|
| InterHand PA-MPJPE | 19.58 | 16.83 | 16.79 | 16.81 | **16.76** |
| InterHand wrist-rel | 86.32 | 84.29 | 84.36 | 84.26 | 84.32 |
| UBody `[wa]` NME | 0.219 | 0.229 | 0.229 | 0.228 | **0.229** |
| UBody `[abs]` NME | 0.311 | 0.324 | 0.324 | 0.323 | **0.324** |
| UBody tip NME | 0.263 | 0.287 | 0.288 | 0.286 | **0.288** |
| EHF Face PA-MPVPE | 6.09 | 6.23 | 6.26 | 6.26 | **6.25** |
| EHF Hands PA-MPVPE | 15.97 | 15.62 | 15.70 | 15.70 | **15.69** |

**判读**：

- **UBody `[wa]` 整个 epoch 2 死平在 0.228–0.229**，符合决策规则 2 预判的平台；Phase 2 极小 LR 对自然手目标几乎零作用。UBody 实测 2D 的微峰在 itr2560（`[wa]` 0.228 / tip 0.286），但与 snap2 差 0.001，纯噪声。
- Phase 2 本质是 wash：InterHand PA 微降 16.83→16.76（-0.07），代价是 EHF 微退（Face +0.02 / Hands +0.08），UBody 不动。三者均噪声量级，**未实现 Stage 3 突破自然手平台的目标**。
- 规则 3 的 `[wa]≤0.225` 远未达到 → **停训，不跑 epoch 3**；`itr3840` 离 epoch 末仅 7 iter、结果必≈snap2，无需评测。

**终点定 `snapshot_2`**（在 snap1/snap2 间按 InterHand+EHF guardrail 取舍）：snap2 拿下最好的 InterHand 头条数 16.76mm（-14% vs OSX 19.58），UBody 与 snap1 持平（0.229），EHF 那点退化（Face 6.23→6.25 / Hands 15.62→15.69）在 guardrail 容差内。snap1 是"无任何退化"的保守替代，二者实质等价。

**Stage 3 总结**：手部基准（InterHand 16.76 / -14%）稳固；自然手退化已大幅回收到接近 OSX（`[wa]` 0.250→0.229 vs OSX 0.219）但仍未追平、且确认到平台；身体/脸保持平价。这是当前冻结框架能拿到的结果。下一步进入身体 shape 独立小实验（T1，代码已实现）。

### 【待办·已立项】身体定性观感小实验 T1（shape + cam，独立于手部线）

**目标**：让真实图上的身体 mesh 定性观感优于原始 OSX。前提认知：身体在当前 frozen 设置下**结构上无法在 pose 上超过 OSX**，本实验只针对可撬动的 shape/cam，不碰 articulated pose。

**诊断（基于 demo 定性核对，非定量；所有 snapshot 身体一致＝body 冻结）**：

- 证据图：`demo/output_face_ubody_e/compare/{render,kpts}/{24,39,69,92}_img.jpg`（精瘦男多姿态、全身可见）、`demo/snapshot4/{kpts,render}.jpg`（换胎深蹲复杂姿势）、`demo/output-ori/`（bar 遮挡场景）。
- **投影/全局朝向：基本 OK**——kpts 贴脸、mesh 姿态大体跟随，相机系没坏。
- **体型 betas：系统性偏「软/厚」，抓不住精瘦体格**——全身可见的精瘦男被渲染成通用偏厚身材；bar 场景的「圆肚子」是躯干全遮挡 → betas 退化到均值的极端 case。即「轮廓缺陷」＝ betas mean-reversion，而非投影问题。
- **复杂姿势（换胎深蹲）**：主误差是 articulated body pose ＋ 全局平移/尺度，shape 次要。

**关键架构事实（决定可行性）**：

- `BodyRotationNet`（`common/nets/module.py:67`）的 `shape_out` 与 `cam_out` 是**独立线性头**（`shape_out: Linear(feat_dim→10)` 只吃 `shape_token`；`cam_out: Linear(feat_dim→3)` 只吃 `cam_token`），与 `root_pose_out/body_pose_out` 解耦 → 可单独解冻、不动 pose。
- `loss['smplx_shape']`（`model_core.py:572`）**早已在算**，只因 `body_regressor` 冻结而空转；解冻 `shape_out` 即激活梯度通路，几乎不用加 loss 代码。
- shape GT 来源：BEDLAM `shape_valid=1`（合成、可靠）；UBody `shape_valid` 条件性 1（**拟合伪 GT，常偏均值，会污染**，见 `data/UBody/UBody.py:449/557`）；InterHand 恒 0。betas 维度 = 10。

**手部—投影耦合分析（决定为何不碰 pose）**：

- `get_coord`（`model_core.py:325`）跑完整运动链，手部只做 wrist 平移减法（`lhand_cam - lwrist_cam`，保留全局朝向）。
- **PA-MPJPE（手部赢点）**：Procrustes 去全局 → 只测局部手指 → 与身体 pose/cam **解耦，安全**。
- **wrist-rel MPJPE（卡 84-86）/ UBody `[abs]`（退化）**：保留全局朝向 ＋ 走身体 `cam_trans` 投影 → **瓶颈在冻结身体 pose＋cam，不在手模块**。这解释了 wrist-rel 一直下不去、`[abs]` 退化。
- 推论：`cam_out` 改善全局平移 → 同时利好身体投影**与绝对手位 2D**；解冻 body pose 则会牵动手部线，必须带 guardrail。

**分层与决策**：

| 层 | 解冻 | 目标 | 独立性 |
|---|---|---|---|
| T0 | `shape_out` | 轮廓 | 纯独立 |
| **T1（采用）** | `shape_out` ＋ `cam_out` | 轮廓 ＋ 全局贴合 ＋ 绝对手位 2D | 独立（两个解耦小头） |
| T2 | ＋ body pose / 解冻 `body_position_net` | 真·身体姿态 | 会动手部线，另立项目 |

**为什么不以「复杂姿势大误差」为入口**：那是 articulated pose 问题——①被冻结 `encoder`/`body_position_net` 卡死，只重训 pose 头天花板低；②BEDLAM/UBody/InterHand 不覆盖此类长尾姿势；③等于在 OSX 最强项上以少胜多。不现实。

**T1 配方（待启动，建议手部线收尾后再开）**：

- 新开实验线（如 `--exp_name output/body_shape_t1`），**不要**和手部 run 混。
- 只解冻 `body_regressor.shape_out` 与 `body_regressor.cam_out`，极小 LR（1e-5 量级）。
- shape 监督**只用 BEDLAM**：屏蔽 UBody 的 `smplx_shape_valid`（避免伪 GT 均值污染），或给极低权重。
- **双口径验证**：① BEDLAM/AGORA 的量化 shape 指标（有 GT betas，MPVPE）；② 重渲染 `demo/snapshot4` 与精瘦男多姿态，定性对比 OSX。**定性目标软，务必挂量化指标兜底。**
- 手部 guardrail 照跑（InterHand PA <17.0 / UBody `[wa]`），确认模块隔离未被带歪。

**需要的代码改动**：

- freeze 框架目前是**模块级**（`frozen_modules` 按模块名，`model_core.py:69`）；T1 需**子模块级解冻**（只放 `shape_out`/`cam_out` 进优化器），并同步更新 `common/base.py` 的 `_verify_freeze_status`，否则会因 `body_regressor` 整块仍标记冻结而断言失败。
- 复用 `main/train.py:_configure_training_phase` 的 param-group 思路，新增一个 body-shape group。

**【已实现 2026-06-25，默认关闭、对手/脸零影响】**：采用 **name-prefix 白名单 + flag gating**。`body_regressor` 仍留在 `frozen_modules`，新增 `Model.body_shape_trainable_prefixes`（`cfg.train_body_shape` 真→`['body_regressor.shape_out','body_regressor.cam_out']`，否则 `[]`）。改动 5 处：

- `main/config.py`：加 `train_body_shape = False` 默认。
- `main/model_core.py`：`__init__` 定义前缀；`freeze_modules()` 末尾对匹配前缀参数 `requires_grad=True`（BodyRotationNet 无 BN，eval 无害）。
- `common/base.py`：`_verify_freeze_status` 加 `is_body_shape` 白名单（优先判为 trainable）；`save_model` 额外保存匹配前缀张量。
- `main/train.py`：加 `--train_body_shape`/`--body_shape_lr`；`_configure_training_phase` 加 `body_shape` optimizer group（lr=`body_shape_lr`）并入 grad-clip；`_check_gradient_flow` 加白名单。

关键执行顺序（`base.py:_make_model`）：`freeze_modules()`(602，含解冻) → `model.train()`(605) → `_verify_freeze_status`(608，含白名单)，verify 在解冻后跑、靠白名单放行。save→load 闭环成立：`_load_lightweight_trained_modules`(base.py:180) 按键全量 overlay，eval/resume 时 `shape_out/cam_out` 正确加载。**flag 关闭时前缀为空 → 所有新分支短路 → 手/脸路径字节级不变**。`getattr` 自动语法检查因 Bash 分类器服务故障未跑成，5 处编辑已人工复核（缩进/作用域/逻辑），startup 也会即时暴露问题。

**启动命令**（前置：`config.py` 改 `trainset_3d=['BEDLAM'], trainset_2d=[]`；暖启 snapshot 待定）：

```bash
cd /workspace/myosx/main
python train.py --gpu_ids 0 --lr 1e-5 --train_batch_size 64 --num_thread 8 \
  --end_epoch 2 --phase1_epochs 0 \
  --exp_name output/body_shape_t1 --decoder_setting pytorch --encoder_setting osx_l \
  --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
  --init_trained_path ../output/joint_polish_f/model_dump/snapshot_X.pth.tar \
  --no_train_hand_modules --train_body_shape --body_shape_lr 1e-5
```

startup 应显示 `🔧 body-shape T1: 解冻 ~13k 参数` + verify/grad-flow 通过。

**预期管理**：机制可行、最小、对手部低风险、试错便宜；但本质是「冻结特征上训 10 维线性头 ＋ 监督几乎全是合成 betas」→ **大概率动得了合成口径 shape，真实图轮廓改善不确定、可能有限**。

### 【工具】`main/train.py` 新增 `--save_iters` 中途存盘

- `--save_iters N`：每 N 个 iter 在 epoch 中途额外存 `snapshot_{epoch}_itr{N}.pth.tar`（评估候选点，与 epoch 末快照分开命名、互不覆盖）；`0`＝关闭（默认，保持原行为）。
- 注意：中途快照 `state['epoch']` 仍为当前 epoch，拿来 `--continue_train` 会从下一 epoch 起步（续训仍以 epoch 为粒度）；定位为评估候选点。每个轻量快照 ~399M。
- 用法：在训练命令末尾加 `--save_iters 1280`（约 1/3 epoch，`itr_per_epoch=3847`）。

## 2026-06-24

### Stage 3 `joint_polish_f` snapshot_1（2 epochs）guardrail 通过，UBody 进入平台

训练已完成到 snapshot_1，并评估三件套：

- UBody：`output/eval_joint_polish_f/result/snapshot_1/UBody_result.txt`
- InterHand：`output/eval_joint_polish_f/result/snapshot_1/InterHand26M_test_result.txt`
- EHF：`output/eval_joint_polish_f/result/snapshot_1/EHF_result.txt`

关键对比：

| 指标 | 原始 OSX / `_c` | `face_ubody_e` 旧 | `joint_polish_f` snap0 | `joint_polish_f` snap1 | 判读 |
|---|---:|---:|---:|---:|---|
| UBody `[wa]` NME | 0.219 | 0.250 | 0.229 | **0.229** | 旧退化大幅回收，但 snap0→snap1 平台 |
| UBody `[abs]` PCK@0.2 | 0.422 | 0.383 | 0.405 | **0.406** | 摆放略回收，仍低于 OSX |
| UBody tip NME | 0.263 | 0.337 | 0.289 | **0.287** | 末端继续微小改善 |
| UBody tip/j1 | 1.28 | 1.62 | 1.40 | **1.39** | 末端斜率明显修复 |
| InterHand PA-MPJPE | `_c` 16.82 | - | 16.85 | **16.83** | guardrail 通过 |
| InterHand wrist-rel | `_c` 84.58 | - | 84.57 | **84.29** | 不退，略好 |
| EHF Face PA-MPVPE | OSX 6.09 | 6.20 | 6.23 | **6.23** | 脸稳定，无明显退化 |
| EHF Hands PA-MPJPE | - | - | 15.89 | **15.71** | 比 snap0 略好 |

snapshot_1 UBody 细项：Evaluated hands 3446；`[abs]` PCK@0.2 0.406 / NME 0.324；`[wa]` PCK@0.2 0.560 / NME 0.229；`[wa-finger]` thumb 0.222 / index 0.249 / middle 0.245 / ring 0.241 / pinky 0.244；`[wa-level]` j1 0.206 / j2 0.225 / j3 0.242 / tip 0.287。

训练日志 `output/joint_polish_f/log/train_logs.txt` 未见 Traceback/RuntimeError/NaN。epoch1 相比 epoch0 loss 继续下降：全局 `loss_joint_img` 均值约 0.3946→0.3670，`loss_smplx_joint_img` 约 0.5665→0.5325；UBody ROI gate pass 均值约 0.70，n_valid 约 14。训练本身健康，但评测显示 UBody 主要收益已在第 1 个 epoch 获得。

**判读**：

- `joint_polish_f` snapshot_1 是当前 best/tied best：UBody 不低于 snapshot_0，InterHand / EHF guardrail 更好或持平。
- Stage 3 验证了 UBody 自然手监督 + predicted ROI + mixed data 组合配方有效；但不能把收益单独归因到“GT 框注入导致 crop 分布差”，该单因子还没有隔离实验。
- InterHand PA 仍贴近 `_c` best（16.82→16.83），说明自然手回收没有明显牺牲 InterHand benchmark。
- wrist-rel 仍在 84mm 高位，和 `_c` 一样，说明当前训练主要修相对手形/姿态，未解决绝对手腕朝向/全局手系问题。

**下一步**：

1. 如果训练已经继续，等 snapshot_2 后再评估 UBody / InterHand / EHF。
2. 若 UBody `[wa]` 仍在 0.228-0.229，停止长训，在 snapshot_1/2 中按 InterHand 与 EHF 选最终点。
3. 若 UBody 收到 <=0.225 且 InterHand <17.0mm、EHF Face 约 6.20mm，可再考虑 1 个 epoch。
4. 不建议盲目跑满 4 epoch；后续若要突破平台，再考虑末端 2D loss 加权或降低 UBody 比例/学习率的小实验。

### Stage 3 `joint_polish_f` snapshot_0（1 epoch）UBody 自然手大幅回收

Stage 3 已按 `main/train.sh` 启动：从 `face_ubody_e` snapshot_4 暖启（`--init_trained_path`），lr 5e-5，`posnet_lr_mult 0.5`，三数据集 `BEDLAM:0.20 / InterHand26M:0.35 / UBody:0.45`，UBody 用 box_net 预测框（不注入 GT 框，因 UBody 无 `is_hand_only`），开 `--bedlam_no_hand_img_loss --bedlam_use_hand_roi_quality --ubody_use_hand_roi_quality`。已跑 1 epoch，UBody eval：`output/eval_joint_polish_f/result/snapshot_0/UBody_result.txt`。

UBody 自然手四档对比（OSX / 我们旧=face_ubody_e / Stage3 1ep / 随机手头下界）：

| 指标 | OSX | 我们旧 | Stage3 1ep | 随机 | 判定 |
|---|---:|---:|---:|---:|---|
| 整手 `[wa]` NME | 0.219 | 0.250 | **0.229** | 0.262 | 区间 72%→23%，距 OSX 只剩 0.010 |
| tip/j1 放大比 | 1.28 | 1.62 | **1.40** | 1.77 | 末端退化斜率显著收窄 |
| tip NME | 0.263 | 0.337 | **0.289** | 0.367 | 末端回收 0.048 |
| `[abs]` PCK@0.2 | 0.422 | 0.383 | **0.405** | 0.343 | 摆放也改善 |
| `[abs]` NME | 0.311 | 0.337 | **0.324** | 0.351 | 摆放也改善 |

per-level 改善幅度沿指节深度递减（与退化模式镜像对称）：j1 0.208→0.206（不动）、j2 0.236→0.225（-0.011）、j3 0.270→0.243（-0.027）、tip 0.337→0.289（-0.048）。per-finger 五指全部改善（thumb -0.009 / index -0.023 / middle -0.028 / ring -0.025 / pinky -0.024），之前退化最重的食/中/无名改善最大，无副作用集中点。

**判读**：
- 成功判据 1（wa NME <0.240）✅ 达标（0.229）；判据 2（tip/j1 从 1.62 往 1.28 收）✅ 显著回收（1.40）。
- 退化模式被针对性修复：UBody 真实手监督确实修到末端屈伸链，不是整体平移。后续 snapshot_1 进一步显示，组合配方有效，但“GT 框注入/crop 分布差”单因子尚未隔离。
- 1 epoch 即见效且未收敛，继续跑大概率再收一点，但边际递减、过拟合风险随 epoch 上升。

**后续补测**：InterHand guardrail 已补齐。snapshot_0 为 16.85mm PA / 84.57mm wrist-rel，snapshot_1 为 16.83mm PA / 84.29mm wrist-rel，均未破 17.0mm。

**下一步状态**：该条目的待办已由 snapshot_1 条目覆盖；当前优先级是评估 snapshot_2 是否突破 UBody 平台。

### UBody `[wa]` per-finger / per-level 三档诊断（退化模式已收敛）

per-finger 拆分实现后，重跑三档 UBody natural-hand 2D（`test_sample_interval=10`，3446 手）：

- 原始 OSX `normal`：`output/eval_ubody/result/UBody_result.txt`
- `face_ubody_e` snapshot_0 / snapshot_4（手权重=`_c snapshot_8`，冻手等价，per-finger 完全相同）：`result/snapshot_0/`、`result/snapshot_4/`
- `pretrained`（pytorch 映射预训练分支，保留未覆盖随机-ish 手头，退化参照）：`result/pretrained/`

整手 `[wa]` NME：OSX 0.219 → 我们 0.250 → `pretrained` 0.262。我们处在 OSX→`pretrained` 退化区间的 **72%**，离 `pretrained` 只差 0.012、离 OSX 差 0.031，是显著退化而非轻微漂移。

per-level（核心模式：退化沿指节深度递增，指根不动）：

| level | OSX | 我们 | `pretrained` | 我们-OSX | 区间位置 |
|---|---:|---:|---:|---:|---:|
| j1 指根 | 0.205 | 0.208 | 0.207 | +0.003 | ~噪声 |
| j2 | 0.221 | 0.236 | 0.241 | +0.015 | 75% |
| j3 | 0.231 | 0.270 | 0.286 | +0.039 | 71% |
| tip 指尖 | 0.263 | 0.337 | 0.367 | +0.074 | 71% |

tip/j1 放大比：OSX 1.28 → 我们 1.62 → `pretrained` 1.77。指根三档几乎重合，退化几乎全部累积在末端。

per-finger：thumb 退化最小（+0.020，区间 83%）；index/middle/ring 退化最大（+0.033~0.038，区间 60-70%）；pinky 的"区间 103%"是噪声（`pretrained` 小指反而 0.267 略低于我们 0.268，小指标注稀疏/方差大，不可信）。

**诊断结论**：我们的退化曲线和 `pretrained` 随机-ish 手头参照几乎平行（j2/j3/tip 区间位置稳定 71-75%），只是整体下移一点 → 是**整体末端预测能力下降，接近"末端优先坏"模式**，不是某根手指/某指节的特异性缺陷。当前证据不优先支持：单根手指标注偏置、纯全局手腕摆放/平移问题（`[wa]` 后仍退）、单点过拟合。最可能根因：手部分支几乎只在 InterHand 域（GT 框、手占满画面）学到末端细节，在自然图（预测框 crop、小手、真实背景）末端屈伸链泛化掉；可能叠加 GT 框注入训练没见过 box_net 预测框 crop 分布，使 `[wa]` 整体下移。

**Stage 3 量化判据（重跑同一 UBody per-finger 比对）**：

1. 整手 `[wa]` NME 从 0.250 往 OSX 0.219 收，目标 < 0.240（区间位置 72% → < 45%）。
2. tip/j3 区间位置从 71% 下降，tip/j1 放大比从 1.62 往 1.28 收（末端退化斜率变缓）。
3. InterHand-test PA-MPJPE 不破 17.0mm；EHF Face 保持 ~6.20mm。

不达 1/2 则 Stage 3 没有真正修到末端自然图泛化，需回退配方。

### UBody `[wa]` per-finger / per-level 诊断实现

- `data/UBody/UBody.py` 已在 natural-hand 2D 评测中新增 `[wa-finger]` NME 与有效计数：
  `thumb/index/middle/ring/pinky`。
- 同时新增 `[wa-level]` NME 与有效计数：`j1/j2/j3/tip`。
- wrist alignment 仍只做一次，归一化仍用整只 GT hand keypoint bbox diagonal。
- 已加入 UBody hand order 显式校验，确认 GT 顺序为 `wrist, thumb, index, middle, ring, pinky`。
- 判读 caveat：per-finger/per-level 是各自可见子集统计，不能用五指均值直接对账整手 `[wa]` NME；看横向差异时必须同时看 `N`。
- ✅ 已完成：重跑原始 OSX normal、`face_ubody_e` snapshot_0/4、`pretrained`（随机-ish 手头参照）三档，退化模式诊断见上一条目。

### InterHand 全量对等基线确认

结果文件：

- 原始 OSX normal：`output/eval_interhand_bedlam_c/result/InterHand26M_test_result.txt`
- `_c` snapshot_8：`output/eval_interhand_bedlam_c/result/snapshot_8/InterHand26M_test_result.txt`
- `_c` snapshot_10：`output/eval_interhand_bedlam_c/result/snapshot_10/InterHand26M_test_result.txt`

| 模型 | Evaluated hands | PA-MPJPE | wrist-rel MPJPE |
|---|---:|---:|---:|
| 原始 OSX `normal` | 52033 | 19.58mm | 86.32mm |
| `_c` snapshot_8 | 52033 | 16.82mm | 84.58mm |
| `_c` snapshot_10 | 52033 | 16.82mm | 84.11mm |

结论：

- 原始 OSX 全量 InterHand baseline 已跑完，不再是待办。
- `_c` 相对 OSX 的手部增益为 **-2.76mm PA-MPJPE / -14%**，这是手部贡献的硬证据。
- wrist-rel 只改善约 1.7-2.2mm 且仍在 84mm+，说明 InterHand 收益主要体现在 PA 对齐后的相对手形/姿态；绝对手腕朝向/全局手系问题仍存在。
- `result/pretrained/InterHand26M_test_result.txt` 的 25.17mm 是 `decoder_setting=pytorch` 的映射预训练分支，不是原始 OSX normal baseline。

### `face_ubody_e` 暖启和手冻结确认

- 训练日志 `output/face_ubody_e/log/train_logs.txt` 显示：`face_ubody_e` warm-start 自 `../output/interhand_bedlam_c/model_dump/snapshot_8.pth.tar`。
- 同一日志显示 `train_hand_modules=False`，即 Stage 2 冻手训脸。
- 已运行：

```bash
python tool/check_hand_frozen.py \
  --ref output/interhand_bedlam_c/model_dump/snapshot_8.pth.tar \
  --ckpt output/face_ubody_e/model_dump/snapshot_0.pth.tar \
         output/face_ubody_e/model_dump/snapshot_4.pth.tar \
         output/face_ubody_e/model_dump/snapshot_8.pth.tar \
  --also_face
```

结果：snapshot0/4/8 的 `hand_position_net`、`hand_decoder`、`hand_regressor` 均 **Δ=0**；face 分支均已更新。结论：`face_ubody_e` 的 UBody 手部评测可以视作 `_c snapshot_8` 手部权重的自然手表现。

### UBody annotation cache 一致性检查

- 重新生成 filtered pkl 后，15 个 scene 都带 `_ubody_orig_ann_order`。
- Movie scene train 路径：原始 JSON 与 filtered pkl datalist 长度均为 5222，逐样本签名一致。
- Olympic scene test 路径，`test_sample_interval=10`：原始 JSON 与 filtered pkl datalist 长度均为 136，逐样本签名一致。
- 结论：当前 UBody train/test 读取 filtered pkl 与读取原始 JSON 的最终样本输出一致；改 images/annotation 后需要重跑 filter cache。

### UBody natural-hand 2D 首轮结果

评估目录：`output/eval_ubody`

归属说明：

- `output/eval_ubody/result/UBody_result.txt` 是原始 OSX `normal`，top-level eval 没有 `--continue_train_path`。
- `output/eval_ubody/result/snapshot_*/UBody_result.txt` 是 `decoder_setting=pytorch` 加载 `face_ubody_e` 对应 snapshot。

| 指标 | 原始 OSX | `face_ubody_e` snapshot_0 | snapshot_4 | snapshot_8 |
|---|---:|---:|---:|---:|
| PA MPVPE Hands | 10.29 | 10.28 | 10.28 | 10.28 |
| PA MPJPE Hands | 10.55 | 10.49 | 10.49 | 10.49 |
| `[abs]` PCK@0.2 | 0.422 | 0.383 | 0.383 | 0.383 |
| `[abs]` NME | 0.311 | 0.337 | 0.337 | 0.337 |
| `[wa]` PCK@0.2 | 0.587 | 0.515 | 0.515 | 0.515 |
| `[wa]` NME | 0.219 | 0.250 | 0.250 | 0.250 |

结论：

- 微调后真实自然手 2D 低于原始 OSX。
- 伪 GT hand PA-MPJPE 不反映该退化。
- 当前保留的 snapshot0/4 自然手完全一致；snapshot8 由手权重冻结等价检查支持同一结论，说明 Stage 2 脸训练不是退化来源。
- 下一步先加 per-finger / per-level 拆分，再决定是否做自然手恢复短训。

### UBody hand joint order 核对

- `smpl_x.orig_hand_regressor` 的实际顺序是 `wrist, thumb, index, middle, ring, pinky`。
- UBody COCO-WholeBody 手 GT 顺序也是 `wrist, thumb, index, middle, ring, pinky`。
- `human_models.py` 中旧 `orig_joints_name` 对 SMPL-X 22+ 行有误导，不能用它解码 `J_regressor` 行号。
- 结论：UBody natural-hand 2D 当前没有跨手指错配；但建议后续用显式 finger index 和注释降低误读风险。

## 2026-06-23

### UBody pkl / filtered cache

- 新增 UBody annotation pkl 读取：优先读 `keypoint_annotation.pkl` / `smplx_annotation.pkl`，缓存过期回退 JSON。
- 新增 `tool/UBody/filter_annotations_by_images.py`：按已抽帧图片预过滤 annotations，可 `--pkl_only`。
- `Movie` 参考速度：原始 JSON 初始化约 22.9s，annotation pkl 约 9.6s，filtered+pkl 约 1.7s。
- 修复 filtered cache 下 sample interval 语义变化：过滤脚本写 `_ubody_orig_ann_order`，loader 用原始 annotation 序号抽样。

### UBody test bug 修复

- 修复 `UBody` 外层缺 `joint_set` 导致 natural-hand 2D evaluate 报错。
- 修复 `generate_mesh_gt()` 中 CPU/CUDA tensor 混用。
- 修复 UBody train target 缺 `smplx_cam_trans`，避免 epoch-vis 走 test 分支时报错。

## 2026-06-20 至 2026-06-23 摘要

- `interhand_bedlam_c` 完成，InterHand-test PA-MPJPE 16.82mm。
- 编码器 pos_embed bug 已修复，EHF body 回到原始 OSX 附近。
- `face_ubody_e` 完成，EHF Face 回到约 6.20mm，但未超越原始 OSX。
- 战略重判：脸不是主要贡献，BEDLAM 在 frozen 设置下价值有限，下一步聚焦自然手质量。
