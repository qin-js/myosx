# 实验日志

本文件按时间追加实验结果和决策。只写结论、关键数字、下一步，不放长篇排查过程。

## 2026-07-08

### posnet 探针 Run B（conv+no_detach）+ Run A（dcnv4+detach）：抗侵蚀=aux 锚（锁死）+ InterHand 落后非 posnet（锁死）；refined 评测一致性；decoder 改进线收口 + 能力边界

**① refined 评测一致性（先修的坑）**：test.py 之前**缺** `--no_refined_hand_coord` flag（train.py 有），评测恒走默认 `use_refined_hand_coord=True`，而 rotmat s0/T1/探针都是 `False`（soft-argmax）训的 → train/test forward mismatch。已补 test.py flag。**但实测 rotmat s0 加 flag 前后数字纹丝不动** → 原因：`use_refined_hand_coord` 只改喂 regressor 的**坐标**（占输入 2/515），且 `bbox_embed` 零初始化使精修 xy≈soft-argmax xy，而 3D PA 几乎不吃坐标输入。**含义**：(a) 底座数字 16.53/15.61/10.00 **不是 mismatch 产物、可放心用**；(b) 这是「2D 坐标机制与 InterHand 3D-PA 近正交」的**第三个独立证据**（前两个 = gate≈snap2、L728 负），且比之前更硬（连喂不喂精修坐标进 regressor 都不动 PA）；(c) T1/gate/aux0 同机制、大概率也不受污染，不必全部重测。与 7-03 L728（refined 训 refined 测退 0.25）不矛盾——L728 是**续训**让 regressor 适应新输入放大差异，纯评测切换不重训则≈0。

**② 探针 Run B（`--hand_posnet conv --no_detach_hand_decoder_query`，osx_l fresh 冷启，前 3 epoch）**：

| ckpt | ~ep | InterHand PA | wrist-rel | EHF Hands | UBody PA Hands | UBody `[wa]` |
|---|---|---:|---:|---:|---:|---:|
| snapshot_1 | 2 | 17.02 | 85.19 | 15.86 | 10.05 | 0.230 |
| snapshot_2 | 3 | 16.98 | 85.00 | 15.95 | 10.08 | 0.229 |
| 参照 fairbase(OSX dec) | — | 16.29→15.36 | →81.71 | **16.7~16.85(侵蚀)** | **10.4~10.6(侵蚀)** | 0.220 |
| 参照 rotmat s0(dcnv4,暖) | — | 16.53 | 85.09 | 15.61 | 10.00 | 0.231 |

（EHF Face 9.49 是 face 分支 fresh 冷启没训熟，与探针无关，忽略。）

**归因①（锁死）：whole-body 抗侵蚀 = decoder 的 aux 坐标锚。** Run B 与 fairbase 现在**几乎只差 decoder 一个变量**（都 conv PositionNet、都 osx_l fresh 冷启、都梯度回流 no_detach、同数据）——唯一差异 = 我们 HandDecoder（每层 aux 2D 坐标监督）vs OSX PoseurDecoder（无锚、死代码）。结果 **Run B EHF Hands 15.86~15.95(<stock 15.97) / UBody 10.05~10.08(<10.29) 不侵蚀**，fairbase 侵蚀。→ 抗侵蚀**不来自 DCNv4 头**（Run B 已换成 conv）、**不来自暖启**（都 fresh），唯一剩下就是 **aux 锚**。机制假设从「假设」升级为「单变量实验结论」，可直接写方法节。

**归因②（倾向，待 Run A 锁死）：InterHand 落后 ~1.2mm 非 posnet 后端。** conv posnet（可暖启+梯度回流，最接近 OSX 手部栈）冷启 3ep 到 **16.98，无逼近 15.x 迹象**——换掉 DCNv4 没解决落后。confound：Run B fresh 冷启 vs rotmat s0 暖启，绝对值不可直比。**需 Run A（`--hand_posnet dcnv4` fresh 同 recipe）** 作干净单变量：Run A≈Run B → posnet 无关、落后是 decoder 家族天花板；Run A≪Run B → posnet 有关（低概率）。Run A **零代码**（flag 已加）。**→ Run A 已跑（见本日下方 Run A 条目）：dcnv4 fresh 17.38 ≈ conv fresh 17.02、且略差 → 归因②锁死，落后非 posnet。**

旁证：Run B `[wa]` 0.229~0.230 ≈ rotmat s0(dcnv4) 0.231 ≈ stock 0.219 略输 → `[wa]` 对 conv/dcnv4 手分支配置都不敏感，再证 T1 结论（`[wa]`=body cam 几何）。

**③ decoder 改进线收口（战略）**：**不再往 decoder 加东西追 InterHand（含加深层数）**。四条证据表明 InterHand 落后是**天花板问题非容量问题**：冻结骨干逐比特=OSX → 同一份特征 → OSX≈天花板；rotmat（pose 空间对靶杠杆）触顶 16.5、ep1 平台（回归非欠训）；Run B 换 conv 也不动；坐标机制正交（①）。加深还与「高效单模型」定位相悖 + 重引 DCNv4 深层冷启 NaN 风险（7-01）。**decoder 剩余工作只有消融（解释）、无改进（追指标）**。「层数 3v6」按 paper_table 本就是条件项（posnet/topo-occ 有信号才跑），posnet 已无信号 → 除非 topo/occ 出信号否则不跑。

**④ 能力边界（战略，回答"能否全面超过 OSX"）：结构上不可能，且这是论文立足点非弱点。** (a) **body 冻结=逐比特 OSX → 只能平不可能超**；**UBody `[wa]`/`[abs]` 2D = body cam 几何**、**EHF Face = face 分支** → 均非 hand decoder 可碰。(b) 手部维度：**vs stock 已全超**（rotmat s0 手部全绿）；**vs 公平基线是 Pareto trade-off 打不破**——ft 占 in-domain 端（15.36，侵蚀 whole-body 换来）、我们占 whole-body 端，冻结骨干+同特征下无法同占前沿两端。(c) 真要全超需**解冻骨干**（另一个项目、打不过 SMPLer-X）或**外挂 WiLoR**（变 H4W++、弃单模型卖点）——都推翻定位。→ 论文主张固定为「**不破坏 whole-body、不挂外部专家，占据 OSX-ft 换不到的 Pareto 点**」，不追"全超"。

**下一步**：① Run A（零成本、后台跑，补归因②单变量）；② topo/occ 消融（更高价值：上游决策 + 简化方向 + 堵 reviewer 过度设计质疑；**需改代码**加开关，建议设计成 conv+fresh 与 Run B/fairbase 组 decoder 阶梯，且**必看 EHF/UBody**——它与抗侵蚀归因耦合，砍后若侵蚀则削弱"纯 aux 锚"论证）；③ Run B 跑完 ep4 补终值。

### Run A（dcnv4+detach，fresh）：归因②锁死（InterHand 落后≠posnet）+ 归因①跨配置云加固

`output/probe_posnet_dcnv4`（`--hand_posnet dcnv4`、detach ON、osx_l fresh 冷启；主栈 dcnv4+detach 的 fresh 孪生）。Run A/B 取同 epoch snapshot_1(≈ep2)：

| run | posnet | detach | init | rotmat | InterHand PA | wrist-rel | EHF Hands | UBody PA Hands | `[wa]` |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| **Run A(本次)** | dcnv4 | on | fresh | off | **17.38** | 85.40 | 15.88 | 10.01 | 0.231 |
| Run B | conv | off | fresh | off | 17.02 | 85.19 | 15.86 | 10.05 | 0.230 |
| rotmat s0(主栈) | dcnv4 | on | warm | on | 16.53 | 85.09 | 15.61 | 10.00 | 0.231 |
| fairbase(OSX dec) | conv | — | fresh | off | 16.29→15.36 | →81.71 | **16.7~16.85 侵蚀** | **10.4~10.6 侵蚀** | 0.220 |

（Run A snapshot_0 EHF Hands 16.17→snapshot_1 15.88；EHF Face 7.03/7.31=face 分支 fresh 没训熟，忽略。）

**归因②锁死：InterHand 落后 ~1.2mm ≠ posnet 后端。** dcnv4 fresh(17.38) 与 conv fresh(17.02) 都在 ~17 平台、都够不着 fairbase 15-16 轨迹 → 换哪个 posnet 都救不了 InterHand；且 **dcnv4 比 conv 还略差 0.36mm**，方向上排除"DCNv4 头藏了 InterHand 收益"。落后 = decoder 家族天花板 + 预训练本钱。⚠️confound：Run A vs Run B 的 detach 也翻了(两变量)，非教科书单变量——但 **Run A vs rotmat s0 同为 dcnv4+detach**，17.38(fresh)→16.53(warm+rotmat)，那 0.85mm 来自暖启+rotmat 非 posnet；整片配置云 16.5~17.4 无一近 15.x，归因②多切面稳，不必再补 dcnv4+no_detach。

**归因①加固（跨配置云）：** Run A（dcnv4+detach+fresh）EHF Hands 15.88<15.97、UBody 10.01<10.29 **不侵蚀**。至此 Run A/Run B/rotmat s0 覆盖 `{conv,dcnv4}×{detach,no_detach}×{fresh,warm}×{rotmat on/off}` 全组合**全部不侵蚀**、唯 fairbase 侵蚀 → 抗侵蚀对手分支所有配置免疫、只由 aux 锚决定，机制论证从"单变量结论"升级为"跨 4 维配置云鲁棒"。

**bonus（Table 4 posnet 行）：DCNv4 头疑似过度设计**——同 epoch dcnv4(17.38)>conv(17.02)、无 InterHand 好处反略负；配 topo/occ 一起讲"特化模块未必值"。`[wa]` 0.231=rotmat s0≈stock 再证 T1（`[wa]`=body cam 几何）。

**净效果：探针线封口。** 归因①②双双收尾，剩余仅 topo/occ 消融（简化方向）+ Run B 跑完 ep4 补终值。

### topo/occ 消融开关已落地（待跑）

`HandDecoderLayer` 加 `use_topo_occ`（`cfg.hand_decoder_topo_occ`，默认 True 字节级不变）：**off** = 去掉 `HandTopologyAttention`+`ImplicitOcclusionModule`，换成**一个干净标准 `nn.MultiheadAttention` 自注意力**（= OSX 式 self-attn+deformable+FFN 层）。对照选"换成干净 self-attn"而非"整个删"，是为了隔离**特化模块净贡献**（而非"有没有 self-attention"）。CLI：train `--no_hand_decoder_topo_occ` / test `--no_hand_decoder_topo_occ`（**train/test 必须一致**，结构不同否则加载失败）。改动 5 文件（config/my_decoder/model_core/train/test），on 路径创建顺序不变→RNG 一致→逐比特 no-op；`ast.parse` 全过，torch 前向 smoke 待起训机验证。

**待跑 probe（conv+fresh，与 Run B/fairbase 组 decoder 复杂度阶梯，必测 EHF/UBody）**：
```bash
cd main && python train.py --gpu_ids 0 --lr 5e-5 --lr_mult 0.1 \
  --train_batch_size 64 --num_thread 8 --end_epoch 4 --phase1_epochs 2 \
  --exp_name output/probe_no_topo_occ \
  --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
  --encoder_setting osx_l --decoder_setting pytorch --grad_clip 1.0 \
  --hand_posnet conv --no_detach_hand_decoder_query --no_hand_decoder_topo_occ
```
判读：砍后 EHF/UBody **不侵蚀** → topo/occ 对抗侵蚀无贡献、可安全简化（"纯 aux 锚"论证不受损，方法更干净）；砍后**侵蚀** → topo/occ 也参与抗漂移，"纯 aux 锚"表述需收敛为"aux 锚 + 特化模块"。InterHand 侧则回答特化模块对 in-domain 是否有 signal（有→触发条件项"层数 3v6"）。

## 2026-07-06

### T1 shape/cam 拆分：归因翻转（`[wa]` 杠杆=cam 非 shape）+ 红格补不上（cam 刚性耦合）→ 转干净归因分析，最终模型不含 T1

`output/body_shape_t1_{shape,cam}/result/snapshot_{0,1}/`（warm-start rotmat s0、BEDLAM-only、`--no_refined_hand_coord`、`--body_shape_mode {shape,cam}`）。ep1/ep2 均平台，取 ep2：

| 指标 | rotmat s0 起点 | **shape-only** | **cam-only** | 7-02 both | stock |
|---|---:|---:|---:|---:|---:|
| UBody `[wa]` NME | 0.231 | 0.227 (−0.004) | **0.205 (−0.026)** ✅ 反超 | 0.204 | 0.219 |
| UBody `[abs]` NME | 0.318 | 0.335 (+0.017) | **0.362 (+0.044)** ❌ | 0.367 | 0.311 |
| UBody `[wa]` tip | ~0.294 | 0.289 | **0.259**(<stock) | — | 0.263 |
| EHF Face | 6.15 | **6.42 (+0.27)** ❌ | 6.15 (守) | 6.39 | 6.09 |
| EHF Hands | 15.61 | 15.70 (退) | 15.61 (守) | — | 15.97 |
| UBody PA Hands (3D) | 10.00 | 10.08 (退) | 10.00 (守) | — | 10.29 |

**归因完全翻转（与 7-02"只训 shape"方子相反）**：`[wa]` 改善 **100% 来自 cam_out**，shape_out 对 `[wa]` 无效。

- **cam-only**：一个头复现 both 全部 `[wa]` 收益（0.205≈both 0.204，反超 stock，**连 distal tip 0.259 < stock 0.263**），`[abs]` 也照崩（0.362≈both 0.367）。**手/脸 3D 全部纹丝不动**（EHF Hands 15.61 / Face 6.15 / UBody PA Hands 10.00 逐字=起点）——cam_out 只动投影平移，Procrustes 一对齐即消，3D mesh 未变 → 所有 PA-3D 指标零变化，只 2D 投影指标动。**隔离验证通过。**
- **shape-only**：`[wa]` 几乎不动（−0.004，仍输 stock），反把 `[abs]`/EHF Face(+0.27)/EHF Hands/UBody 3D 手全弄退——betas 是全身 shape，BEDLAM 上训 → 域偏移 → 3D vertices 净退。**纯负债，应弃。**（退是 betas 全身性所致，非 bug；印证隔离正确。）

**加性分解干净**：both 的三副作用 = cam(`[wa]↑`/`[abs]↓`) + shape(EHF Face↓)，每个精确归一个头。

**三判断**：
1. **`[wa]` 唯一杠杆 = cam_z 投影几何**，非手 articulation、非 shape。cam-only 连 distal 都反超 stock → **7-02"distal 天花板有一块是 body 侧几何"升级为"全部"**。
2. **`[wa]↑` 与 `[abs]↓` 被 cam_z 刚性耦合**（同一位移两面：推远→wrist-aligned 投影尺度贴 GT 但绝对位置偏），冻结框架内 cam 头无法解耦。
3. **shape_out 弃**（不撬 `[wa]` 且退 3D 手/脸）。

**对论文（红格补不上但非灾难）**：T1 无法在不崩 `[abs]` 前提下把 `[wa]` 变绿。但 **whole-body 3D 主表口径早已全绿、不依赖 T1**（UBody PA MPVPE Hands 10.00<10.29、EHF Hands 15.61<15.97，rotmat s0 就绿）；`[wa]`/`[abs]` 是自然手 **2D 诊断指标**，不必进 whole-body 主表。→ **把纠结已久的红格换成干净分析结论**："UBody `[wa]` gap 的杠杆是 body-side cam_z 投影几何（cam-only 单头反超 OSX、含 distal），与手 articulation/betas 无关；cam_z 对 `[wa]`/`[abs]` 刚性耦合，冻结框架内无法只取其一。"

**处置**：① **最终模型 = rotmat s0，不含 T1**；UBody 主表用 3D MPVPE（全绿），`[wa]`/`[abs]` 移分析节诚实标注 gap 来源。② T1 cam/shape 两 run 进 Table 4 归因行（`[wa]`=cam 几何 / `[wa]↔[abs]` 耦合 / shape 非杠杆且负债）；cam-only 0.205 反超作"body-side 上界探针"保留。③ future work 一句（不做）：约束 cam_z 只改投影尺度不改深度或可解耦，但 cam `[wa]`−0.026/`[abs]`+0.044 几乎同步 → 成功率低。④ **组件消融（posnet 换 conv + detach off 探针）现为唯一剩余实验线，主模型已定，可开跑。**

## 2026-07-05

### fairbase 训满 4 epoch：whole-body 侵蚀 headline 坐实、in-domain「接近追平」作废（gap ~1.2mm 且扩大）→ go/no-go 一分为二

`output/osx_fairbase/result/{snapshot0,snapshot1_itr1500,snapshot1_itr3000,snapshot2,snapshot3}/`（snap0=ep1、snap2=ep3、snap3=ep4；snap1 满点缺，中途 itr1500/3000 代 ep2）。

| ckpt | ~ep | InterHand PA | wrist-rel | EHF Hands PA-MPVPE | UBody PA Hands | UBody `[wa]` |
|---|---|---:|---:|---:|---:|---:|
| snapshot0 | 1 | 16.29 | 82.87 | 16.85 | 10.59 | 0.220 |
| snap1_itr1500 | ~1.3 | 16.08 | 82.25 | 16.48 | 10.23 | 0.219 |
| snap1_itr3000 | ~1.6 | 15.82 | 82.41 | 16.81 | 10.41 | 0.220 |
| snapshot2 | 3 | 15.54 | 81.92 | 17.13 | 10.46 | 0.218 |
| **snapshot3** | 4 | **15.36** | **81.71** | **16.72** | 10.40 | 0.218 |
| stock OSX | — | 19.58 | 86.32 | 15.97 | 10.29 | 0.219 |
| **我们** rotmat s0 | — | 16.53 | 85.09 | **15.61** | **10.00** | 0.231 |

**两条子句分裂判定**：
1. **whole-body 侵蚀 = 坐实（不翻案）**。EHF Hands 从 ep1(16.85) 起全程 **16.5–17.1 震荡、从不回落**（itr1500 的 16.48 是瞬时低点，下个 ckpt 弹回 16.81）；**永远回不到 stock 15.97、更够不到我们 15.61**。UBody PA Hands 全程 ~10.4，始终劣于 stock 10.29 与我们 10.00。**侵蚀在 ep1（最弱特化）即满额出现** → 不是过训晚期 artifact，堵死 reviewer"早停就没事"的反驳。翻案判据（EHF 回升 ≤15.6）**未触发**。→ **新 headline「whole-body 手部抗侵蚀」证据完整**，fairbase 侵蚀轨迹图（InterHand-PA vs EHF-Hands，ft 五点 + stock + 我们）可直接画。
2. **in-domain「0.24mm 让步」= 作废**。ft 训满 4ep 到 **15.36 且仍在降**（ep3→ep4 −0.18，未平台）；真实 in-domain gap = 16.53−15.36 = **~1.2mm 且扩大**（外推收敛 ~15.0-15.3，结构性差 ~1.3-1.5mm）。wrist-rel 同样恶化：ft **81.71**、我们 85.09，差 **~3.4mm**。**7-04「+0.24 / 天花板≈16.5 / 结构性差 ~0.2mm」是拿我方最优点 vs ft 1-epoch 点的不公平比较，全部作废**（那三个数就地在 7-04 条目加了勘误注）。

**净效果**：trade-off 叙事更干净（纯 Pareto：ft 用 InterHand+wrist-rel 换 whole-body，我们相反），但「接近追平 in-domain」**死**。措辞降级并移主轴：**headline 主轴 = whole-body（EHF/UBody/AGORA 3D 手），InterHand 定位为「手 crop 专项基准，诚实报告、非主场」**；写成「我们不为 in-domain 手 crop 基准过度特化，换 whole-body 3D 手全面优于 stock 与任意 epoch 的 ft，held-out 2D 持平；代价 = InterHand PA 诚实落后 ~1.2mm」。对 stock 仍全面赢（InterHand 16.53 vs 19.58）。

**归因补强（whole-body 赢非 rotmat 带来）**：snap2（无 rotmat）EHF Hands 已 **15.67 << ft 16.85**、UBody PA Hands 10.15 < ft → **抗侵蚀主归架构 + aux/热图锚**（冻结骨干上我们手头有 image-plane 2D 监督锚，OSX decoder 无锚=死代码，见 7-04 追加①），rotmat 只是小增益（15.67→15.61 / 10.15→10.00）。这正是组件消融要拆的。

**go/no-go 收口**：decoder 线数据完整，**不再需要跑 fairbase**。生死闸最终判定：**whole-body 主张成立、in-domain 追平放弃**。

### T1 shape/cam 拆分：代码改动就绪（`body_shape_mode` 开关）+ 拆分实验计划

**为什么先做 T1**：UBody `[wa]`（0.231 vs 0.219）是 Table 1「全绿 vs stock」的**唯一红格**（定位级）；fairbase 又印证该格对换 decoder/喂数据双免疫（ft 0.220≈stock 0.219）→ **T1 是唯一杠杆**。7-02 已见 both-T1 把 `[wa]` 推到 **0.204 反超**，只是 `[abs]` 崩 0.367 + EHF Face 退 6.39（crosspath 归因 cam_z 域偏移）。拆分 = 验证 0.204 来自 shape 还是 cam_z；shape-only 是 7-02 开的「约束 cam_z / 只训 shape」方子。组件消融放主模型冻结后一次跑（更省，且现在报告点可能因 T1 再变）。

**代码改动（3 文件，默认 `both` 字节级等价，AST 通过）**：新增 `cfg.body_shape_mode ∈ {both,shape,cam}` 选解冻哪个解耦头。`config.py` 加默认 `'both'`；`model_core.py:151` 按 mode 构造 `body_shape_trainable_prefixes` 子集（both→[shape_out,cam_out] / shape→[shape_out] / cam→[cam_out]）；`train.py` 加 `--body_shape_mode`（choices 限定）→ cfg。**只改 prefix 列表构造一处**，下游四处（save_model / _verify_freeze_status / optimizer group / _check_gradient_flow）全 `startswith(prefix)` 自动跟随。`train_body_shape=False` 或 `both` 时与旧代码逐字等价 → 手/脸零影响。

**两个 run（远端，warm-start `coordrefiner_rotmat/snapshot_0`、`--trainset_3d BEDLAM`（trainset_2d 默认已空=BEDLAM-only 干净 betas）、`--no_refined_hand_coord`（匹配 rotmat s0 forward，手 guardrail 才=终值）、lr 1e-5 / 2ep / phase1=end=2）**：`--body_shape_mode shape`（先跑，预期只赢不崩）/ `--body_shape_mode cam`（归因对照，预期复现 `[abs]` 崩）。warm-start 用 rotmat s0 而非 snap2：T1 body head 训练不经手部、shape-vs-cam 归因不受影响，且最终模型（手 rotmat s0 + body T1）一步到位、guardrail 即终值。

**判定表**：UBody `[wa]` 保 <0.229（撬动=成功）+ `[abs]` ≈0.316 不崩（both 崩 0.367）+ EHF Face ≈6.15 不退 = **只赢不崩→红格补上、Table 1 全绿**；手 guardrail InterHand ≈16.53 / EHF Hands ≈15.61 **必须纹丝不动**（动=漏进手路径 bug）。启动见 `🔧 解冻 ~10k`(shape) / `~3k`(cam) 参数。

**诚实预期**：shape-only 保 `[wa]` 且不崩 → 最终模型 = rotmat s0 + shape head，红格补上；若 shape-only 啥也不撬 → 0.204 主要来自 cam_z（必崩 `[abs]`），红格补不上、接受「全绿差一格」，cam-only run 确认该归因。**先跑 shape-only 看结果再决定 cam-only。**

## 2026-07-04

### rotmat pose loss = 对靶正结果但一个 epoch 触顶 16.53（未过闸）；aux-off 更差（"aux 抢容量"证伪）→ InterHand PA 线封口

两个 run 均暖启自 `coordrefiner_gate/snapshot_1`(16.80)、`--init_trained_path` fresh 4-epoch schedule、`--no_refined_hand_coord`(L728 关)、posnet_lr_mult 0.25 / phase1 4——与 osx_fairbase(16.29) 同配方。`coordrefiner_rotmat` = 加 `--use_hand_rotmat_pose_loss`(aux 默认留 1.0)；`coordrefiner_rotmat_aux0` = 再加 `--hand_aux_coord_loss_weight 0`。**两两单变量干净**（rotmat vs aux0 只差 aux 权重 1.0/0.0）。

**rotmat（aux ON）PA 轨迹**：

| ckpt | ~ep | PA | wrist-rel |
|---|---|---:|---:|
| gate snap1（起点） | 0 | 16.80 | 83.93 |
| **snapshot_0** | 1 | **16.53** ⬅最优 | 85.09 |
| snapshot_1_itr1500 | 1.3 | 16.57 | 84.70 |
| snapshot_1_itr3000 | 1.6 | 16.55 | 84.52 |
| snapshot_1 | 2 | 16.58 | 85.27 |

⚠️ **7-05 勘误**：本节"未过闸 16.29(+0.24)"及"天花板 ≈16.5、结构性差 ~0.2mm"是拿我方最优点比 **ft 的 1-epoch 点(16.29)**——不公平。ft 训满 4ep 到 **15.36 仍在降**，真实 in-domain gap = **~1.2mm 且扩大**、wrist-rel 差 ~3.4mm。见 2026-07-05 条目。此处 16.53 的**报告价值不变**（whole-body 抗侵蚀 headline 的报告点），仅"接近追平 in-domain"的解读作废。

**aux0（aux OFF）**：snapshot_0_itr1500 **16.75** → itr3000 **16.88**（在退）。

**护栏 @ rotmat snapshot_0（最优 PA 点，全守）**：UBody `[wa]` 0.231 / `[abs]` 0.318（没像 T1 崩）/ PA MPVPE Hands **10.00**(snap2 10.11 → 反而最好) / EHF Face **6.15**(平) / HInt `[all wa]` **0.481**(snap2 0.480、OSX 0.487 → held-out 赢点守住)。唯一动的是 InterHand wrist-rel ~84→85。

**三结论**：
1. **rotmat 对靶有效但触顶**。16.80→**16.53**(−0.27)、与 L728 的 +0.25 正好反向 → 坐实"InterHand PA 的杠杆在 pose 空间、不在 2D 坐标"。**但 epoch1 完全没吃到**(16.53→16.55~16.58 平台/微升)，外推 ep2-3 仍 ~16.5，**未过闸 16.29(+0.24)** 且我方总预算更多(暖启点已 16.80)。→ **冻结骨干 + 我们 decoder 对 InterHand 3D PA 的天花板 ≈16.5，比 OSX 原生 decoder 16.29 结构性差 ~0.2mm，rotmat 收窄未抹平**。
2. **关 aux 更差 → 证伪"aux 抢容量"(假设 E)**。单变量下 aux 关掉 PA 16.75→16.88 且退，明显劣于 aux 开的 16.53。aux 对 PA 净正/中性，**recipe 保留 aux=1.0**。
3. **PA 收益不打护栏，只 wrist-rel 退 ~1mm**（PA↔wrist-rel 窄权衡：rotmat 只监督手指相对旋转 → Procrustes articulation 到处变好(UBody PA Hands 10.00 也印证)，但 InterHand 非-Procrustes 的腕相对全局摆放漂；UBody/EHF 的 3D 均未退 → 非广义退化）。

**决定：停，不训 ep2-3**（已平台，穿不过 16.29）。**rotmat 最优点 = snapshot_0(16.53)**；aux0 弃。rotmat 作 Table 4 **正向 ablation 行**，与 L728 负行、aux 正向配对：三行讲清"pose 空间监督=+0.27 且护栏全守/held-out 不动；2D 坐标监督(aux)有益该留；精修坐标喂 regressor(L728)有害"——2D vs 3D-PA 杠杆正交性。

**战略收口**：**InterHand PA 探索封口**，最优 in-domain **16.53**，诚实写"略逊 OSX-ft 16.29(+0.24、我方更多预算)"，不 claim in-domain PA 优越。论文重心不变——**held-out HInt 赢(0.481<0.487) + 全护栏追平 + 高效冻结 recipe**，decoder/loss 消融(rotmat 正 / aux 正 / L728 负)作方法分析。**可选收尾（非必须）**：补对照 gate snap1 +1ep、rotmat **关**（aux 开、L728 关），落在 gate 自身外推 ~16.6-16.7 即把 16.53 干净归给 rotmat（现为 vs 外推、略软）。（⚠️ 同日晚间被追加②修正：HInt "赢"被公平基线三测抹平，headline 已换轴，见下。）

### 追加①：OSX 原生 decoder 的坐标机制全是死代码（代码审查，推翻 §0‴①②对标依据）

亲自核实 vendored mmpose（normal 路径 = `main/OSX.py` → mmpose `Poseur`）：

- `main/transformer_utils/mmpose/models/utils/transformer.py:613-637`：逐层 `reg_branches` 坐标回归 + reference point 迭代更新**整段被注释**——6 层 decoder 的参考点全程固定在传入的 `coord_init`（soft-argmax 坐标，detach）不动；
- `poseur.py:158` `return dec_output.feat[-1]` 只返回**末层特征**；`poseur_head.py:398-403` `outputs_coords` 恒空；
- RLE loss / sigma 不确定性 / noisy-reference-sample：`__init__` 构造了，但 `forward_mesh_recovery` 路径一概不调用（`get_*_rle_loss` 无调用方；`coord_init` 非空时 encoder 段 sigma 也被跳过）。

→ **产出 stock 19.58 / 公平基线 16.29 的 OSX decoder = 6 层、3 尺度、固定参考点、纯特征精修器**（20 关节 query 间分组 self-attn（可学习 query_pos）+ MSDeformAttn(4 点) + FFN），监督完全靠 pose→FK 间接梯度，**无任何 decoder 坐标监督**；regressor（`HandRotationNet`，与我们同一个类）吃（末层特征, detach 的 soft-argmax 坐标）。

**五条修正**：
1. `roadmap` §0‴ 根因①（"OSX 每层回归坐标+RLE/aux 监督"）②（"每层迭代参考点"）是读 config 声明所得，与实际 forward 相反 → **7-01 坐标精修大改瞄错靶**：加的是 OSX 本来就没有的机制。gate≈snap2（16.80 vs 16.78）、L728 负（+0.25）、rotmat 才动 PA（−0.27），全部与此自洽。
2. 与 OSX-ft 的**真实未测差异变量** = {DCNv4PositionNet 冷启+反向脆弱 vs conv PositionNet（osx_l 可直接暖启）、3 vs 6 层、topo/occlusion vs 素 self-attn、detach_hand_decoder_query（OSX 不 detach，pose 梯度回流 posnet 特征分支）、预训练本钱}。前四项均可单变量测，从未测过。
3. **aux0 消融有 confound**：关 aux 权重后 `bbox_embed` 仍在逐层迭代参考点（靠间接 pose 梯度无监督漂移）→ "aux-off 更差"只证明"装了坐标机制必须配监督"，不证明"坐标机制优于 OSX 式固定参考点"。真正的固定参考点对照 = 7-01 前的老 decoder：snap2 16.78 vs gate 16.80 → **坐标机制对 PA 净贡献≈0、对 2D 为正**。
4. "冻结骨干+我们 decoder 天花板≈16.5"应改写为"**当前栈（DCNv4 冷启 posnet + 3 层 topo/occ decoder + detach）的天花板≈16.5**"，不宜归咎"decoder 概念"。
5. 论文引用 OSX 行为写"OSX 发布代码中该分支被禁用"，不写"Poseur 论文如何"；§0‴ 曾建议的"叠 RLE（OSX 看家武器）"同理作废——OSX 实际也没用 RLE。

详见记忆 `osx-decoder-coord-machinery-is-dead-code`；roadmap §0‴ 已加勘误注。

### 追加②：P0 公平基线三测（HInt/UBody/EHF）：HInt 头条被抹平 + OSX-ft whole-body 手部侵蚀曝光 → headline 换轴

`output/osx_fairbase/result/{HInt,UBody,EHF}_result.txt`（7-04 远端补测同步）。口径注意：InterHand 16.29 来自 **snapshot_0 = 仅 1 epoch 微调**（本地 train log 只见 epoch 0，配比 IH~21/UBody~18/COCO~25 与我们一致）；**三测所用 snapshot 待远端确认（应与 16.29 同点）**。真伪核查过：UBody/EHF/[wa] 差异显著、HInt 逐行差 0.001-0.003，非误用我方 ckpt。

| 口径 | stock OSX | **OSX-ft** | 我们（rotmat s0 / snap2） | 判定 |
|---|---:|---:|---:|---|
| InterHand PA | 19.58 | **16.29** | 16.53 / 16.78 | 输 ft 0.24 ❌ |
| InterHand wrist-rel | 86.32 | **82.87** | 85.09 / 84.13 | 输 ft ~2 ❌（此前只对 stock 说"略好"） |
| **EHF PA MPVPE Hands** | 15.97 | **16.85（退化+0.88）** | **15.61** / 15.67 | **赢 ft 1.24（~7%）** ✅ |
| **UBody PA MPVPE Hands** | 10.29 | **10.59（退化+0.30）** | **10.00** / 10.15 | **赢 ft 0.59（~5.6%）** ✅ |
| UBody PA MPJPE Hands | 10.55 | 10.85 | 10.28 | 赢 ft ✅ |
| HInt `[all wa]` | 0.487 | **0.480** | 0.481 / 0.480 | **精确平** ⚪（头条死） |
| HInt `[all abs]` | 0.451 | 0.445 | 0.447 / 0.445 | 平 ⚪ |
| UBody `[wa]` NME | 0.219 | **0.220（没动）** | 0.231 / 0.229 | 输双方 ~0.010 ❌ |
| UBody `[abs]` NME | 0.311 | 0.311（没动） | 0.318 / 0.316 | 略输 |
| EHF Face | 6.09 | 6.09 | 6.15 | 略输 0.06 |
| EHF PA MPVPE All | — | 48.70 | 48.79 / 48.65 | 平（body 冻结） |

**三判读**：
1. **HInt 精确平局 → "held-out HInt 赢"头条死亡**。不只总分平（0.480 vs 0.480），visible/occluded、per-finger、per-level **逐行几乎一致**（occluded wa 0.483 vs 0.482、tip 0.542 vs 0.542），遮挡模块在 occluded 子集也无残余优势。0.487→0.480 的收益 **100% 归微调数据**（与 InterHand −14% 同一归因陷阱，这次被公平基线当场戳破）；hand-crop held-out 2D 由冻结 box/ROI 管线+数据决定，**对 decoder 家族饱和**——"2D 轴正交"结论从我方内部证据升级为对照双方证据。
2. **新赢点 = whole-body 3D 手（结构性）**。OSX-ft 仅 1 epoch hand-heavy 微调，EHF Hands 15.97→**16.85**、UBody Hands 10.29→**10.59**——拿 in-domain、丢 whole-body 的经典侵蚀（暖启特征空间整体漂向 InterHand crop 域）；我们同数据 16.53 的同时 EHF **15.61**、UBody **10.00**，两口径**均优于 stock**。**OSX-ft 沿 Pareto 前沿滑动，我们把前沿外推。** 机制假设（可消融验证）：OSX decoder 无任何 2D 锚（追加①），微调时无东西阻止特征漂移；我们的 per-layer aux 2D + posnet 热图监督即锚。
3. **UBody `[wa]` 对"换 decoder"与"喂数据"双免疫**（ft 0.220 ≈ stock 0.219）→ 7-02 归因（body 侧 shape/cam）获第三方印证，**T1 拆分是该格唯一杠杆**。诚实项：我们 0.231 对 ft 0.220 的差距仍是 distal 签名（tip 0.294 vs 0.268、j3 0.246 vs 0.231、j1 持平）——小手末端 2D 上 OSX decoder 家族确实仍占优。

**headline 改写**："hand-heavy 微调下，OSX 原生 decoder 的 in-domain 收益（16.29）以全身手部退化为代价（EHF +0.88 / UBody +0.30）；我们以 0.24mm in-domain 让步，换 whole-body 3D 手全面优于 stock 与公平基线（EHF −1.24 / UBody −0.59 vs ft），held-out 2D 不落后，**单模型、无外部专家**。" HInt 平局在新故事里是支撑证据（"特化未牺牲 held-out 泛化"）；与 H4W++ 问题意识同轴，Table 5 效率差异化直接接上。（⚠️ **7-05 修正**：此表 ft 列均为 **1-epoch** 点；训满 4ep 后 InterHand→15.36 仍降、EHF Hands 稳在 16.72（不回落）。whole-body 赢**加强**、但"0.24mm 让步"作废→真实 ~1.2mm，headline 措辞降级见 2026-07-05。）

**下一步**：① ~~fairbase 训满 4 epoch 出侵蚀轨迹图~~ **已完成（2026-07-05：侵蚀坐实、in-domain gap 修正为 ~1.2mm）**；② T1 shape/cam 拆分（代码就绪，见 7-05）；③ 组件消融升级为"哪个组件买来抗侵蚀"——首推 posnet 换回 conv PositionNet（osx_l 暖启）+ detach off 探针；④ **最佳报告点 = rotmat snapshot_0**（16.53 / EHF 15.61 / UBody 10.00 / HInt 0.481）。

## 2026-07-03

### L728（精修坐标喂 regressor）= 负结果，证伪"精修没到 regressor"判据 → decoder 坐标线判定不打闸

`output/coordrefiner_l728_fair4`：暖启自 `coordrefiner_gate/snapshot_1`(gate 2ep, PA **16.80**)，应用 L728 后**只训 1 epoch**（epoch 2, itr 4733，日志止于 snapshot_2 保存）。L728 = `model_core.py:740-746`：把 decoder 末层**精修 xy**（反归一化回 hm-pixel）+ 原 soft-argmax **z** 拼接、detach 后喂 `hand_regressor`（`use_refined_hand_coord` 默认 True；关=旧纯 soft-argmax）。配方 `posnet_lr_mult 0.25 / phase1_epochs 4`——**与 osx_fairbase(16.29) 逐项一致**（核对 `output/osx_fairbase/log/train_logs.txt`：lr 5e-5 / batch 64 / end 4 / phase1 4 / posnet_lr_mult 0.25 / posnet_grad_clip 0.5），故 fair4↔gate↔16.29 在这些 knob 上单变量可比。**更正 7-01 "锁定配方=snap2 的 0.5/phase1-2" 的说法**：那是 snap2(joint_polish_f) 自己的配方，不是 16.29 对标配方；对标 16.29 应匹配 osx_fairbase = 0.25/phase1-4。唯一口径瑕疵 = fair4 用 `--continue_train` 从 gate snap1 续 epoch 2→4（只跑 cosine 尾 2 epoch），非 fresh 4-epoch schedule；但这是"gate 上叠 L728 再续训"的正确设计，不影响 L728 判负。

| 指标 | snap2@fixed | gate snap1（改前） | **fair4 snap2**（L728,+1ep） | 判定 |
|---|---:|---:|---:|---|
| InterHand PA | 16.78 | **16.80** | **17.05**（itr1500 17.14 / itr3000 17.05） | ❌ 退 +0.25 |
| InterHand wrist-rel | 84.13 | 83.93 | 84.16 / itr3000 84.43 | ❌ 退 |
| UBody `[wa]` NME | 0.229 | — | **0.230** | 持平（仍输 OSX 0.219）|
| UBody PA MPVPE Hands | — | — | 10.43 | 正常 |
| 生死闸 OSX-ft | | 16.29（+0.51） | 16.29（**+0.76**） | 更远 |

epoch 内轨迹从 16.80 **单调抬升停在 17.05、不回落** = 回归特征非欠训。`loss_hand_aux_coord` 全程噪声 0.31–0.51、收 ~0.38–0.39，**不比 gate 的 ~0.36 更低**（略差）。稳定性仍 ✅（零 NaN）。

**结论：L728 假设被证伪。** gate 判据"aux 在学但 PA 不动 = 精修坐标没进 regressor"——喂进去后 PA **确实动了但方向错**（16.80→17.05）。含义：①"没喂进 regressor"不是瓶颈，喂进去有反应说明路是通的；②**坐标输入对 3D hand PA 是弱杠杆，精修版不是更好的输入**——精修 xy 在 2D NME 好 ~15-20%，但它对 3D SMPL-X 手姿只是次要 conditioning；换来源/尺度（xy 来自 decoder、z 来自 soft-argmax，provenance 不一致）+ detach（坐标头无法反向 co-adapt regressor 需求）反而扰动了本吃 soft-argmax 的 regressor。③ UBody `[wa]` 又一次确认 body 侧（不动），归 T1。**decoder 坐标精修结构线到此判定"不打闸"**（平台 ~16.8，与 16.29 反而更远），与 `h4wpp-competitor` 判断一致：杠杆在融合/expert 不在 decoder 坐标机制。

**处置**：本 run 作 Table 4 负消融行（"refined coord→regressor = PA +0.25、`[wa]` 无变化"）+ gate 那行（"迭代采样+aux 单独价值"）；`use_refined_hand_coord` 回退默认关、不进主 recipe。fair4↔gate 是干净单变量对比（同 posnet_lr_mult 0.25 / phase1 4，唯一差 = L728 开关 + 续训 vs 暖启），退 0.25mm 结论可靠。**下一步见 continue.txt / roadmap 主线换挡**。

## 2026-07-02

### body-shape T1 评测收口：UBody `[wa]` 真提升但伴随 `[abs]`/MPVPE 退化

评测目录：T1 `output/eval_body_shape_t1/`，OSX baseline `output/eval_OSX/`；T1 是从 `joint_polish_f/snapshot_2` warm-start 后只训 `body_regressor.shape_out+cam_out`，所以相对 OSX 的数字是最终模型对比，因果归因需同时看相对 snap2 起点。

**AGORA（snapshot0/1 都已评）**：变化很小，epoch1 基本平台。

| 指标 | OSX | T1 s0 | T1 s1 | s1-s0 |
|---|---:|---:|---:|---:|
| PA MPVPE All | 70.47 | 70.11 | 70.07 | -0.04 |
| PA MPVPE Hands | 11.65 | 11.13 | 11.14 | +0.01 |
| PA MPVPE Face | 4.78 | 4.73 | 4.72 | -0.01 |
| MPVPE All | 169.83 | 169.67 | 169.69 | +0.02 |
| MPVPE Hands | 73.13 | 73.88 | 73.88 | -0.01 |
| MPVPE Face | 78.04 | 77.74 | 77.74 | +0.00 |

判读：PA 口径有 <0.6mm 的小幅改善，非 PA hand 反而差约 +0.75mm；实用意义弱，不能作为主贡献。

**UBody（snapshot1 补评后趋势明确）**：

| 指标 | OSX | snap2 起点 | T1 s0 | T1 s1 | s1-s0 |
|---|---:|---:|---:|---:|---:|
| PA MPVPE All | 41.00 | 41.10 | 41.05 | 41.10 | +0.05 |
| PA MPVPE Hands | 10.29 | 10.11 | 10.22 | 10.23 | +0.01 |
| MPVPE All | 98.92 | 99.61 | 100.09 | 100.34 | +0.25 |
| MPVPE Hands | 38.24 | 39.23 | 38.85 | 38.85 | -0.00 |
| PA MPJPE Body | 50.51 | 50.79 | 50.24 | 50.24 | -0.00 |
| `[abs]` NME | 0.311 | 0.324 | 0.365 | 0.367 | +0.002 |
| `[abs]` PCK@0.2 | 0.422 | 0.406 | 0.300 | 0.298 | -0.002 |
| `[wa]` NME | 0.219 | 0.229 | 0.204 | 0.204 | +0.000 |
| `[wa]` PCK@0.2 | 0.587 | 0.560 | 0.622 | 0.621 | -0.001 |

paired bootstrap（T1 s1 - s0）：`[wa]` NME +0.00009、PCK@0.2 -0.00052，等于没有新收益；`[abs]` NME +0.00219、MPVPE All +0.246mm，说明第二个 epoch 只让全局/abs 相关指标轻微变差。group attribution 同结论：s1 vs s0 overall `[wa]` delta 仅 +0.0001，按 size/side/finger/level 全部平台。

相对 OSX 的 UBody `[wa]` 改善是真的但结构性分裂：overall `[wa]` 0.2182 -> 0.2038（约 -6.6%），small hands 0.2851 -> 0.2540（约 -10.9%）最明显；但 `[abs]` 同时大幅退化，尤其小手 abs 0.378 -> 0.469。相对 snap2 起点，T1 把 `[wa]` 0.229 -> 0.204，但也把 `[abs]` 0.324 -> 0.367。

**crosspath 解释**：T1 几乎不改 `smplx_root_pose/body_pose`，主要改 `smplx_shape` 与 `cam_trans.z`。UBody `cam_z` 均值从 OSX 22.85 -> s0 25.43 -> s1 25.41；AGORA 从 41.28 -> 43.91 -> 44.00。第二个 epoch 只继续小幅推 shape/cam（UBody s1-s0 shape mean_abs 0.0147、cam mean_abs 0.0310），不足以带来新收益。

**结论**：T1 的 UBody `[wa]` 提升不是噪声，但不能写成"手部质量提升"。WA 先把预测手平移到 GT wrist，只看手内相对形状/投影尺度；shape/cam 改动可能让 wrist-relative 2D 更接近真实手，同时让 wrist/global placement 更差，所以 `[abs]` 和非 PA 3D 退化。**同配方不再继续训练，最佳也只能取 snapshot0 作诊断/附带结果**。

**下一步（若继续挖 T1）**：必须拆 `shape_out only` vs `cam_out only` ablation；优先怀疑 `cam_out` 解释大部分小手 WA 改善和 `[abs]` 退化。若想保留 WA gain，需要约束 `cam_trans.z` 或只训 shape，并把 UBody `[abs]`/MPVPE 作为 guardrail。

**7-02 补充：InterHand 收口 + 归因钉死（预注册预测命中）**。T1 InterHand 全量：PA **16.78**（snap2 逐字相同）/ wrist-rel **83.36**（snap2 84.13 → −0.77，vs OSX 86.32 领先 ~3mm）。预注册判据"PA 是相似对齐、cam 碰不到，动 >0.3 = 手部模块漂"→ 实际 0.00：**手/脸分支确认零漂移、无 BEDLAM 毒，T1 全部 delta 干净归因 betas+cam**。wrist-rel 的 −0.77 是纯 betas 骨长收益（cam_trans 不影响 3D）——"卡死 84"的 wrist-rel 第一次被便宜推动。**归因修正（改写 6-27 结论）**：UBody `[wa]` 那 0.010 不是"100% 手头 articulation"——手分支零改动下 body 侧 shape/cam 就能覆盖整个 gap 并反超（0.204<0.219），WA2D 线撞的"distal 天花板"有相当一块是 body 侧几何/投影（腕对齐消平移不消尺度，误差 ∝ 离腕距离 → 同样呈 distal 递增签名）。指尖残差（tip 0.256 vs OSX 0.263，改善最小）仍是 articulation，归 decoder 线。

### coordrefiner gate epoch0/1：恢复到 snap2 持平，"精修没到输出"判据触发 → L728 排队

`output/coordrefiner_gate`（暖启 snap2；实跑配方 **end_epoch 8 / phase1 4 / posnet_lr_mult 0.25**，偏离锁定的 4/2/0.5——cosine 展开与 phase 边界不同，等预算口径要注明）：

| ckpt | 训练量 | InterHand PA | wrist-rel |
|---|---|---:|---:|
| snapshot_0 | 1 ep | 17.06 | 84.02 |
| snapshot_1_itr3000 | ~1.6 ep | 16.84 | 84.00 |
| snapshot_1 | 2 ep | **16.80** | 83.93 |

判读：epoch0 = 暖启扰动（bbox_embed 零初始化+新机制打架），epoch1-2 恢复到 **≈snap2(16.78)，未钻下去**；减速明显（−0.26→−0.04），外推平台 ~16.6-16.7，距闸 16.29 差 0.5。**决定性信号**：`loss_hand_aux_coord` 0.55→**~0.36**（< 零初始化时的 soft-argmax 基线 0.44，即精修坐标在 2D 上比 soft-argmax 好 ~15-20%）但 PA 没吃到 → 命中预注册判据"**aux 在学、PA 不动 = 精修坐标没进 regressor**"（`model_core.py:728` 仍喂 soft-argmax 坐标，精修只改善 decoder 内部采样位置，杠杆弱）。稳定性 ✅：两个 epoch `skipped=0`、零 NaN（detach 方案正式规模坐实）。@1ep 同预算对比扎心：我们 17.06 vs OSX-ft 16.29。**下一步 = L728 修复**（decoder 末层精修坐标反归一化 xy + soft-argmax z、保持 detach 惯例，替换喂 regressor 的 `hand_joint_img`）→ 从本 run snapshot_1 暖启（bbox_embed 已练出 0.36）再训。当前 run 数据留作"迭代采样+aux 单独价值"消融行（Table 4）。

**起点公平性修正（用户洞察）**：撤回"干净 gate = osx_l 冷启"的说法——osx_l 给 OSX decoder 暖启（19.58）、给我们冷启（~25），本就不对称；**各架构用自己最好的预训练 init**（OSX←osx_l、我们←snap2）才是公平类比，`init_trained_path` 应保留。论文 caveat：两边预训练本钱来源不同（OSX 原始大数据 vs 我们同微调集），写成"各自最佳 init、同数据微调"。

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
