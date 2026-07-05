# 论文表格规划（"全绿 vs OSX" 非顶会版）

更新时间：2026-07-04（headline 换轴：whole-body 手部抗侵蚀；生死闸已判负、按兜底路线改写）

本文件把"能发的最小实验集 + 表格骨架"固化下来。
- 目标定位：**非顶会**。相对 stock OSX 全面提升（全绿）+ 一个能写成一句话的方法贡献。
- 结果记录见 `docs/experiment_log.md`；路线/决策见 `docs/post_stage3_roadmap.md`（§0⁵ = 7-04 换轴）；竞品分析见记忆 `h4wpp-competitor.md`。
- 一句话定位（7-04 改写）：**"冻结骨干上的 hand-heavy 微调，OSX 原生 decoder 拿 in-domain（16.29）要付出 whole-body 手部退化的代价（EHF +0.88/UBody +0.30）；我们的 decoder 以 0.24mm in-domain 让步，换 whole-body 3D 手全面优于 stock 与公平基线，held-out 2D 不落后——单模型、无外部专家。"** 仍**不写**"我们打败了 OSX 的 decoder"（in-domain 确实没打过）；也**不再写**"held-out HInt 赢"当 headline（已被公平基线抹平成精确平局，降级为支撑证据）。

---

## 0. 图例（每格的角色）

| 标记 | 含义 |
|---|---|
| 🟢 | **必须赢**——对 **stock OSX**，构成"全绿"对比表 |
| 🔑 | **必须赢**——对 **公平基线 OSX**（OSX decoder 同条件微调），= **方法闸**，过不了就没有方法贡献 |
| ⚪ | **同台即可**——对近期 SOTA（SMPLer-X / H4W++ / Multi-HMR），差异化靠效率 + 诚实 |
| 🔍 | **诚实分析**——平手/小赢都算成功（held-out 泛化、身体不退化、效率） |
| ⚠️ | **风险格**——训练域外（当前训 InterHand+UBody+MSCOCO），可能退化，重点盯 |

**已知种子数字**（7-04 全套，报告点 = `coordrefiner_rotmat/snapshot_0`）：InterHand PA：stock **19.58** · OSX-ft **16.29**（snapshot_0=1ep，wrist-rel **82.87**）· 我们 **16.53**（wrist-rel 85.09；snap2 16.78/84.13）。**EHF Hands PA-MPVPE：stock 15.97 · ft 16.85（侵蚀）· 我们 15.61** ✅。**UBody PA MPVPE Hands：stock 10.29 · ft 10.59（侵蚀）· 我们 10.00** ✅。HInt `[all wa]`：stock 0.487 · ft **0.480** · 我们 0.481（**精确平**）。UBody `[wa]`：stock 0.219 · ft 0.220 · 我们 0.231（❌ 归 T1；T1 已 0.204 但 `[abs]` 崩，待拆分）。EHF Face：6.09/6.09/6.15。

---

## Table 1 — 主表：全身（whole-body）
testset = **EHF + UBody**；列 = MPVPE / PA-MPVPE 的 (all / hands / face)

| 行（方法） | all | hands | face | body 部分 |
|---|---|---|---|---|
| OSX (stock, 2023) | 基准 | 基准 | 基准 | 基准 |
| SMPLer-X / Multi-HMR | ⚪ | ⚪ | ⚪ | — |
| H4W++ (CVPR26) | ⚪ | ⚪（它强，大概率不赢） | ⚪ | — |
| **Ours** | 🟢 | 🟢（EHF=⚠️） | 🟢 | 🔍 与 OSX≈相同 → "身体零退化" |

- **body 部分是隐藏王牌**：冻结骨干 ⇒ 身体逐比特≈OSX ⇒ 写"几乎相同"，当"可证明不破坏身体"的卖点，不是输赢。
- **EHF hands = ⚠️→✅（7-04 解除并升级为 headline 格）**：我们 **15.61** 优于 stock 15.97，且公平基线 ft 侵蚀到 **16.85**——EHF/UBody hands 两格现在是新 headline「whole-body 抗侵蚀」的正面战场，配 fairbase 侵蚀轨迹图（fairbase 训满 4ep 逐 snapshot 评，待跑）。
- **7-02 更新**：UBody hands `[wa]` 格有了 body 侧杠杆——T1 已做到 **0.204 反超 OSX 0.219**，但原样伴随 `[abs]` 0.367 崩 + EHF Face 6.39，**须等 shape/cam 拆分消融拿到"只赢不崩"版本才能上表**；`[abs]` 与 EHF Face 因此升级为**受监护格**（任何 T1 变体上表前必须核）。7-04 补：fairbase `[wa]` 0.220≈stock → 该格对换 decoder/喂数据双免疫，T1 是唯一杠杆。AGORA 现可测（`--testset AGORA`），可作 Table 1 附加列（T1 在 AGORA PA 全线小赢）。

## Table 2 — 手部专项表（核心战场 / 生死闸）
testset = **InterHand26M (+ HIC 若有)**；列 = MPJPE / MPVPE / MRRPE（per-hand root+scale 对齐，IntagHand 协议）

| 行（方法） | MPJPE | MPVPE | MRRPE |
|---|---|---|---|
| OSX (stock) = 19.58 | 🟢 | 🟢 | 🟢 |
| **OSX 公平基线 = 16.29↓** | 🔑 | 🔑 | 🔑 |
| H4W++ / WiLoR 系 | ⚪ | ⚪ | ⚪ |
| **Ours（新 decoder）** | 🔑 必须 < 16.29↓ | 🔑 | 🔑 |

> **7-04 判定：🔑 生死闸已判负（我们最优 16.53 vs 16.29，wrist-rel 85.09 vs 82.87），按预设兜底转"分析型"，不 claim in-domain 方法优越。** 但兜底档位被同日 fairbase 三测**上修**：本表加一列/一行 **EHF Hands + UBody Hands**（或脚注引 Table 1），把 ft 的 whole-body 侵蚀（16.85/10.59）与我们的 15.61/10.00 并排——本表从"我们输 0.24"单行叙事改为「in-domain 0.24mm 让步 ↔ whole-body 全面赢」的 trade-off 叙事。**wrist-rel 必须按公平基线口径入表**（此前"略好"只对 stock 成立）。fairbase 口径 caveat：snapshot_0=1 epoch、"仍在降"——侵蚀轨迹图（训满 4ep 逐点评）是本叙事的封面证据，先跑防翻案。

## Table 3 — 泛化表（held-out，诚实分析）
testset = **HInt（NewDays/VISOR/Ego4D 子集）+ HAA500**；2D PCK 类指标

| 行 | HInt `[all wa]` | HAA500 |
|---|---|---|
| OSX | 0.487 基准 | 基准 |
| **OSX 公平基线** | **0.480**（7-04 补） | 待测 |
| **Ours** | 0.481（**与 ft 精确平**） | 🔍/🟢（fork 数据，自定协议） |

- **7-04 改写**：原卖点"两模型都没训过 → 纯泛化小赢"**只对 stock 成立**；对公平基线是精确平局（visible/occluded/per-level 逐行一致）→ 0.487→0.480 归微调数据。本表的新角色 = **支撑证据**："我们的 in-domain 特化没有以 held-out 2D 泛化为代价（与 ft 持平）"，与 Table 1/2 的 whole-body 抗侵蚀主张互补，诚实度仍拉满，但不再分胜负、不当 headline。

## Table 4 — 消融表（reviewer 最较真，决定接收）
固定 testset = **InterHand + EHF-hands + UBody-hands**（7-04 起加后两者：消融要同时回答 in-domain 与抗侵蚀两轴），逐行只加一个东西：

| 行 | 配置 | 看什么 |
|---|---|---|
| (a) | 冻结骨干 + OSX 式 decoder（=公平基线）⚠️ 注：即"6 层固定参考点特征精修器"——其坐标机制在发布代码中被禁用（死代码，见 roadmap §0⁵），论文描述要按实际行为写 | 地板 + **侵蚀参照**（EHF 16.85/UBody 10.59） |
| (b) | (a) + **结构改进 decoder**（每层出坐标 + aux 监督 + 迭代参考点） | decoder Δ：对 PA≈0（gate 16.80≈snap2 16.78）、对 2D 为正——按"2D 机制/3D-PA 正交"写，**不再当 in-domain 方法贡献 🔑** |
| (c) | (b) + rotmat pose loss【已跑，正 −0.27】/ aux on-off【已跑，off 更差，注意 confound】/ L728【已跑，负 +0.25】 | loss Δ 三件套 |
| (d) | (c) + graft（Procrustes + 边界平滑）（可选） | graft Δ（主要在 vertex 指标） |
| (e) | (d) ± DCNv4 头（**换回 conv PositionNet + osx_l 暖启 + detach off，最高优先探针**） | 头 Δ + **抗侵蚀归因** + 测 §0⁵ 最大嫌疑簇 |
| **(f)** | 完整模型（rotmat snapshot_0） | 终值 |

**组件内消融**（各一小表；7-04 起核心问题从"为什么追不上 16.29"升级为"**哪个组件买来抗侵蚀**"——假设：aux 2D+热图监督是抗漂移锚，OSX decoder 无锚故 whole-body 漂移）：
- **posnet：DCNv4 vs conv PositionNet（osx_l 暖启）± detach**——首推，一石二鸟；
- **有/无 topo/occlusion 模块**（`my_decoder.py:454-458`，照旧必跑：方法节存在性论证）；
- decoder 层数 3 vs 6（仅当上两项有信号再跑）；
- 有/无 每层 aux 坐标监督【已有 aux0 数据，⚠️ confound：关 loss 后 bbox_embed 仍无监督迭代参考点，只证"有机制必须配监督"，写表时措辞按"机制内必要性"】；
- 有/无 迭代参考点（vs 固定 `coord_init`）【已有 gate vs snap2 数据：对 PA≈0、对 2D 为正】。

## Table 5 — 效率表（对 H4W++ 的差异化武器）
| 行 | #Params | FLOPs | FPS | 需要外部专家? |
|---|---|---|---|---|
| OSX | | | | 否 |
| H4W++ | | | | **是（WiLoR + YOLO + DWPose）** |
| **Ours** | | | | **否（单模型）** |

> H4W++ 手部更准但挂了三个外部模型。我们的故事 = "**单模型、无需外部专家/检测器，精度全面超 OSX**"。Table 5 是这句话的证据。

---

## 最小可发子集（时间紧时的优先级）

- **必做（定生死）**：Table 2（含 🔑 公平基线行）+ Table 4 的 (a)(b)(f) 三行。
- **强烈建议**：Table 1（EHF+UBody）+ Table 5（效率）。决定"全绿"成立 + 差异化。
- **加分（决定档位上限）**：Table 3 泛化 + Table 4 完整六行 + 组件内消融。够一够 3DV/WACV/BMVC，否则期刊/workshop。

## 档位预期（诚实）

| 档位 | 概率 | 备注 |
|---|---|---|
| Workshop（CVPRW/ICCVW/ECCVW） | 接近必中 | 分量轻、常非存档 |
| 中低 SCI（TCSVT / TMM / PR / CVIU / IVC / Neurocomputing / The Visual Computer） | 现实可期 | 全绿 + 干净 ablation + 诚实分析正中口味 |
| ICIP / ICME / ICPR / IEEE Access | 现实可期 | 接受增量、周期快 |
| 3DV / WACV / BMVC | 够一够 | 靠强 ablation + 效率角度 + 对标 H4W++ |

## 必做的两件"防过时"

1. **基线不止 OSX(2023)**：表里放 SMPLer-X、H4W++（、Multi-HMR）。不必全赢，能同台 / 在效率上差异化即可。
2. **正面对标 H4W++(CVPR26)**：related work 写清差异 = 单模型、不挂 WiLoR 重专家、无需额外手检测器、冻结骨干诚实分析。不写 → 好一点的地方 desk-reject。

---

## 一句话钉死

**生死闸已于 7-04 判定：没过（16.53 vs 16.29），按兜底转分析型；但同日 fairbase 三测把兜底上修成 trade-off 方法故事。** 现在唯一每天看的东西换成：**fairbase 侵蚀轨迹（训满 4ep，逐 snapshot 的 InterHand PA vs EHF/UBody Hands）是否坐实"它拿 in-domain 必丢 whole-body"。**
- 坐实（EHF/UBody 单调或持续劣于我们）→ 新 headline 成立，Table 1/2/3 按 7-04 口径填。
- 翻案（ft 后续 epoch EHF 回升到 ≤15.6 且 InterHand 继续降）→ 退回纯"高效可复现配方 + 冻结骨干分析"，whole-body 主张降级为"1-epoch 侵蚀现象"。
