# 论文表格规划（"全绿 vs OSX" 非顶会版）

更新时间：2026-07-01

本文件把"能发的最小实验集 + 表格骨架"固化下来，配合 decoder 结构性大改边训边对。
- 目标定位：**非顶会**。相对 stock OSX 全面提升（全绿）+ 一个能写成一句话的方法贡献（结构改进 decoder）。
- 结果记录见 `docs/experiment_log.md`；路线/决策见 `docs/post_stage3_roadmap.md`；竞品分析见记忆 `h4wpp-competitor.md`。
- 一句话定位：卖点写成 **"在冻结骨干上、用结构改进的 decoder 提升手/脸且可证明身体零退化的高效单模型配方"**，**不写**"我们打败了 OSX 的 decoder"（会被公平基线反杀）。

---

## 0. 图例（每格的角色）

| 标记 | 含义 |
|---|---|
| 🟢 | **必须赢**——对 **stock OSX**，构成"全绿"对比表 |
| 🔑 | **必须赢**——对 **公平基线 OSX**（OSX decoder 同条件微调），= **方法闸**，过不了就没有方法贡献 |
| ⚪ | **同台即可**——对近期 SOTA（SMPLer-X / H4W++ / Multi-HMR），差异化靠效率 + 诚实 |
| 🔍 | **诚实分析**——平手/小赢都算成功（held-out 泛化、身体不退化、效率） |
| ⚠️ | **风险格**——训练域外（当前训 InterHand+UBody+MSCOCO），可能退化，重点盯 |

**已知种子数字**（InterHand PA-MPJPE，全量）：stock OSX **19.58** · OSX 公平基线 **16.29**（仍在降，未平台）· 我们 snap2（旧 3 层 decoder）**16.78**。HInt `[all wa]`：OSX **0.487** / 我们 **0.480**。EHF Hands PA-MPVPE snap2 **15.67**，Face **6.15**。

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
- **EHF hands = ⚠️**：EHF 不在训练域，手能否泛化是全绿最大风险。盯它；真退化就补 EHF-like 手数据，或至少做到不退（平手算绿）。
- **7-02 更新**：UBody hands `[wa]` 格有了 body 侧杠杆——T1 已做到 **0.204 反超 OSX 0.219**，但原样伴随 `[abs]` 0.367 崩 + EHF Face 6.39，**须等 shape/cam 拆分消融拿到"只赢不崩"版本才能上表**；`[abs]` 与 EHF Face 因此升级为**受监护格**（任何 T1 变体上表前必须核）。AGORA 现可测（`--testset AGORA`），可作 Table 1 附加列（T1 在 AGORA PA 全线小赢）。

## Table 2 — 手部专项表（核心战场 / 生死闸）
testset = **InterHand26M (+ HIC 若有)**；列 = MPJPE / MPVPE / MRRPE（per-hand root+scale 对齐，IntagHand 协议）

| 行（方法） | MPJPE | MPVPE | MRRPE |
|---|---|---|---|
| OSX (stock) = 19.58 | 🟢 | 🟢 | 🟢 |
| **OSX 公平基线 = 16.29↓** | 🔑 | 🔑 | 🔑 |
| H4W++ / WiLoR 系 | ⚪ | ⚪ | ⚪ |
| **Ours（新 decoder）** | 🔑 必须 < 16.29↓ | 🔑 | 🔑 |

> **这张表的 🔑 行是整篇论文的生死闸。** 新 decoder 压不过公平基线（且它还在降），方法贡献不成立，只能退回分析型论文。**这是修 decoder 期间唯一要每天盯的数字。**

## Table 3 — 泛化表（held-out，诚实分析）
testset = **HInt（NewDays/VISOR/Ego4D 子集）+ HAA500**；2D PCK 类指标

| 行 | HInt `[all wa]` | HAA500 |
|---|---|---|
| OSX | 0.487 基准 | 基准 |
| **Ours** | 🔍 0.487→0.480（小赢=bonus） | 🔍/🟢（fork 数据，自定协议） |

- 卖点 = "**两模型都没训过 → 纯泛化**"，诚实度拉满，不靠它分胜负。

## Table 4 — 消融表（reviewer 最较真，决定接收）
固定 testset = **InterHand + EHF-hands**，逐行只加一个东西：

| 行 | 配置 | 看什么 |
|---|---|---|
| (a) | 冻结骨干 + OSX 式 decoder（=公平基线） | 地板 |
| (b) | (a) + **结构改进 decoder**（每层出坐标 + aux 监督 + 迭代参考点 + 层数↑） | **decoder Δ = 方法贡献** 🔑 |
| (c) | (b) + 多参考系 loss（part / inter-hand / IH-vec + 显式 root_pose） | loss Δ |
| (d) | (c) + graft（Procrustes + 边界平滑） | graft Δ（主要在 vertex 指标） |
| (e) | (d) + DCNv4 头（若保留） | 头 Δ |
| **(f)** | 完整模型 | 终值 |

**组件内消融**（各一小表，证明结构选择，直接对应 `post_stage3_roadmap.md` §0‴ 根因）：
- decoder 层数 3 vs 6；
- 有/无 每层 aux 坐标监督；
- 有/无 迭代参考点（vs 全程共用一次 `coord_init`）；
- **有/无 topo/occlusion 模块**（`my_decoder.py:454-458`，怀疑帮倒忙——砍掉有理或证明有用，两种结果都诚实可写）。

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

修 decoder 期间唯一每天看的数字：**Table 2 里 Ours 的 MPJPE 有没有钻到公平基线 16.29（且它还在降）以下。**
- 过闸 → Table 1/3/5 基本顺手填绿，方法论文成立。
- 不过闸 → 老实转 "高效可复现配方 + 冻结骨干分析" 分析型论文（更低档但仍可发），且**不得声称方法优越性**。
