# Recent Experiment Summary: TipW / WA2D / Group Attribution

更新时间：2026-06-29

本文件是 `docs/hand_tipw_wa2d_handoff.md` 与
`docs/wa2d_group_attribution.md` 之后的补充小结，单独记录最近一轮实验、
代码改动、结果判断和下一步建议。

## 当前问题

目标仍是提升 UBody full-body natural-hand wrist-aligned 2D 指标，最好追平或
超过原始 OSX，同时守住 InterHand / HInt / EHF。

当前主要参照：

| 模型 / checkpoint | UBody `[wa]` | tip | `[abs]` | 备注 |
|---|---:|---:|---:|---|
| 原始 OSX | 0.2188 / group 0.2182 | group 0.2622 | 0.311 左右 | 目标基线 |
| `joint_polish_f snapshot_2` | 0.2291 / group 0.2281 | 0.2880 / group 0.2874 | 0.3236 | fixed encoder 后的干净工作基线 |
| `hand_tipw itr3000` | 0.2274 / group 0.2268 | 0.2841 / group 0.2837 | 0.3154 | 目前稳定最佳之一 |
| `hand_wa2d_distal_guard itr500` | 0.226 / group 0.2256 | 0.281 / group 0.2810 | 0.315 | 最近 WA2D 正向信号 |

结论：`hand_tipw` 和 WA2D 都有小幅正收益，但目前仍显著落后 OSX，
剩余 gap 约 `0.0075` 到 `0.0086`。

## Group Attribution 结果

新增 per-hand / per-joint dump 后，比较：

- A(ref): `output/eval_ubody_group/result/perhand_UBody.npz`
- B(snap2): `output/eval_snap2_group/result/snapshot_2/perhand_UBody.npz`

主要结果：

```text
overall [wa]  A=0.2182  B=0.2281  delta=+0.0099
```

差距按 hand size / occlusion / side 基本都是全局存在：

- small/mid/large 都更差，delta 约 `+0.0087 ~ +0.0108`
- clear hand 占绝大多数，delta `+0.0100`
- 左右手接近，left `+0.0094`，right `+0.0104`

最关键的是 per-level：

| level | OSX | snap2 | delta |
|---|---:|---:|---:|
| j1 | 0.2049 | 0.2058 | +0.0009 |
| j2 | 0.2203 | 0.2243 | +0.0040 |
| j3 | 0.2298 | 0.2415 | +0.0117 |
| tip | 0.2622 | 0.2874 | +0.0252 |

判断：

- 不是小手 / 遮挡 / 左右手某一类导致的单点问题。
- 主要 gap 集中在 distal articulation，尤其 tip，其次 j3。
- j1 已经几乎追平 OSX，继续优化应重点看 j3/tip，而不是整体平移。

对 `hand_tipw itr3000` 做同样 group attribution：

```text
overall [wa]  A=0.2182  B=0.2268  delta=+0.0086
tip           A=0.2622  B=0.2837  delta=+0.0216
```

相比 snap2，tipw 确实缩小了 gap，但只补回一小部分。

## Direct WA2D Loss 实验

新增 direct wrist-aligned 2D hand loss，默认关闭。核心逻辑：

- 在 full-body heatmap 坐标中计算，不经过 hand ROI 坐标改写。
- pred / GT 先按 wrist 做 translation alignment。
- 用 GT hand keypoint bbox diagonal 归一化，直接对齐 UBody `[wa]` metric。
- 支持 source mask，当前主要用 `ubody,mscoco`。
- 支持 j1/j2/j3/tip level weight。

新增和修复过的保护：

- `hand_wa_2d_loss_sources` 支持中文逗号归一化，例如 `ubody，mscoco`。
- 非法 source 会直接报错，避免 silent inactive。
- WA2D 距离从 raw `torch.linalg.norm()` 改为带 eps 的
  `sqrt(sum(diff^2) + 1e-8)`，并先清掉 invalid joint diff。
- 训练中加入 non-finite loss / grad / param guard。

## WA2D 失败 / 无效结果

### Aggressive WA2D

目录：`output/hand_wa2d_distal`

大致配置：

- `hand_wa_2d_loss_weight=1.0`
- sources: `ubody`
- level weights: `j1=0.25, j2=0.5, j3=1.5, tip=3.0`

itr500 UBody 结果明显崩：

```text
[wa] NME = 0.260
tip NME = 0.364
PA MPJPE Hands = 10.93
```

判断：权重过强，并且当时还没有完整 non-finite guard。该 checkpoint 不建议再用。

### `hand_wa2d_distal_2`

目录：`output/hand_wa2d_distal_2`

这次看似是 conservative 设置，但实际有两个问题：

- source 曾传入中文逗号形式 `ubody，mscoco`，在修复前会导致 WA2D inactive。
- itr500 结果与 aggressive bad checkpoint 表现一致，不能视作有效 conservative 结果。

判断：这条结果不作为路线依据。

## Guarded WA2D Run

目录：`output/hand_wa2d_distal_guard`

配置要点：

- warm-start: `../output/hand_tipw/model_dump/snapshot_0_itr3000.pth.tar`
- `hand_wa_2d_loss_weight=0.1`
- sources: `ubody,mscoco`
- level weights: `j1=0, j2=0, j3=1, tip=1`
- `hand_tip_loss_weight=2.0`
- `hand_j3_loss_weight=1.5`
- position net 仍可训练，但 lr 很小：`posnet_lr_mult=0.25`

训练现象：

- 修复 WA2D norm 后，训练能跑到 itr820。
- loss 在 itr799 仍正常，没有整体爆炸。
- 但 itr820 后 backward 出现 non-finite gradients，仍集中在
  `module.hand_position_net.dcnv4_blocks.0`。

判断：

- 不是 WA2D loss 数值本身直接爆。
- 更像 `hand_position_net` / DCNv4 backward 在少数自然手 batch 上不稳定。
- `posnet_lr_mult=0` 或小 lr 不够，因为只要参与 backward，grad 仍可能 NaN。

## Guarded WA2D itr500 评测

checkpoint：

```text
output/hand_wa2d_distal_guard/model_dump/snapshot_0_itr500.pth.tar
```

UBody 结果：

```text
PA MPVPE Hands: 10.14 mm
MPVPE Hands:    38.96 mm
PA MPJPE Hands: 10.45 mm

[abs] NME: 0.315
[wa]  NME: 0.226
tip   NME: 0.281
```

Group attribution vs OSX：

```text
overall [wa]  A=0.2182  B=0.2256  delta=+0.0075
j1            A=0.2049  B=0.2056  delta=+0.0007
j2            A=0.2203  B=0.2234  delta=+0.0031
j3            A=0.2298  B=0.2386  delta=+0.0088
tip           A=0.2622  B=0.2810  delta=+0.0188
```

与 `hand_tipw itr3000` 相比：

- overall gap 从约 `+0.0086` 缩到 `+0.0075`
- tip gap 从约 `+0.0216` 缩到 `+0.0188`
- 3D hand 指标没有坏，反而略好

判断：

- 这是正向信号，说明 direct WA2D 确实在拉 distal/tip。
- 但收益仍很小，还没有达到可以写论文的程度。
- 当前 checkpoint 可作为候选 warm-start，但不能继续用原设置训练，因为后面 itr820 会炸。

## 新增稳定路线：冻结 hand_position_net

为了解决 DCNv4 backward 非有限梯度，新增：

```bash
--freeze_hand_position_net
```

实现要点：

- 默认关闭，不影响旧实验。
- 开启后 `hand_position_net.requires_grad=False`。
- `hand_position_net` 不进入 optimizer。
- 仍保留在 `trainable_module_names` 中，所以 lightweight snapshot 继续保存
  warm-start 的 position head 权重。
- 后续只训练 `hand_decoder` 和 `hand_regressor`。

相关文件：

- `main/config.py`
- `main/train.py`
- `main/model_core.py`

验证：

```bash
python -m py_compile main/config.py main/train.py main/model_core.py
git diff --check -- main/config.py main/train.py main/model_core.py
python main/train.py --help
```

## Freeze Position Net 结果

目录：

```text
output/hand_wa2d_distal_guard_freeze_hand_position_net
```

这条线从 `hand_wa2d_distal_guard snapshot_0_itr500` warm-start，开启
`--freeze_hand_position_net`，继续只训练 `hand_decoder` / `hand_regressor`。

### itr500

Group attribution vs OSX：

```text
overall [wa]  A=0.2182  B=0.2250  delta=+0.0068
j1            A=0.2049  B=0.2055  delta=+0.0006
j2            A=0.2203  B=0.2231  delta=+0.0028
j3            A=0.2298  B=0.2379  delta=+0.0080
tip           A=0.2622  B=0.2795  delta=+0.0174
```

相比 non-freeze guarded itr500：

| checkpoint | overall B | gap vs OSX | j3 gap | tip gap |
|---|---:|---:|---:|---:|
| guarded itr500 | 0.2256 | +0.0075 | +0.0088 | +0.0188 |
| freeze itr500 | 0.2250 | +0.0068 | +0.0080 | +0.0174 |

判断：冻结 position net 不只是避免 DCNv4 NaN，也让 WA2D polish 更干净。
这是目前 UBody natural-hand 上最好的候选。

### itr1000

Group attribution vs OSX：

```text
overall [wa]  A=0.2182  B=0.2251  delta=+0.0069
j1            A=0.2049  B=0.2055  delta=+0.0006
j2            A=0.2203  B=0.2231  delta=+0.0028
j3            A=0.2298  B=0.2379  delta=+0.0081
tip           A=0.2622  B=0.2796  delta=+0.0175
```

itr1000 相比 itr500 基本无进步：

| checkpoint | overall B | gap vs OSX | j3 gap | tip gap |
|---|---:|---:|---:|---:|
| freeze itr500 | 0.2250 | +0.0068 | +0.0080 | +0.0174 |
| freeze itr1000 | 0.2251 | +0.0069 | +0.0081 | +0.0175 |

判断：当前 `hand_wa_2d_loss_weight=0.1` + freeze-pos 设置的收益主要在前
500 iter 吃完，之后进入平台。继续同配置拉长训练不优先，`freeze itr500`
应暂定为这条线的 best。

## 当前推荐下一步

### 1. 先补 guardrail

对 `hand_wa2d_distal_guard_freeze_hand_position_net itr500` 补：

- InterHand PA / wrist-rel
- HInt `[all wa]`
- EHF Hands / Face

如果 guardrail 没明显坏，就把它视作当前最佳候选。

### 2. 做 bootstrap 验证小收益

`freeze itr500` 相比 `hand_tipw itr3000` 的提升只有约 `0.001` 到
`0.002`，必须做 paired bootstrap：

- `hand2d_nme_wa`
- `hand2d_wa_nme_tip`
- `hand2d_nme_abs`

只有 bootstrap 显著，才把 freeze itr500 作为正式“比 tipw 更好”的结果记录。

### 3. 不继续同配置长训

`itr1000` 没有继续下降，因此当前 `weight=0.1` 的 freeze-pos run 不建议再
优先跑到更长。若已经有 itr1500，可评估确认；否则先停在 itr500。

### 4. 下一轮只改一个变量：WA2D weight

如果 guardrail 没坏，下一轮从同一个 `guarded itr500` warm-start，仍冻结
position net，只把：

```text
hand_wa_2d_loss_weight: 0.1 -> 0.2
```

短跑 500 iter。若 `0.2` 仍无法把 overall 推到 `0.224.x`，说明 direct WA2D
在固定 position head 下基本到上限，应转向 OSX teacher distillation 或更细的
thumb / distal 专项设计。

## 当前阶段判断

1. Route C 暂时不是优先路线。
2. tip/j3 hand-space 加权有效，但已经平台。
3. direct WA2D loss 有正向信号，尤其 tip/j3，但收益仍小。
4. `hand_position_net` 的 DCNv4 backward 不稳定已由 freeze-pos 路线绕开。
5. freeze-pos 版在 itr500 达到当前最佳，但 itr1000 平台，说明同配置长训不是出路。
6. 如果 freeze-pos WA2D 仍无法把 UBody `[wa]` 推到 `0.224` 左右，下一步应转向
   OSX teacher distillation，而不是继续盲调 WA2D 权重。
