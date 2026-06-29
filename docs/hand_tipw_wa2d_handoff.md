# Hand TipW / WA2D Handoff

更新时间：2026-06-28

本文件整理当前会话中的实验结果、判断和下一步建议，供转交他人讨论。项目背景：OSX 研究分支 `myosx`，当前分支 `dev`；目标是在冻结 OSX backbone 的 PyTorch decoder 路径上微调手/脸头，争取手部姿态超过原始 OSX。

## 当前基线和目标

干净基线是 `joint_polish_f snapshot_2 @ fixed encoder`：

- 评估目录：`output/eval_joint_polish_f/result/snapshot_2`
- checkpoint：`output/joint_polish_f/model_dump/snapshot_2.pth.tar`
- UBody natural-hand `[wa] NME = 0.2291`
- UBody natural-hand `tip NME = 0.2880`
- UBody natural-hand `[abs] NME = 0.3236`
- InterHand PA-MPJPE 约 `16.78`
- HInt `[all wa] NME = 0.480`
- EHF Face PA-MPVPE 约 `6.15`

原始 OSX baseline：

- 评估目录：`output/eval_ubody/result`
- UBody natural-hand `[wa] NME = 0.2188`
- 这意味着 snap2 相对 OSX 仍有约 `0.0103` 的 UBody `[wa]` gap。

阶段目标仍是提高 UBody full-body natural-hand 指标，尤其 `[wa]`，同时守住 InterHand / HInt / EHF guardrail。

## Route C 结论

Route C 指 `--train_hand_roi`，解冻 `hand_roi_net` 小学习率共适应。

已试过两类：

- Clean C，InterHand/UBody only，`hand_roi_lr=1e-6`：UBody 没有改善，`[wa]` 仍约 `0.229`。
- C + tipw + MSCOCO，`hand_roi_lr=3e-6`：训练约 itr700 后不稳定，UBody 到 itr1000 明显变差，`[wa]` 约 `0.260`，tip 约 `0.364`。

判断：Route C 不是当前优先方向。它要么无效，要么高 LR/MSCOCO 下不稳定，收益/风险比差。

## hand_tipw 实验

训练目录：`output/hand_tipw`

配置大意：

- warm-start：`../output/joint_polish_f/model_dump/snapshot_2.pth.tar`
- dataset mix：`InterHand26M=0.35 / UBody=0.25 / MSCOCO=0.40`
- frozen `hand_roi_net`
- 训练手部 position/decoder/regressor
- `--hand_tip_loss_weight 2.0`
- `--hand_j3_loss_weight 1.5`

关键 UBody 结果：

| checkpoint | UBody `[wa]` | tip | `[abs]` | PA Hands |
|---|---:|---:|---:|---:|
| snap2 fixed | 0.229 | 0.288 | 0.324 | 10.15 |
| hand_tipw itr3000 | 0.227 | 0.284 | 0.315 | 10.08 |
| hand_tipw itr4000 | 0.227 | 0.284 | 0.315 | 10.10 |

guardrail：

| checkpoint | InterHand PA | HInt `[all wa]` | EHF Hands | EHF Face |
|---|---:|---:|---:|---:|
| hand_tipw itr3000 | 16.95 | 0.480 | 15.62 | 6.15 |
| hand_tipw itr4000 | 16.97 | 0.480 | 15.65 | 6.15 |

判断：

- 选 `snapshot_0_itr3000`，不是 itr4000。
- itr3000 与 itr4000 UBody natural-hand 基本一样，但 itr3000 的 InterHand/EHF/PA Hands 略好。
- hand_tipw 是有效小幅 polish，不是突破。

## Bootstrap 统计结论

使用工具：

```bash
python tool/analysis/bootstrap_ci.py \
  --a A.npz \
  --b B.npz \
  --key hand2d_nme_wa \
  --lower-is-better
```

### snap2 vs hand_tipw itr3000

A = `output/eval_joint_polish_f/result/snapshot_2/bootstrap_UBody.npz`

B = `output/eval_hand_tipw_bootstrap/result/snapshot_0_itr3000/bootstrap_UBody.npz`

`hand2d_nme_wa`：

- A mean `0.2291`
- B mean `0.2274`
- paired delta `A - B = 0.0016`
- 95% CI `[0.0003, 0.0029]`
- `P(delta>0)=0.993`
- 结论：B 显著更好，但幅度很小。

`hand2d_wa_nme_tip`：

- A mean `0.2880`
- B mean `0.2841`
- paired delta `0.0039`
- 95% CI `[0.0021, 0.0057]`
- 结论：tip 改善显著。

`hand2d_nme_abs`：

- A mean `0.3236`
- B mean `0.3154`
- paired delta `0.0082`
- 95% CI `[0.0039, 0.0124]`
- 结论：absolute 2D 也显著改善。

### OSX vs hand_tipw itr3000

A = `output/eval_ubody/result/bootstrap_UBody.npz`

B = `output/eval_hand_tipw_bootstrap/result/snapshot_0_itr3000/bootstrap_UBody.npz`

`hand2d_nme_wa`：

- A mean `0.2188`
- B mean `0.2274`
- paired delta `A - B = -0.0086`
- 95% CI `[-0.0094, -0.0078]`
- `P(delta>0)=0.000`
- 结论：hand_tipw itr3000 仍显著不如原始 OSX。

总体判断：

- snap2 gap：`0.2291 - 0.2188 = 0.0103`
- tipw gain：`0.2291 - 0.2274 = 0.0017`
- tipw 大约只补回原 gap 的 16%。

## 为什么继续只拉长 tipw 不够

itr3000 / itr4000 已平台：

- UBody `[wa]` 都约 `0.227`
- tip 都约 `0.284`
- guardrail 没有进一步改善

因此继续同一路线训练，大概率只在 `0.227` 附近波动，难以追到 OSX 的 `0.2188/0.219`。

剩余 gap 更可能来自：

- 训练目标没有直接对齐 UBody wrist-aligned natural-hand metric；
- 自然图手部真实 2D 监督分布不足或 noisy；
- PyTorch decoder/hand head 与 OSX normal decoder 仍有局部手形先验差异；
- 单纯 tip/j3 加权只修末端局部，无法补完整相对手形 gap。

## 新增代码：direct wrist-aligned 2D hand loss

本会话已新增一个默认关闭的 loss，用来直接对齐 UBody `[wa]` 指标。

改动文件：

- `main/config.py`
- `main/train.py`
- `main/model_core.py`
- `data/UBody/UBody.py`
- `data/MSCOCO/MSCOCO.py`
- `data/dataset.py`

新增 CLI：

```bash
--hand_wa_2d_loss_weight
--hand_wa_2d_loss_sources
--hand_wa_2d_loss_min_joints
```

实现要点：

- 默认 `hand_wa_2d_loss_weight = 0.0`，关闭时旧实验行为不变。
- UBody/MSCOCO 训练 loader 会额外输出 `coco_hand_joint_img` 和 `coco_hand_joint_trunc`，形状为 `(2, 21, 3)` / `(2, 21, 1)`。
- 21 点顺序是每只手：wrist + thumb/index/middle/ring/pinky，每指 4 点。
- loss 在 hand target/proj 被改到 hand ROI 坐标之前计算，即 full-body heatmap 坐标。
- pred 和 GT 先用真实 hand wrist 做 translation alignment。
- 误差按 GT hand keypoint bbox diagonal 归一化。
- 只对 source mask 允许的数据源启用，默认 `ubody,mscoco`。
- 日志里会出现：
  - `loss_hand_wa_2d`
  - `loss_hand_wa_2d_raw`
  - `loss_hand_wa_2d_active`

已做验证：

- `python -m py_compile main/config.py main/train.py main/model_core.py data/UBody/UBody.py data/MSCOCO/MSCOCO.py data/dataset.py` 通过。
- `git diff --check -- ...` 通过。
- 没有跑完整训练。当前交互环境导入 DCNv4 时 CUDA 枚举失败，实际 smoke train 需要在正常训练环境跑。

注意：工作树里还有一些和本次无关的脏文件，例如 `demo/*.npy`、`main/train.sh`、`main/test.sh` 等，本次未处理。

## 下一步建议

第一轮不要改数据集比例，先做纯 loss ablation。

保持当前比例：

```text
InterHand 0.35 / UBody 0.25 / MSCOCO 0.40
```

原因：现在要验证 direct wrist-aligned 2D loss 本身是否有效。如果同时改比例，结果不好时无法判断是 loss 问题还是采样比例问题。

建议启动命令：

```bash
cd /workspace/myosx/main
export UBODY_ANNOTATION_DIR=/workspace/myosx/dataset/UBody/annotations_filtered

python train.py \
  --gpu_ids 0 --lr 1e-5 --lr_mult 0.1 \
  --train_batch_size 64 --num_thread 8 \
  --end_epoch 1 --phase1_epochs 0 \
  --exp_name output/hand_wa2d_tipw \
  --decoder_setting pytorch --encoder_setting osx_l --grad_clip 1.0 \
  --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
  --init_trained_path ../output/hand_tipw/model_dump/snapshot_0_itr3000.pth.tar \
  --posnet_lr_mult 0.25 \
  --hand_tip_loss_weight 2.0 --hand_j3_loss_weight 1.5 \
  --hand_wa_2d_loss_weight 1.0 \
  --hand_wa_2d_loss_sources ubody,mscoco \
  --save_iters 500
```

第一轮观察：

- 训练日志中 `loss_hand_wa_2d_raw` 是否有限、稳定下降或至少不爆；
- `loss_hand_wa_2d_active` 是否非零；
- itr500 / itr1000 的 UBody `[wa]` 是否低于 hand_tipw itr3000 的 `0.2274`；
- 同时看 InterHand / HInt / EHF 是否明显掉。

建议 stop/go：

- 如果 UBody `[wa] <= 0.226`：说明 direct loss 有效，可以继续调数据比例或 loss 权重。
- 如果 UBody `[wa]` 没动或变差：先不要改比例，优先试 `--hand_wa_2d_loss_sources ubody` 或降低/升高 loss weight。
- 如果 UBody 改善但 InterHand/HInt 明显掉：考虑降低 UBody 强度、增加 InterHand 比例，或只在短程后 early stop。

第二轮才考虑改比例：

```text
InterHand 0.30 / UBody 0.45 / MSCOCO 0.25
```

或更激进：

```text
InterHand 0.25 / UBody 0.55 / MSCOCO 0.20
```

但这应建立在第一轮证明 `hand_wa_2d_loss` 有正收益之后。

## 可能的后续路线

如果 direct WA2D loss 仍无法把 UBody `[wa]` 推到 `0.224~0.225`，建议不要继续盲调 loss。下一步更可能需要：

- 对 UBody 做 residual / failure case 可视化：比较 OSX、snap2、hand_tipw、WA2D 的差异样本；
- 按 finger / level / left-right / hand size / occlusion 分组分析剩余 gap；
- 做 OSX teacher distillation：用原始 OSX normal decoder 在自然图上生成 teacher hand joints/pose，训练 PyTorch hand decoder 先追平 teacher，再用 GT/tip/WA2D loss 尝试超过；
- 检查 MSCOCO hand 2D label 噪声，必要时只用 UBody 或给 MSCOCO 更低权重。

一句话阶段结论：

> `hand_tipw itr3000` 相比 `snap2@fixed` 在 UBody natural-hand 上有统计显著但很小的提升，尤其改善 tip；但它仍显著落后原始 OSX。下一步应验证 direct wrist-aligned 2D hand loss，而不是继续拉长 tipw 或重开 Route C。
