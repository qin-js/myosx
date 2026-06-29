# UBody `[wa]` 逐关节分组归因方案

更新时间：2026-06-28

## 为什么做这个

当前 UBody `[wa]` 仍落后 OSX 0.010（snap2 0.2291 vs OSX 0.2188），编码器修复后已确认这 100% 是手头问题（body 与 OSX 逐比特一致）。但"手头问题"下面还分好几种可能根因，**路线选择直接取决于差距主体是哪一类**：

| 差距主体在 | 指向的根因 | 对症路线 |
|---|---|---|
| 小手 / 低分辨率 crop | ROI crop 质量差 | Route C（解冻 ROI）、WA2D 的 ROI 部分 |
| 重遮挡手 | 可见性/监督不足 | 更多自然手监督、soft gate |
| 清晰大手的中段指节 | decoder 手形先验不如 OSX | WA2D loss、teacher distillation |
| 末端 tip/j3（已有证据） | 末端 articulation 泛化差 | tip/j3 加权（已试，仅小 polish） |
| 左右手不对称 | 翻转增强 / 标注偏置 | 查 MSCOCO label、增强策略 |

现在 `bootstrap_UBody.npz` 存的是**逐手标量**（每只手一个 NME，per-finger/per-level 已各自 `mean()` 压过），只能回答"哪根指/哪个节整体差"，**不能在同一只手上同时打多个分组标签**。本方案补一个**逐关节 dump**，让差距能按 hand-size / occlusion / left-right / finger×level 交叉分组，从而定位根因、决定 WA2D 这条路有没有上限。

**前置已确认（2026-06-28）**：OSX 与 snap2 的逐手数组**逐元素对齐**——`hand2d_n` 都 3446，`hand2d_nme_wa`/per-hand dump 都 3432，每个 per-finger 子集样本数完全一致。说明两次 eval 走的是 deterministic 同一批样本、同一顺序（`test_sample_interval=10` + 固定 split），**逐手配对合法**。实际分析脚本还会再检查 side / hand-size / visible-count / img_path，避免顺序不一致时误做配对。

## 现状：dump 粒度差在哪

`UBody.py:895-932` 的 eval 循环里，**逐关节误差已经算出来了**：

```python
err_wa = np.linalg.norm(pred_wa - gt, axis=1) / span   # (21,) 逐关节 wa 误差  ← line 917
ew = err_wa[v]                                          # 只取可见关节
eval_result['hand2d_nme_wa'].append(float(ew.mean()))   # ← 立刻 mean() 压成标量
```

问题就是最后一行 `.mean()` 把 21 维向量压掉了。所以改动**极小**——不是重写评测，是在 `mean()` 之前把原始向量 + 手属性存下来。

## 第一层：零代码改动，立刻能跑

用现有 npz 就能做的窄问题（今天出结果）：

### 1. 左右手不对称
snap2 `wa_l`=0.224 vs `wa_r`=0.234（右差 0.010）。对比 OSX 的左右差。若我们的左右差显著大于 OSX → 退化集中在右手 → 查翻转增强 / 右手更常被遮挡。

### 2. per-level 退化模式（已有数据，重新配对解读）
j1/j2/j3/tip 的逐手均值已在 npz。但**注意陷阱**：per-finger/per-level 是各自可见子集的统计，第 i 个元素不一定是同一只手（不同手的可见关节不同）。要做"同一只手 snap2 vs OSX 的 j3 差多少"，必须用第二层的逐关节 dump + joint_valid mask，否则子集顺序错位。

## 第二层：加一小段 dump，做真正分组归因（核心）

### 改动位置
`data/UBody/UBody.py`，`evaluate()` 里 line 895-932 的 hand2d 循环。

### 改动内容
在 `eval_result` 初始化处（line 733 附近）加几个 list：

```python
eval_result['hand2d_wa_joints'] = []       # (N,21) 逐关节 wa 误差，核心
eval_result['hand2d_abs_joints'] = []      # (N,21) 逐关节 abs 误差
eval_result['hand2d_joint_valid'] = []     # (N,21) 可见性 mask
eval_result['hand2d_side'] = []            # (N,) 0=左 1=右
eval_result['hand2d_hand_size'] = []       # (N,) GT 手 bbox 对角线 span（px）
eval_result['hand2d_n_visible'] = []       # (N,) 可见关节数（遮挡代理）
eval_result['hand2d_img_path'] = []        # (N,) 样本定位
```

在 line 917 `err_wa = ...` 之后、`mean()` 之前，把这只手的逐关节量存下来（左右各存一条）：

```python
# err_wa 已是 (21,) 逐关节；span / v / side 都现成
eval_result['hand2d_wa_joints'].append(err_wa.astype(np.float32))
eval_result['hand2d_abs_joints'].append(err_abs.astype(np.float32))
eval_result['hand2d_joint_valid'].append(v.astype(np.float32))   # (21,) bool->float
eval_result['hand2d_side'].append(0.0 if side == 'l' else 1.0)
eval_result['hand2d_hand_size'].append(float(span))
eval_result['hand2d_n_visible'].append(float(int(v.sum())))
eval_result['hand2d_img_path'].append(annot['img_path'])
```

注意：`span`（line 901）就是 GT 手 keypoint bbox 对角线，已经是 hand-size 的现成代理；`v.sum()`（line 898）是可见关节数，遮挡代理。`annot['img_path']` 在 line 761 已拿到。**全部现成，不用新算。**

### dump 落盘
当前实现走的是单独 `perhand_<testset>.npz`，不污染 bootstrap 的 1D-array contract：

```python
# main/test.py
_save_perhand_npz(osp.join(cfg.result_dir, f'perhand_{tag}.npz'), eval_result)
```

数值 key 会被 stack 成 `(N,21)` 或 `(N,)`，`hand2d_img_path` 以字符串数组保存；保存前会检查各 key 的 N 是否一致。

### 跑两次 eval（OSX + snap2，都带新 dump）
```bash
cd main
export UBODY_ANNOTATION_DIR=/workspace/myosx/dataset/UBody/annotations_filtered
# A: OSX normal
python test.py --gpu 0 --testset UBody --exp_name output/eval_ubody_group \
  --decoder_setting normal --encoder_setting osx_l \
  --pretrained_model_path ../pretrained_models/osx_l.pth.tar --dump_analysis
# B: snap2 (pytorch)
python test.py --gpu 0 --testset UBody --exp_name output/eval_snap2_group \
  --decoder_setting pytorch --encoder_setting osx_l \
  --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
  --continue_train_path ../output/joint_polish_f/model_dump/snapshot_2.pth.tar --dump_analysis
```
两次必须**同一 `UBODY_ANNOTATION_DIR` / `test_sample_interval` / `test_batch_size`**（和现有 crosspath 守卫一致），保证逐手对齐。

### 分析脚本（纯 numpy，放 `tool/analysis/group_attribution.py`）

核心是：把 snap2 和 OSX 的 `(N,21)` 逐关节表对齐，按各种 mask 分组，算**配对 delta**（同一只手 snap2_err − osx_err），看 delta 集中在哪个分组。

```python
import numpy as np
A = np.load('.../eval_ubody_group/result/perhand_UBody.npz')      # OSX
B = np.load('.../eval_snap2_group/result/perhand_UBody.npz')      # snap2

wa_A   = A['hand2d_wa_joints']      # (N,21)
wa_B   = B['hand2d_wa_joints']      # (N,21)  必须同 N、同顺序
valid  = A['hand2d_joint_valid']    # (N,21)  OSX/snap2 可见性应一致（同 GT）
size   = A['hand2d_hand_size']      # (N,)
nvis   = A['hand2d_n_visible']      # (N,)
side   = A['hand2d_side']           # (N,)

# 配对 delta：同一只手同一关节，snap2 比 OSX 差多少
delta = wa_B - wa_A           # (N,21)，>0 表示 snap2 更差

def group_stat(mask, name):
    # 只在两模型都可见的关节上算
    if mask.ndim == 1:
        m = mask[:, None] & (valid > 0)
        n_hands = int(mask.sum())
    else:
        m = mask & (valid > 0)
        n_hands = int(mask.any(axis=1).sum())
    d = delta[m]
    print(f'{name:20s} N_hands={n_hands:5d}  mean_delta={d.mean():+.4f}  '
          f'(snap2 {wa_B[m].mean():.4f} vs osx {wa_A[m].mean():.4f})')

# A. 按 hand-size 三档
p33, p66 = np.percentile(size, [33, 66])
group_stat((size < p33), '小手')
group_stat((size >= p33) & (size < p66), '中手')
group_stat((size >= p66), '大手')

# B. 按遮挡三档（可见关节数）
group_stat(nvis < 8, '重遮挡(<8)')
group_stat((nvis >= 8) & (nvis < 16), '中遮挡')
group_stat(nvis >= 16, '清晰手(>=16)')

# C. 左右手
group_stat(side == 0, '左手')
group_stat(side == 1, '右手')

# D. finger × level 交叉（在同一批手上）
FINGER = {'thumb':[1,2,3,4],'index':[5,6,7,8],'middle':[9,10,11,12],'ring':[13,14,15,16],'pinky':[17,18,19,20]}
LEVEL  = {'j1':[1,5,9,13,17],'j2':[2,6,10,14,18],'j3':[3,7,11,15,19],'tip':[4,8,12,16,20]}
for ln, li in LEVEL.items():
    idx = li
    m = np.zeros_like(valid, dtype=bool); m[:, idx] = True
    group_stat(m, f'level {ln}')

# E. failure case：最差的 50 只手
per_hand = (wa_B * valid).sum(1) / valid.sum(1).clip(min=1)
worst = np.argsort(per_hand)[-50:]
print('worst-50 img_path / size / nvis / side:')
for i in worst:
    print(f'  {A["hand2d_img_path"][i]}  size={size[i]:.0f} nvis={nvis[i]:.0f} side={side[i]:.0f} '
          f'wa_snap2={per_hand[i]:.3f} wa_osx={((wa_A[i]*valid[i]).sum()/valid[i].sum()):.3f}')
```

### 怎么读结果 → 驱动什么决策

读 **delta 集中在哪个分组**（`mean_delta` 越大 = snap2 在这组比 OSX 差得越多）：

- **delta 集中在"小手"档** → ROI/crop 是主因 → WA2D loss 的 ROI 对齐部分对症，Route C 也值得重估（但 box_net 已与 OSX 等价，ROI 机制理论收益缩水）。**WA2D 直接 loss 上限可能有限**。
- **delta 集中在"重遮挡"档** → 监督/可见性问题 → WA2D loss 帮助有限（loss 只在可见关节上算，遮挡手本来就少监督）；应转向更多自然手监督或 soft gate。
- **delta 在"清晰大手"也有、且集中在 j2/j3（非末端）** → decoder 手形先验问题 → **WA2D loss 对症，teacher distillation 更对症**（把 OSX 自然图手形先验灌进来）。这是 WA2D 路线最希望看到的结果。
- **delta 仍在 tip/j3 末端** → 和现有结论一致，WA2D 覆盖全 21 点比 tipw 强，但天花板可能就在这。
- **delta 左右严重不对称** → 查 MSCOCO/UBody 标注偏置或翻转增强。

### failure-case 可视化（最直接）
E 步骤吐出最差 50 只手的 img_path。用 `demo.py` 或现有 vis 流程，把这 50 只手在 snap2 vs OSX 下的投影画出来对比，**肉眼判断**：是小手 crop 错了、手指 articulation 错了、还是 wrist 摆偏了。这比任何统计都直接，也是文档"后续路线"里 failure-case 可视化的最小实现。

## 成本与优先级

- 改动：`UBody.py` eval 循环加逐关节 append；`test.py` 新增 `_save_perhand_npz` 单独保存 per-hand dump。**半小时内能写完。**
- 运行：两次 UBody eval（带 `--dump_analysis`），各约 10-15 分钟。
- 分析：纯 numpy 脚本，秒级。
- **总成本 < 1 小时，不训练，不占 GPU 训练线。** 建议在 WA2D 第一轮启动前/同时做，因为它决定 WA2D 这条路有没有上限，以及 teacher distillation 是否该提前。

## 与现有工具的关系

- 复用 `test.py --dump_analysis` 的 dump 机制（只是多存几个 key）。
- 复用 `tool/analysis/` 的纯 numpy 风格。
- 不影响现有 bootstrap / crosspath 流程（当前实现是单独 `perhand_<testset>.npz`）。
- 配对合法性先由现有 bootstrap CI 间接验证（样本数逐一对齐），再由 `group_attribution.py` 显式检查 side / hand-size / visible-count / img_path。
