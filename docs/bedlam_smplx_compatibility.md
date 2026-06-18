# 问题：
myosx 全局 SMPL-X 在 `common/utils/human_models.py` 中创建，它没有显式指定 `flat_hand_mean`。BEDLAM数据集中，`flat_hand_mean` 默认值是True

# 解决方案
 SMPL-X 的 flat_hand_mean 机制就是简单的 axis-angle 加法（已确认 body_models.py:1144），逆操作就是减法，精确等价，不是近似：

## 从 SMPL-X layer 提取 MANO hand mean（只需做一次）
pose_mean = smpl_x.layer['neutral'].pose_mean.detach().cpu().numpy()
lhand_mean = pose_mean[75:120]   # 45 维
rhand_mean = pose_mean[120:165]  # 45 维

## 在 BEDLAM loader 里转换：
lhand_pose = pose[75:120] - lhand_mean   # flat_hand_mean=True → False
rhand_pose = pose[120:165] - rhand_mean

转换后的 hand pose 喂进 myosx 的 smpl_x.layer（flat_hand_mean=False），layer 内部加回 hands_mean，最终结果和 BEDLAM 官方 layer（flat_hand_mean=True）数学上完全一致。


改动量：BEDLAM.py 里加约 10 行代码（初始化时取一次 hand mean，每条记录减一下），就解决了整个 flat_hand_mean 问题