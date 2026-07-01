import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

# ================================================================
#   可变形注意力导入（二选一）
#   方案 A: Deformable DETR 的 MSDeformAttn（默认）
#   方案 B: FlashDeformAttn（更快，需安装 flash_deform_attn）
# ================================================================
try:
    from DCNv4 import FlashDeformAttn as MSDeformAttn
    print("[Decoder] Using FlashDeformAttn (faster)")
except ImportError:
    from ops.modules import MSDeformAttn
    print("[Decoder] Using MSDeformAttn (default)")


# ================================================================
#                     Part 1: 工具函数
# ================================================================

def build_hand_adjacency(num_joints):
    """
    构建手部骨骼邻接矩阵。

    20 关节排布（5 指 × 4 关节，无手腕）:
        Thumb:  0  → 1  → 2  → 3
        Index:  4  → 5  → 6  → 7
        Middle: 8  → 9  → 10 → 11
        Ring:   12 → 13 → 14 → 15
        Pinky:  16 → 17 → 18 → 19

        掌部连接: 0─4─8─12─16

    21 关节排布（含手腕 joint 0）:
        Wrist: 0
        Thumb:  1  → 2  → 3  → 4
        Index:  5  → 6  → 7  → 8
        Middle: 9  → 10 → 11 → 12
        Ring:   13 → 14 → 15 → 16
        Pinky:  17 → 18 → 19 → 20
    """
    if num_joints == 20:
        edges = [
            # ---- 手指内部链 ----
            (0, 1), (1, 2), (2, 3),            # Thumb
            (4, 5), (5, 6), (6, 7),            # Index
            (8, 9), (9, 10), (10, 11),         # Middle
            (12, 13), (13, 14), (14, 15),      # Ring
            (16, 17), (17, 18), (18, 19),      # Pinky  ← 修复: 原版漏掉了小拇指
            # ---- 掌部跨指连接 ----
            (0, 4), (4, 8), (8, 12), (12, 16),
        ]
    elif num_joints == 21:
        edges = [
            # 手腕 → 各指根
            (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
            # Thumb
            (1, 2), (2, 3), (3, 4),
            # Index
            (5, 6), (6, 7), (7, 8),
            # Middle
            (9, 10), (10, 11), (11, 12),
            # Ring
            (13, 14), (14, 15), (15, 16),
            # Pinky
            (17, 18), (18, 19), (19, 20),
        ]
    elif num_joints == 15:
        # SMPL-X 旋转关节（无指尖，无手腕）
        edges = [
            (0, 1), (1, 2),
            (3, 4), (4, 5),
            (6, 7), (7, 8),
            (9, 10), (10, 11),
            (12, 13), (13, 14),
            (0, 3), (3, 6), (6, 9), (9, 12),
        ]
    else:
        # fallback: 线性链
        edges = [(i, i + 1) for i in range(num_joints - 1)]

    adj = torch.zeros(num_joints, num_joints)
    for i, j in edges:
        if i < num_joints and j < num_joints:
            adj[i][j] = 1.0
            adj[j][i] = 1.0
    adj.fill_diagonal_(1.0)  # 自连接
    return adj


def compute_graph_distance(adj):
    """
    Floyd-Warshall 最短路径距离。
    仅在 __init__ 中调用一次，N≤21 时计算量可忽略。
    """
    N = adj.shape[0]
    dist = torch.full((N, N), float(N))
    dist[adj > 0] = 1.0
    dist.fill_diagonal_(0.0)
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist


def build_face_regions(num_joints):
    """
    定义面部关键点的语义区域。

    68 点 (iBUG 标准):
        0-16:  下颌轮廓    17-21: 左眉     22-26: 右眉
        27-35: 鼻子         36-41: 左眼     42-47: 右眼
        48-67: 嘴巴
    """
    if num_joints >= 68:
        regions = {
            'jaw':        list(range(0, 17)),
            'left_brow':  list(range(17, 22)),
            'right_brow': list(range(22, 27)),
            'nose':       list(range(27, 36)),
            'left_eye':   list(range(36, 42)),
            'right_eye':  list(range(42, 48)),
            'mouth':      list(range(48, 68)),
        }
    elif num_joints >= 51:
        regions = {
            'left_brow':  list(range(0, 5)),
            'right_brow': list(range(5, 10)),
            'nose':       list(range(10, 19)),
            'left_eye':   list(range(19, 25)),
            'right_eye':  list(range(25, 31)),
            'mouth':      list(range(31, min(51, num_joints))),
        }
    else:
        regions = {'all': list(range(num_joints))}

    # 收集未被分配到任何区域的关节
    assigned = set()
    for indices in regions.values():
        assigned.update(indices)
    unassigned = [i for i in range(num_joints) if i not in assigned]
    if unassigned:
        regions['other'] = unassigned

    return regions


# ================================================================
#              Part 2: 手部专用注意力模块
# ================================================================

class HandTopologyAttention(nn.Module):
    """
    基于手部骨骼拓扑的自注意力。

    原理:
    - 使用图最短路径距离作为 attention bias (加到 QK^T 分数上)
    - 近邻关节: bias 接近 0 → 注意力权重高
    - 远端关节: bias 为大负数 → 注意力权重低 (softmax 后趋近 0)
    - 每个注意力头有独立的可学习缩放系数
    """

    def __init__(self, d_model, nhead, num_joints, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model

        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # 构建拓扑先验 (只在初始化时计算一次)
        adj = build_hand_adjacency(num_joints)
        graph_dist = compute_graph_distance(adj)
        self.register_buffer('graph_dist', graph_dist)  # (J, J)

        # 每个 head 独立的缩放系数
        # 初始化为正数，配合 -dist 使得近邻 bias 更高
        self.topo_scale = nn.Parameter(torch.ones(nhead, 1, 1) * 0.5)

        self.norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, tgt):
        """
        Args:
            tgt: (B, J, C) 关节 query 特征
        Returns:
            tgt: (B, J, C) 拓扑感知后的特征
        """
        B, J, C = tgt.shape

        # 构造 additive attention bias: (B*nhead, J, J)
        # -dist * |scale|: 确保 scale 为正 → 近邻 bias 高，远端 bias 低
        dist = self.graph_dist[:J, :J]  # 兼容不同关节数
        topo_bias = -dist.unsqueeze(0) * self.topo_scale.abs()     # (nhead, J, J)
        topo_bias = topo_bias.unsqueeze(0).expand(B, -1, -1, -1)  # (B, nhead, J, J)
        topo_bias = topo_bias.reshape(B * self.nhead, J, J)        # (B*nhead, J, J)

        # 拓扑引导的自注意力
        tgt2, _ = self.attn(tgt, tgt, tgt, attn_mask=topo_bias)
        tgt = tgt + self.dropout_layer(tgt2)
        tgt = self.norm(tgt)
        return tgt


class ImplicitOcclusionModule(nn.Module):
    """
    隐式遮挡处理模块 — 后注意力门控。

    ⚠️ 关键修复 (Softmax 陷阱):
        错误做法: 用 confidence 缩放 Key/Value → softmax 后零向量变成均匀权重
        正确做法: 先正常做自注意力 → 再用 confidence 门控残差更新

    原理:
        confidence → 1 (可见):  信任注意力输出，正常更新特征
        confidence → 0 (遮挡):  不信任注意力输出，保留原始特征
    """

    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()

        # 逐关节置信度预测 (无需额外标注，由 loss 自动驱动学习)
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

        # 标准自注意力 (所有关节正常互相看)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, tgt):
        """
        Args:
            tgt: (B, J, C)
        Returns:
            tgt: (B, J, C) 遮挡感知后的特征
        """
        # 预测置信度
        confidence = self.confidence_head(tgt)  # (B, J, 1)

        # 正常自注意力 (Key/Value 不做任何缩放!)
        tgt2, _ = self.self_attn(tgt, tgt, tgt)

        # 后门控: confidence 控制残差更新幅度
        tgt = tgt + self.dropout_layer(tgt2 * confidence)
        tgt = self.norm(tgt)
        return tgt


# ================================================================
#              Part 3: 脸部专用注意力模块
# ================================================================

class FaceRegionAttention(nn.Module):
    """
    面部区域注意力:
    1. Intra-region attention: 同一面部区域内的关键点互相交互
       (如嘴巴内部的 20 个点互相看)
    2. Cross-region attention: 每个关节查询各区域的摘要信息
       (如嘴角从眼睛区域获取上下文 — 微笑时眼睛和嘴巴协同变化)

    不需要额外标注: 区域划分来自解剖学先验 (iBUG 68 点标准)。
    """

    def __init__(self, d_model, nhead, num_joints, dropout=0.1):
        super().__init__()
        self.regions = build_face_regions(num_joints)
        self.num_joints = num_joints

        # ---- Intra-region 注意力 (每个区域独立) ----
        self.region_attns = nn.ModuleDict()
        for name, indices in self.regions.items():
            if len(indices) < 2:
                continue
            # 确保 head 数量合法: d_model % n_heads == 0
            n_h = min(nhead, len(indices))
            while d_model % n_h != 0 and n_h > 1:
                n_h -= 1
            self.region_attns[name] = nn.MultiheadAttention(
                d_model, n_h, dropout=dropout, batch_first=True
            )

        # ---- Cross-region 注意力 ----
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # 可学习的区域嵌入 (帮助区分不同区域的摘要)
        self.region_embed = nn.ParameterDict({
            name: nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            for name in self.regions
        })

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, tgt):
        """
        Args:
            tgt: (B, J, C)
        Returns:
            tgt: (B, J, C)
        """
        B, J, C = tgt.shape

        # ==== 1. Intra-region attention ====
        # 安全做法: 先收集所有区域的更新量，再一次性加上 (避免 in-place 问题)
        all_updates = torch.zeros_like(tgt)
        for name, indices in self.regions.items():
            if name not in self.region_attns:
                continue
            region_feat = tgt[:, indices]  # (B, R_i, C)
            region_out, _ = self.region_attns[name](
                region_feat, region_feat, region_feat
            )
            all_updates[:, indices] = region_out
        tgt = tgt + self.dropout_layer(all_updates)
        tgt = self.norm1(tgt)

        # ==== 2. Cross-region attention ====
        # 为每个区域生成摘要 token
        region_tokens = []
        for name, indices in self.regions.items():
            summary = tgt[:, indices].mean(dim=1, keepdim=True)  # (B, 1, C)
            summary = summary + self.region_embed[name]
            region_tokens.append(summary)
        region_tokens = torch.cat(region_tokens, dim=1)  # (B, num_regions, C)

        # 所有关节查询区域摘要
        tgt2, _ = self.cross_attn(tgt, region_tokens, region_tokens)
        tgt = tgt + self.dropout_layer(tgt2)
        tgt = self.norm2(tgt)

        return tgt


class GlobalExpressionModule(nn.Module):
    """
    全局表情聚合模块。

    动机: 表情是全局性的 — 微笑同时涉及嘴角上扬、眼睛眯起、脸颊隆起。
    单靠局部区域注意力无法捕捉这种全局协同。

    步骤:
    1. 聚合: 可学习的全局 token 从所有关节收集信息
    2. 分发: 每个关节从全局 token 获取整体表情上下文

    不需要额外标注: 全局 token 通过 loss 自动学习聚合有意义的表情信息。
    """

    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.global_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 聚合注意力: global_token ← all joints
        self.aggregate_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # 分发注意力: all joints ← global_token
        self.distribute_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, tgt):
        """
        Args:
            tgt: (B, J, C)
        Returns:
            tgt: (B, J, C) 注入全局表情上下文后的特征
        """
        B = tgt.shape[0]
        global_q = self.global_query.expand(B, -1, -1)  # (B, 1, C)

        # Step 1: 全局 token 聚合所有关节信息
        global_feat, _ = self.aggregate_attn(global_q, tgt, tgt)  # (B, 1, C)

        # Step 2: 每个关节从全局 token 获取表情上下文
        tgt2, _ = self.distribute_attn(tgt, global_feat, global_feat)  # (B, J, C)

        tgt = tgt + self.dropout_layer(tgt2)
        tgt = self.norm(tgt)
        return tgt


# ================================================================
#              Part 4: 解码器层（单层）
# ================================================================

class HandDecoderLayer(nn.Module):
    """
    手部解码器单层:

    ┌──────────────────────────────────────┐
    │  1. HandTopologyAttention            │  骨骼拓扑约束的自注意力
    │  2. ImplicitOcclusionModule          │  遮挡感知的后门控
    │  3. MSDeformAttn (cross-attention)   │  从多尺度特征采样
    │  4. FFN                              │  前馈网络
    └──────────────────────────────────────┘
    """

    def __init__(self, d_model, nhead, num_joints,
                 n_levels, n_points, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        # Sub-layer 1: 拓扑自注意力
        self.topo_attn = HandTopologyAttention(d_model, nhead, num_joints, dropout)

        # Sub-layer 2: 遮挡处理
        self.occlusion = ImplicitOcclusionModule(d_model, nhead, dropout)

        # Sub-layer 3: 可变形交叉注意力
        self.cross_attn = MSDeformAttn(d_model, n_levels, nhead, n_points)
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_dropout = nn.Dropout(dropout)

        # Sub-layer 4: FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, reference_points,
                spatial_shapes, level_start_index):
        """
        Args:
            tgt:                (B, J, C)               query
            memory:             (B, sum(H_l*W_l), C)    多尺度特征
            reference_points:   (B, J, n_levels, 2)     参考点 [0, 1]
            spatial_shapes:     (n_levels, 2)            各层 (H, W)
            level_start_index:  (n_levels,)              各层起始索引
        Returns:
            tgt: (B, J, C)
        """
        # 1. 拓扑感知自注意力
        tgt = self.topo_attn(tgt)

        # 2. 遮挡处理
        tgt = self.occlusion(tgt)

        # 3. 可变形交叉注意力
        tgt2 = self.cross_attn(
            tgt, reference_points, memory,
            spatial_shapes, level_start_index
        )
        tgt = tgt + self.cross_dropout(tgt2)
        tgt = self.cross_norm(tgt)

        # 4. FFN
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.ffn_dropout(tgt2)
        tgt = self.ffn_norm(tgt)

        return tgt


class FaceDecoderLayer(nn.Module):
    """
    脸部解码器单层:

    ┌──────────────────────────────────────┐
    │  1. FaceRegionAttention              │  区域分组自注意力
    │  2. GlobalExpressionModule           │  全局表情聚合
    │  3. MSDeformAttn (cross-attention)   │  从多尺度特征采样
    │  4. FFN                              │  前馈网络
    └──────────────────────────────────────┘
    """

    def __init__(self, d_model, nhead, num_joints,
                 n_levels, n_points, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        # Sub-layer 1: 区域注意力
        self.region_attn = FaceRegionAttention(d_model, nhead, num_joints, dropout)

        # Sub-layer 2: 全局表情
        self.global_expr = GlobalExpressionModule(d_model, nhead, dropout)

        # Sub-layer 3: 可变形交叉注意力
        self.cross_attn = MSDeformAttn(d_model, n_levels, nhead, n_points)
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_dropout = nn.Dropout(dropout)

        # Sub-layer 4: FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, reference_points,
                spatial_shapes, level_start_index):
        # 1. 区域注意力
        tgt = self.region_attn(tgt)

        # 2. 全局表情
        tgt = self.global_expr(tgt)

        # 3. 可变形交叉注意力
        tgt2 = self.cross_attn(
            tgt, reference_points, memory,
            spatial_shapes, level_start_index
        )
        tgt = tgt + self.cross_dropout(tgt2)
        tgt = self.cross_norm(tgt)

        # 4. FFN
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.ffn_dropout(tgt2)
        tgt = self.ffn_norm(tgt)

        return tgt


# ================================================================
#            Part 5: 完整解码器（接口与原始 OSX 完全兼容）
# ================================================================

class HandDecoder(nn.Module):
    """
    手部专用解码器。

    改进 vs 原始 OSX 解码器:
    ┌────────────────────┬──────────────────────────────────────┐
    │ 原始 OSX           │ 本实现                               │
    ├────────────────────┼──────────────────────────────────────┤
    │ 标准 Self-Attention │ HandTopologyAttention (骨骼拓扑约束)  │
    │ 无遮挡处理          │ ImplicitOcclusionModule (后门控)      │
    │ MSDeformAttn       │ MSDeformAttn / FlashDeformAttn (可选) │
    └────────────────────┴──────────────────────────────────────┘

    接口:
        output = decoder(feats, coord_init=..., query_init=...)
        与原始 OSX hand_decoder 完全相同，可直接替换。
    """

    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
        n_levels=3,
        n_points=4,
        num_joints=20,
        feat_channels=256,
    ):
        """
        Args:
            d_model:             decoder 维度 (应与 query_init 维度一致，通常 256)
            nhead:               注意力头数
            num_decoder_layers:  decoder 层数
            dim_feedforward:     FFN 隐藏维度
            dropout:             dropout 率
            n_levels:            多尺度特征层数 (应与 len(hand_feats) 一致)
            n_points:            deformable attention 采样点数
            num_joints:          手部关节数量 (通常 20)
            feat_channels:       输入多尺度特征通道数 (int 或 list)
        """
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels

        # ---- 输入投影: 将各尺度特征映射到 d_model 维度 ----
        if isinstance(feat_channels, int):
            feat_channels = [feat_channels] * n_levels
        assert len(feat_channels) == n_levels

        self.input_proj = nn.ModuleList()
        for fc in feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(fc, d_model, 1),
                    nn.GroupNorm(32, d_model),
                )
            )

        # ---- Level embedding ----
        self.level_embed = nn.Parameter(torch.Tensor(n_levels, d_model))
        nn.init.normal_(self.level_embed)

        # ---- Decoder layers ----
        self.layers = nn.ModuleList([
            HandDecoderLayer(
                d_model, nhead, num_joints,
                n_levels, n_points, dim_feedforward, dropout
            )
            for _ in range(num_decoder_layers)
        ])

        # ---- 每层坐标头 (iterative reference-point refinement, Poseur/Deformable-DETR 式) ----
        # 镜像 PoseurDecoder.bbox_embed (vit.py): 每层从精修特征回归 2D 坐标偏移,
        # 在 logit 空间累加到参考点上再 sigmoid。最后一层 Linear 零初始化 →
        # 第 0 步偏移≈0、首参考点≈coord_init, 贴近暖启的纯特征路径、抑制冷启梯度。
        self.bbox_embed = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
                nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
                nn.Linear(d_model, 2),
            )
            for _ in range(num_decoder_layers)
        ])
        for head in self.bbox_embed:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def _prepare_multi_scale(self, feats):
        """
        将多尺度特征展平为 MSDeformAttn 所需的格式。

        Returns:
            src_flatten:       (B, sum(H_l * W_l), d_model)
            spatial_shapes:    (n_levels, 2)
            level_start_index: (n_levels,)
        """
        src_flatten = []
        spatial_shapes = []
        for lvl in range(self.n_levels):
            # print(f"level {lvl}: {feats[lvl].shape}")
            src = self.input_proj[lvl](feats[lvl])  # (B, d_model, H, W)
            B, C, H, W = src.shape
            spatial_shapes.append((H, W))
            src = src.flatten(2).transpose(1, 2)      # (B, H*W, d_model)
            src = src + self.level_embed[lvl].view(1, 1, -1)
            src_flatten.append(src)

        src_flatten = torch.cat(src_flatten, dim=1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat([
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ])
        return src_flatten, spatial_shapes, level_start_index

    def forward(self, feats, coord_init, query_init):
        """
        Args:
            feats:      list of (B, C, H_l, W_l), len >= n_levels
            coord_init: (B, J, 2) 归一化参考点 ∈ [0, 1]
            query_init: (B, J, d_model) 初始 query 特征

        Returns:
            output:         (B, J, d_model) 精炼后的 query 特征 (喂 regressor, 接口不变)
            outputs_coords: list[L] of (B, J, 2) ∈ [0,1] 每层精修后的参考点 (供 aux 监督)
        """
        assert len(feats) >= self.n_levels, \
            f"feats has {len(feats)} levels, but decoder needs {self.n_levels}"

        # 准备多尺度特征
        memory, spatial_shapes, level_start_index = self._prepare_multi_scale(feats)

        # 迭代参考点: 从 coord_init 起步, 每层回归坐标偏移并在 logit 空间更新 (镜像 PoseurDecoder)
        ref = coord_init.clamp(0.0, 1.0)            # (B, J, 2) ∈ [0,1]
        output = query_init
        outputs_coords = []
        for lid, layer in enumerate(self.layers):
            ref_points_input = ref[:, :, None, :].repeat(1, 1, self.n_levels, 1).clamp(0.0, 1.0)
            output = layer(
                output, memory, ref_points_input,
                spatial_shapes, level_start_index
            )
            # 坐标精修: offset 在 logit 空间累加, 再 sigmoid 回 [0,1]
            tmp = self.bbox_embed[lid](output)
            ref = (tmp + torch.logit(ref.clamp(1e-4, 1.0 - 1e-4))).sigmoid()
            outputs_coords.append(ref)

        return output, outputs_coords


class FaceDecoder(nn.Module):
    """
    脸部专用解码器。

    改进 vs 原始 OSX 解码器:
    ┌────────────────────┬──────────────────────────────────────┐
    │ 原始 OSX           │ 本实现                               │
    ├────────────────────┼──────────────────────────────────────┤
    │ 标准 Self-Attention │ FaceRegionAttention (区域分组注意力)   │
    │ 无全局表情建模       │ GlobalExpressionModule (全局聚合)     │
    │ MSDeformAttn       │ MSDeformAttn / FlashDeformAttn (可选) │
    └────────────────────┴──────────────────────────────────────┘
    """

    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
        n_levels=3,
        n_points=4,
        num_joints=68,
        feat_channels=256,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels

        # 输入投影
        if isinstance(feat_channels, int):
            feat_channels = [feat_channels] * n_levels
        assert len(feat_channels) == n_levels

        self.input_proj = nn.ModuleList()
        for fc in feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(fc, d_model, 1),
                    nn.GroupNorm(32, d_model),
                )
            )

        # Level embedding
        self.level_embed = nn.Parameter(torch.Tensor(n_levels, d_model))
        nn.init.normal_(self.level_embed)

        # Decoder layers
        self.layers = nn.ModuleList([
            FaceDecoderLayer(
                d_model, nhead, num_joints,
                n_levels, n_points, dim_feedforward, dropout
            )
            for _ in range(num_decoder_layers)
        ])

    def _prepare_multi_scale(self, feats):
        """与 HandDecoder 相同的多尺度特征准备。"""
        src_flatten = []
        spatial_shapes = []
        for lvl in range(self.n_levels):
            src = self.input_proj[lvl](feats[lvl])
            B, C, H, W = src.shape
            spatial_shapes.append((H, W))
            src = src.flatten(2).transpose(1, 2)
            src = src + self.level_embed[lvl].view(1, 1, -1)
            src_flatten.append(src)

        src_flatten = torch.cat(src_flatten, dim=1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat([
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ])
        return src_flatten, spatial_shapes, level_start_index

    def forward(self, feats, coord_init, query_init):
        """
        接口与 HandDecoder 完全相同。
        """
        assert len(feats) >= self.n_levels

        memory, spatial_shapes, level_start_index = self._prepare_multi_scale(feats)

        reference_points = coord_init[:, :, None, :].repeat(1, 1, self.n_levels, 1)
        reference_points = reference_points.clamp(0.0, 1.0)

        output = query_init
        for layer in self.layers:
            output = layer(
                output, memory, reference_points,
                spatial_shapes, level_start_index
            )

        return output