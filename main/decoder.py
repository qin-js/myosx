import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 引入之前定义的 PyTorchMSDeformAttn
from MSDA import PyTorchMSDeformAttn 

# ==========================================================
# 1. 辅助模块：位置编码与MLP
# ==========================================================

class SinePositionalEncoding(nn.Module):
    """ 对应 config: positional_encoding=dict(type='SinePositionalEncoding', ...) """
    def __init__(self, num_feats=128, temperature=10000, normalize=True, scale=2 * math.pi):
        super().__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, mask):
        # mask: [B, H, W], 0 for padding, 1 for valid (or opposite, depending on convention)
        # 这里假设 mask 是 Bool Tensor, True 表示 valid/keep
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos # [B, C, H, W]

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

# ==========================================================
# 2. Deformable Transformer Decoder Layer (纯 PyTorch)
# ==========================================================

class DeformableTransformerDecoderLayer(nn.Module):
    """
    对应 config: transformerlayers
    Self-Attn -> Norm -> Cross-Attn (MSDeform) -> Norm -> FFN -> Norm
    """
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, n_heads=8, n_levels=4, n_points=4):
        super().__init__()
        
        # 1. Self Attention (Standard Multihead)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 2. Cross Attention (MultiScale Deformable)
        # 对应 config: type='MultiScaleDeformableAttention_post_value'
        self.cross_attn = PyTorchMSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # 3. FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, src_level_start_index, src_padding_mask=None):
        # tgt: [B, N_q, C] (Query embeddings)
        # query_pos: [B, N_q, C] (Query position embeddings)
        # src: [B, Len_in, C] (Flattened multi-scale features)
        
        # --- Self Attention ---
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        # --- Cross Attention (Deformable) ---
        tgt2 = self.cross_attn(
            query=tgt + query_pos, 
            reference_points=reference_points,
            input_flatten=src, 
            input_spatial_shapes=src_spatial_shapes, 
            input_level_start_index=src_level_start_index, 
            input_padding_mask=src_padding_mask
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # --- FFN ---
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout4(tgt2))
        
        return tgt

# ==========================================================
# 3. Poseur Hand Decoder (主封装类)
# ==========================================================

class PoseurHandDecoder(nn.Module):
    """
    纯 PyTorch 实现 Config 中的 Hand Decoder 部分
    替代 mmcv build_posenet(decoder_cfg.model)
    """
    def __init__(self, 
                 embed_dim=256, 
                 num_heads=8, 
                 num_layers=6, 
                 num_levels=4, 
                 num_points=4, 
                 num_queries=42, # config['num_output_channels']
                 ffn_dim=1024):
        super().__init__()
        
        self.d_model = embed_dim
        self.n_levels = num_levels
        
        # 1. Decoder Layers
        self.layers = nn.ModuleList([
            DeformableTransformerDecoderLayer(
                d_model=embed_dim, 
                d_ffn=ffn_dim, 
                dropout=0.1, 
                n_heads=num_heads, 
                n_levels=num_levels, 
                n_points=num_points
            ) for _ in range(num_layers)
        ])
        
        # 2. Positional Encoding (用于特征图)
        self.pos_embed = SinePositionalEncoding(num_feats=embed_dim // 2, normalize=True)
        
        # 3. Query Embeddings (Learnable)
        self.query_embed = nn.Embedding(num_queries, embed_dim * 2) # *2 for (tgt + pos)
        
        # 4. Input Projections (ResNet/ViT features -> embed_dim)
        # 假设输入特征通道不一定是 embed_dim，需要投影。
        # 这里假设 Multi-scale features 输入，需要 num_levels 个 Conv
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1), # 假设 RoI Net 输出已经是 embed_dim
                nn.GroupNorm(32, embed_dim),
            ) for _ in range(num_levels)
        ])

        # 5. Prediction Heads (Regression)
        # 对应 num_reg_fcs=2
        self.class_embed = nn.Linear(embed_dim, 1) # 这里可能不需要分类，如果是纯回归
        self.bbox_embed = MLP(embed_dim, embed_dim, 2, 3) # 回归 xy 坐标

        # Initialize
        self._reset_parameters()

    def _reset_parameters(self):
        # 模仿 Deformable DETR 初始化
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, features):
        """
        features: List of tensors [B, C, H, W] from RoI Net (Multi-scale)
        """
        bs = features[0].shape[0]
        
        # --- Prepare Inputs for Deformable Attention ---
        srcs = []
        masks = []
        pos_embeds = []
        
        for l, src in enumerate(features):
            # Projection
            src = self.input_proj[l](src)
            
            # Mask (全图有效，设为 False)
            B, C, H, W = src.shape
            mask = torch.zeros((B, H, W), dtype=torch.bool, device=src.device)
            
            # Pos Embed
            pos = self.pos_embed(mask)
            
            srcs.append(src)
            masks.append(mask)
            pos_embeds.append(pos)

        # Flatten Setup
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        
        for l in range(len(srcs)):
            bs, c, h, w = srcs[l].shape
            spatial_shapes.append((h, w))
            src_flatten.append(srcs[l].flatten(2).transpose(1, 2)) # [B, HW, C]
            mask_flatten.append(masks[l].flatten(1))
            lvl_pos_embed_flatten.append(pos_embeds[l].flatten(2).transpose(1, 2))

        src_flatten = torch.cat(src_flatten, 1) # [B, sum(HW), C]
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # --- Prepare Queries ---
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1) # [B, Nq, 2C]
        tgt = torch.zeros_like(query_embed[..., :self.d_model]) # Initial Query content
        query_pos = query_embed[..., :self.d_model] # Pos embedding for query
        # Config 中 query_pose_emb=True, 这里简化，通常 query_embed 包含了 pos

        # Reference Points (Init)
        # Sigmoid(query_pos) as initial reference
        reference_points = torch.sigmoid(self.bbox_embed(query_pos)) # [B, Nq, 2]

        # --- Decoder Loop ---
        outputs_coords = []
        
        # 这里的 tgt 其实应该从 query_embed 拆分，或者全零初始化
        tgt = query_embed[..., self.d_model:] 
        query_pos = query_embed[..., :self.d_model]

        for layer in self.layers:
            # 传入的 reference_points 需要调整形状以适配 function
            # [B, Nq, 2] -> [B, Nq, n_levels, 2]
            ref_points_input = reference_points[:, :, None, :] * valid_ratios[:, None]
            
            tgt = layer(tgt, query_pos, ref_points_input, src_flatten, spatial_shapes, level_start_index, None)

            # Box Refine (Poseur config: with_box_refine=True)
            tmp = self.bbox_embed(tgt)
            tmp = tmp + torch.inverse_sigmoid(reference_points)
            reference_points = tmp.sigmoid()
            outputs_coords.append(reference_points)

        return torch.stack(outputs_coords) # [Layers, B, Nq, 2]