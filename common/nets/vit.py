import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import os
import sys
import os.path as osp
from torchvision.models import resnet18, resnet50
cur_dir = osp.dirname(os.path.abspath(__file__))

# 2. 获取项目根目录 (假设 common 文件夹在上一级)
root_dir = osp.join(cur_dir, '..')

# 3. 将根目录加入 Python 搜索路径
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from common.nets.module import HandRotationNet, FaceRegressor, BoxNet, BoxSizeNet, HandRoI, FaceRoI, BodyRotationNet
from common.nets.loss import CoordLoss, ParamLoss, CELoss
from common.utils.human_models import smpl_x
from common.utils.transforms import rot6d_to_axis_angle, restore_bbox
# from config import cfg

# 假设 cfg 在这里全局可用，或者从 main 导入
from config import cfg 



# ==============================================================================
# 1. 纯 PyTorch 实现 Multi-Scale Deformable Attention
# ==============================================================================
class PyTorchMSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0: raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points): grid_init[:, :, i, :] *= i + 1
        with torch.no_grad(): self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None: value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else: raise ValueError(f'Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]}')
        
        output = torch.zeros_like(query)
        value_list = value.split([H * W for H, W in input_spatial_shapes], dim=1)
        sampling_grids = 2 * sampling_locations - 1

        for lid_, (H, W) in enumerate(input_spatial_shapes):
            value_l_ = value_list[lid_].permute(0, 2, 3, 1).reshape(N * self.n_heads, self.d_model // self.n_heads, H, W)
            sampling_grid_l_ = sampling_grids[:, :, :, lid_, :].transpose(1, 2).flatten(0, 1)
            sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)
            attention_weights_l_ = attention_weights[:, :, :, lid_, :].permute(0, 2, 1, 3).reshape(N * self.n_heads, 1, Len_q, self.n_points)
            output += (sampling_value_l_ * attention_weights_l_).sum(-1).view(N, self.n_heads * (self.d_model // self.n_heads), Len_q).transpose(1, 2)
        return self.output_proj(output)

# ==============================================================================
# 2. Poseur Decoder (动态适配输入通道数)
# ==============================================================================
class SinePositionalEncoding(nn.Module):
    def __init__(self, num_feats=128, temperature=10000, normalize=True, scale=2 * math.pi):
        super().__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
    def forward(self, mask):
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
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, n_heads=8, n_levels=4, n_points=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = PyTorchMSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, src_level_start_index):
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.cross_attn(tgt + query_pos, reference_points, src, src_spatial_shapes, src_level_start_index, None)
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout4(tgt2))
        return tgt


class PoseurDecoder(nn.Module):
    def __init__(self, in_channels_list, embed_dim=256, num_heads=8, num_layers=6, 
                 num_points=4, num_queries=72, ffn_dim=1024, use_internal_backbone=True):
        super().__init__()
        
        self.d_model = embed_dim
         # [修复 1] 保存 num_levels，否则 forward 中的 repeat 会报错
        self.num_levels = len(in_channels_list) 
        # [修复 2] 保存 flag
        self.use_internal_backbone = use_internal_backbone 

        self.use_internal_backbone = use_internal_backbone
        
        if self.use_internal_backbone:
            # 初始化 ResNet50 (用于 Decoder 内部提取特征)
            print("Initializing Internal ResNet50 for Decoder...")
            m = resnet50(pretrained=True)
            self.backbone = nn.ModuleDict({
                'conv1': m.conv1, 'bn1': m.bn1, 'relu': m.relu, 'maxpool': m.maxpool,
                'layer1': m.layer1, 'layer2': m.layer2, 'layer3': m.layer3, 'layer4': m.layer4
            })

        # 直接使用 Transformer Layers
        self.layers = nn.ModuleList([
            DeformableTransformerDecoderLayer(embed_dim, ffn_dim, 0.1, num_heads, len(in_channels_list), num_points)
            for _ in range(num_layers)
        ])
        self.pos_embed = SinePositionalEncoding(num_feats=embed_dim//2, normalize=True)
        self.query_embed = nn.Embedding(num_queries, embed_dim * 2)
        
        self.input_proj = nn.ModuleList()
        for in_ch in in_channels_list:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, embed_dim, kernel_size=1),
                    nn.GroupNorm(32, embed_dim)
                )
            )
        
        self.bbox_embed = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                nn.Linear(embed_dim, 2)
            ) for _ in range(num_layers)
        ])

    def forward(self, x, coord_init=None, query_init=None):
        bs = 0
        features = []

        # 1. Internal Backbone Forward
        if self.use_internal_backbone:
            bs = x.shape[0]
            x = self.backbone['conv1'](x)
            x = self.backbone['bn1'](x)
            x = self.backbone['relu'](x)
            x = self.backbone['maxpool'](x)
            
            c1 = self.backbone['layer1'](x)
            c2 = self.backbone['layer2'](c1)
            c3 = self.backbone['layer3'](c2)
            if self.num_levels > 3:
                c4 = self.backbone['layer4'](c3)
                features = [c1, c2, c3, c4]
            else:
                features = [c1, c2, c3]
        else:
            features = x
            bs = features[0].shape[0]

        # 2. Multi-scale Feature Preparation
        srcs, masks, pos_embeds = [], [], []
        for l, src in enumerate(features):
            src = self.input_proj[l](src)
            mask = torch.zeros((bs, src.shape[2], src.shape[3]), dtype=torch.bool, device=src.device)
            pos = self.pos_embed(mask)
            srcs.append(src)
            masks.append(mask)
            pos_embeds.append(pos)
            
        src_flatten = torch.cat([s.flatten(2).transpose(1, 2) for s in srcs], 1)
        spatial_shapes = torch.as_tensor([(s.shape[2], s.shape[3]) for s in srcs], dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        
        # 3. Query & Reference Points Initialization
        if query_init is not None:
            tgt = query_init
            query_pos = torch.zeros_like(tgt)
        else:
            query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
            tgt = torch.zeros_like(query_embed[..., :self.d_model])
            query_pos = query_embed[..., :self.d_model]

        if coord_init is not None:
            reference_points = coord_init
            # [关键修复] 如果传入的是 3D 坐标 (x,y,z)，强制切片为 2D (x,y)
            if reference_points.shape[-1] == 3:
                reference_points = reference_points[..., :2]
        else:
            reference_points = torch.sigmoid(self.bbox_embed[0](query_pos))

        # 4. Decoder Layers Loop
        outputs_coords = []
        for lid, layer in enumerate(self.layers):
            ref_points_input = reference_points[:, :, None, :].repeat(1, 1, self.num_levels, 1)
            
            # Forward
            tgt = layer(tgt, query_pos, ref_points_input, src_flatten, spatial_shapes, level_start_index)
            
            # BBox Refinement
            tmp = self.bbox_embed[lid](tgt)
            tmp = tmp + torch.logit(reference_points.clamp(1e-4, 1-1e-4))
            reference_points = tmp.sigmoid()
            
            outputs_coords.append(reference_points)

        return tgt, outputs_coords

# ==============================================================================
# 3. ViT Backbone
# ==============================================================================
# ------------------------------------------------------------------------------
# 兼容权重的 StandardViT (替代之前的 ViTBackbone)
# ------------------------------------------------------------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 关键：这里使用 qkv 线性层，而不是 in_proj_weight
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # 兼容 timm 的 qkv 计算方式
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        # eps=1e-6 to match OSX ViT (norm_layer = partial(nn.LayerNorm, eps=1e-6));
        # the nn.LayerNorm default is 1e-5.
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed(nn.Module):
    """ 适配 timm/mmpose 的 PatchEmbed 结构 (包含 .proj) """
    def __init__(self, img_size=(256, 192), patch_size=16, in_chans=3, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # padding=2 matches OSX's ViT PatchEmbed: transformer_utils ViT uses
        # Conv2d(..., padding = 4 + 2*(ratio//2 - 1)) which is 2 at ratio=1
        # (the osx_l config). The previous default (padding=0) produced the SAME
        # (16,12) grid but convolved every patch over a 2px-shifted, unpadded
        # window -> ~24% img_feat divergence vs the MMCV ViT that the frozen OSX
        # body/box/ROI weights expect. Shapes matched either way, so it stayed
        # invisible until the encoder-fidelity probe (tool/analysis/encoder_compare).
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=2)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class StandardViT(nn.Module):
    def __init__(self, img_size=(256, 192), patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        
        # 设定任务 Token 数量
        self.num_task_token = 31
        self.task_tokens = nn.Parameter(torch.zeros(1, self.num_task_token, embed_dim))

        # pos_embed 复刻 OSX ViT 的布局: (1, num_patches + 1, C)。多出来的那一槽
        # ([:, :1]) 是加到每个 patch 上的全局 "cls" pos；task token 不加任何 pos
        # (见 forward)。这样冻结的 OSX 预训练 body/box 网络才能拿到它期望的特征。
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)  # match OSX ViT last_norm (eps=1e-6)
        self.grid_size = self.patch_embed.grid_size

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.task_tokens, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, 192, 1024]

        # OSX ViT 位置编码: patch 加 patch_pos + 单个全局 cls_pos；task token 在其后
        # 拼接、不加任何 pos_embed。复刻 transformer_utils ViT.forward_features。
        x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        task_tokens = self.task_tokens.expand(B, -1, -1)  # [B, 31, 1024]
        x = torch.cat((task_tokens, x), dim=1)  # [B, 31+192, 1024]

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # [核心修复] 拆分 Task Tokens 和 Image Features
        # task_tokens: [B, 31, 1024]
        task_token_out = x[:, :self.num_task_token, :]
        
        # img_features: [B, 192, 1024]
        img_feat_out = x[:, self.num_task_token:, :]
        
        # Reshape Image Features: [B, 1024, H, W]
        img_feat_out = img_feat_out.transpose(1, 2).reshape(B, self.embed_dim, *self.grid_size)
        
        # 返回两个值
        return img_feat_out, task_token_out

class ViTWrapper:
    def __init__(self, backbone): self.backbone = backbone
    def load_state_dict(self, sd, strict=False):
        self.backbone.load_state_dict({k.replace('backbone.', ''): v for k, v in sd.items()}, strict=strict)