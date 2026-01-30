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

from common.nets.module import PositionNet, HandRotationNet, FaceRegressor, BoxNet, BoxSizeNet, HandRoI, FaceRoI, BodyRotationNet
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

    def forward(self, x):
        # x: [B, 3, H, W] (如果用了 internal backbone，这里应该是图片/RoI Crop)
        # 或者 x: List[Tensor] (如果没用 internal backbone)
        
        bs = 0
        features = []

        if self.use_internal_backbone:
            # 假设 x 是 tensor [B, 3, H, W]
            bs = x.shape[0]
            # ResNet Forward
            x = self.backbone['conv1'](x)
            x = self.backbone['bn1'](x)
            x = self.backbone['relu'](x)
            x = self.backbone['maxpool'](x)
            
            c1 = self.backbone['layer1'](x)
            c2 = self.backbone['layer2'](c1)
            c3 = self.backbone['layer3'](c2)
            c4 = self.backbone['layer4'](c3)
            
            features = [c1, c2, c3, c4]
        else:
            features = x
            bs = features[0].shape[0]

        # ... (后续 Transformer 逻辑保持不变) ...
        # src_flatten, spatial_shapes, attention ... 
        # 直接复制之前的 forward 逻辑即可，注意变量名 features
        
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
        
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        tgt = torch.zeros_like(query_embed[..., :self.d_model])
        query_pos = query_embed[..., :self.d_model]
        reference_points = torch.sigmoid(self.bbox_embed[0](query_pos))
        
        outputs_coords = []
        for lid, layer in enumerate(self.layers):
            ref_points_input = reference_points[:, :, None, :].repeat(1, 1, self.num_levels, 1)
            tgt = layer(tgt, query_pos, ref_points_input, src_flatten, spatial_shapes, level_start_index)
            tmp = self.bbox_embed[lid](tgt)
            tmp = tmp + torch.inverse_sigmoid(reference_points)
            reference_points = tmp.sigmoid()
            outputs_coords.append(reference_points)
        return torch.stack(outputs_coords)

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
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
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
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class StandardViT(nn.Module):
    def __init__(self, img_size=(256, 192), patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        # 修改点：使用 PatchEmbed 类
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_task_token = 24 
        self.task_tokens = nn.Parameter(torch.zeros(1, self.num_task_token, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
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
        x = self.patch_embed(x)
        # cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = self.task_tokens.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # 修改 4: 只需要 Feature Map，丢弃 Token
        # (假设我们只需要特征图)
        x = x[:, self.num_task_token:, :] 
        
        return x[:, 1:, :].transpose(1, 2).reshape(B, self.embed_dim, *self.grid_size)

class ViTWrapper:
    def __init__(self, backbone): self.backbone = backbone
    def load_state_dict(self, sd, strict=False):
        self.backbone.load_state_dict({k.replace('backbone.', ''): v for k, v in sd.items()}, strict=strict)


class Model(nn.Module):
    def __init__(self, encoder, body_position_net, body_rotation_net, box_net, hand_position_net, hand_roi_net, hand_decoder,
                 hand_rotation_net, face_position_net, face_roi_net, face_decoder, face_regressor):
        super(Model, self).__init__()
        # body
        self.encoder = encoder
        self.body_position_net = body_position_net
        self.body_regressor = body_rotation_net
        self.box_net = box_net

        # hand
        self.hand_roi_net = hand_roi_net
        self.hand_position_net = hand_position_net
        self.hand_decoder = hand_decoder
        self.hand_regressor = hand_rotation_net

        # face
        self.face_roi_net = face_roi_net
        self.face_position_net = face_position_net
        self.face_decoder = face_decoder
        self.face_regressor = face_regressor

        self.smplx_layer = copy.deepcopy(smpl_x.layer['neutral']).cuda()

        self.coord_loss = CoordLoss()
        self.param_loss = ParamLoss()
        self.ce_loss = CELoss()

        self.body_num_joints = len(smpl_x.pos_joint_part['body'])
        self.hand_num_joints = len(smpl_x.pos_joint_part['rhand'])

        self.trainable_modules = [self.encoder, self.body_position_net, self.body_regressor,
                                  self.box_net, self.hand_position_net, self.hand_roi_net, self.hand_regressor,
                                  self.face_regressor, self.face_roi_net, self.face_position_net]
        self.special_trainable_modules = [self.hand_decoder, self.face_decoder]

    def get_camera_trans(self, cam_param):
        # camera translation
        t_xy = cam_param[:, :2]
        gamma = torch.sigmoid(cam_param[:, 2])  # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(cfg.focal[0] * cfg.focal[1] * cfg.camera_3d_size * cfg.camera_3d_size / (
                cfg.input_body_shape[0] * cfg.input_body_shape[1]))]).cuda().view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:, None]), 1)
        return cam_trans

    def get_coord(self, root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_trans, mode):
        batch_size = root_pose.shape[0]
        zero_pose = torch.zeros((1, 3)).float().cuda().repeat(batch_size, 1)  # eye poses
        output = self.smplx_layer(betas=shape, body_pose=body_pose, global_orient=root_pose, right_hand_pose=rhand_pose,
                                  left_hand_pose=lhand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose,
                                  reye_pose=zero_pose, expression=expr)
        # camera-centered 3D coordinate
        mesh_cam = output.vertices
        if mode == 'test' and cfg.testset == 'AGORA':  # use 144 joints for AGORA evaluation
            joint_cam = output.joints
        else:
            joint_cam = output.joints[:, smpl_x.joint_idx, :]

        # project 3D coordinates to 2D space
        if mode == 'train' and len(cfg.trainset_3d) == 1 and cfg.trainset_3d[0] == 'AGORA' and len(
                cfg.trainset_2d) == 0:  # prevent gradients from backpropagating to SMPLX parameter regression module
            x = (joint_cam[:, :, 0].detach() + cam_trans[:, None, 0]) / (
                    joint_cam[:, :, 2].detach() + cam_trans[:, None, 2] + 1e-4) * cfg.focal[0] + cfg.princpt[0]
            y = (joint_cam[:, :, 1].detach() + cam_trans[:, None, 1]) / (
                    joint_cam[:, :, 2].detach() + cam_trans[:, None, 2] + 1e-4) * cfg.focal[1] + cfg.princpt[1]
        else:
            x = (joint_cam[:, :, 0] + cam_trans[:, None, 0]) / (joint_cam[:, :, 2] + cam_trans[:, None, 2] + 1e-4) * \
                cfg.focal[0] + cfg.princpt[0]
            y = (joint_cam[:, :, 1] + cam_trans[:, None, 1]) / (joint_cam[:, :, 2] + cam_trans[:, None, 2] + 1e-4) * \
                cfg.focal[1] + cfg.princpt[1]
        x = x / cfg.input_body_shape[1] * cfg.output_hm_shape[2]
        y = y / cfg.input_body_shape[0] * cfg.output_hm_shape[1]
        joint_proj = torch.stack((x, y), 2)

        # root-relative 3D coordinates
        root_cam = joint_cam[:, smpl_x.root_joint_idx, None, :]
        joint_cam = joint_cam - root_cam
        mesh_cam = mesh_cam + cam_trans[:, None, :]  # for rendering

        # left hand root (left wrist)-relative 3D coordinates
        lhand_idx = smpl_x.joint_part['lhand']
        lhand_cam = joint_cam[:, lhand_idx, :]
        lwrist_cam = joint_cam[:, smpl_x.lwrist_idx, None, :]
        lhand_cam = lhand_cam - lwrist_cam
        joint_cam = torch.cat((joint_cam[:, :lhand_idx[0], :], lhand_cam, joint_cam[:, lhand_idx[-1] + 1:, :]), 1)

        # right hand root (right wrist)-relative 3D coordinates
        rhand_idx = smpl_x.joint_part['rhand']
        rhand_cam = joint_cam[:, rhand_idx, :]
        rwrist_cam = joint_cam[:, smpl_x.rwrist_idx, None, :]
        rhand_cam = rhand_cam - rwrist_cam
        joint_cam = torch.cat((joint_cam[:, :rhand_idx[0], :], rhand_cam, joint_cam[:, rhand_idx[-1] + 1:, :]), 1)

        # face root (neck)-relative 3D coordinates
        face_idx = smpl_x.joint_part['face']
        face_cam = joint_cam[:, face_idx, :]
        neck_cam = joint_cam[:, smpl_x.neck_idx, None, :]
        face_cam = face_cam - neck_cam
        joint_cam = torch.cat((joint_cam[:, :face_idx[0], :], face_cam, joint_cam[:, face_idx[-1] + 1:, :]), 1)

        return joint_proj, joint_cam, mesh_cam

    def generate_mesh_gt(self, targets, mode):
        if 'smplx_mesh_cam' in targets:
            return targets['smplx_mesh_cam']
        nums = [3, 63, 45, 45, 3]
        accu = []
        temp = 0
        for num in nums:
            temp += num
            accu.append(temp)
        pose = targets['smplx_pose']
        root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose = \
            pose[:, :accu[0]], pose[:, accu[0]:accu[1]], pose[:, accu[1]:accu[2]], pose[:, accu[2]:accu[3]], pose[:,accu[3]:accu[4]]
        shape = targets['smplx_shape']
        expr = targets['smplx_expr']
        cam_trans = targets['smplx_cam_trans']

        # final output
        joint_proj, joint_cam, mesh_cam = self.get_coord(root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape,
                                                         expr, cam_trans, mode)

        return mesh_cam

    def norm2heatmap(self, input, hm_shape):
        assert input.shape[-1] in [2, 3, 4]
        if input.shape[-1] == 2:
            x, y = input[..., 0], input[..., 1]
            x = x * hm_shape[2]
            y = y * hm_shape[1]
            output = torch.stack((x, y), dim=-1)
        elif input.shape[-1] == 3:
            x, y, z = input[..., 0], input[..., 1], input[..., 2]
            x = x * hm_shape[2]
            y = y * hm_shape[1]
            z = z * hm_shape[0]
            output = torch.stack((x, y, z), dim=-1)
        elif input.shape[-1] == 4:
            x, y, w, h = input[..., 0], input[..., 1], input[..., 2], input[..., 3]
            x = x * hm_shape[2]
            y = y * hm_shape[1]
            w = w * hm_shape[2]
            h = h * hm_shape[1]
            output = torch.stack((x, y, w, h), dim=-1)
        return output

    def heatmap2norm(self, input, hm_shape):
        assert input.shape[-1] in [2, 3, 4]
        if input.shape[-1] == 2:
            x, y = input[..., 0], input[..., 1]
            x = x / hm_shape[2]
            y = y / hm_shape[1]
            output = torch.stack((x, y), dim=-1)
        elif input.shape[-1] == 3:
            x, y, z = input[..., 0], input[..., 1], input[..., 2]
            x = x / hm_shape[2]
            y = y / hm_shape[1]
            z = z / hm_shape[0]
            output = torch.stack((x, y, z), dim=-1)
        elif input.shape[-1] == 4:
            x, y, w, h = input[..., 0], input[..., 1], input[..., 2], input[..., 3]
            x = x / hm_shape[2]
            y = y / hm_shape[1]
            w = w / hm_shape[2]
            h = h / hm_shape[1]
            output = torch.stack((x, y, w, h), dim=-1)

        return output

    def bbox_split(self, bbox):
        # bbox:[bs, 3, 3]
        lhand_bbox_center, rhand_bbox_center, face_bbox_center = \
            bbox[:, 0, :2], bbox[:, 1, :2], bbox[:, 2, :2]
        return lhand_bbox_center, rhand_bbox_center, face_bbox_center

    def forward(self, inputs, targets, meta_info, mode):

        body_img = F.interpolate(inputs['img'], cfg.input_body_shape)
        print(body_img.shape)

        # 1. Encoder
        img_feat, task_tokens = self.encoder(body_img)  # task_token:[bs, N, c]
        print(img_feat.shape)
        shape_token, cam_token, expr_token, jaw_pose_token, hand_token, body_pose_token = \
            task_tokens[:, 0], task_tokens[:, 1], task_tokens[:, 2], task_tokens[:, 3], task_tokens[:, 4:6], task_tokens[:, 6:]

        # 2. Body Regressor
        body_joint_hm, body_joint_img = self.body_position_net(img_feat)
        root_pose, body_pose, shape, cam_param, = self.body_regressor(body_pose_token, shape_token, cam_token, body_joint_img.detach())
        root_pose = rot6d_to_axis_angle(root_pose)
        body_pose = rot6d_to_axis_angle(body_pose.reshape(-1, 6)).reshape(body_pose.shape[0], -1)  # (N, J_R*3)
        cam_trans = self.get_camera_trans(cam_param)

        # 3. Hand and Face BBox Estimation
        lhand_bbox_center, lhand_bbox_size, rhand_bbox_center, rhand_bbox_size, face_bbox_center, face_bbox_size = self.box_net(img_feat, body_joint_hm.detach())
        lhand_bbox = restore_bbox(lhand_bbox_center, lhand_bbox_size, cfg.input_hand_shape[1] / cfg.input_hand_shape[0], 2.0).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        rhand_bbox = restore_bbox(rhand_bbox_center, rhand_bbox_size, cfg.input_hand_shape[1] / cfg.input_hand_shape[0], 2.0).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        face_bbox = restore_bbox(face_bbox_center, face_bbox_size, cfg.input_face_shape[1] / cfg.input_face_shape[0], 1.5).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space

        # 4. Differentiable Feature-level Hand/Face Crop-Upsample
        # hand_feat: list, [bsx2, c, cfg.output_hand_hm_shape[1]*scale, cfg.output_hand_hm_shape[2]*scale]
        hand_feats = self.hand_roi_net(img_feat, lhand_bbox, rhand_bbox)  # list, hand_feat: flipped left hand + right hand
        # face_feat: list, [bs, c, cfg.output_face_hm_shape[1]*scale, cfg.output_face_hm_shape[2]*scale]
        face_feats = self.face_roi_net(img_feat, face_bbox)

        # 4. keypoint-guided deformable decoder
        # hand keypoint-guided deformable decoder
        _, hand_joint_img, hand_img_feat_joints = self.hand_position_net(hand_feats[-2])  # (2N, J_P, 3) in (hand_hm_shape[2], hand_hm_shape[1], hand_hm_shape[0]) space
        # [-2]: scale=2, because the roi size = (hand_hm_shape*scale//2)
        hand_coord_init = self.heatmap2norm(hand_joint_img, cfg.output_hand_hm_shape)
        hand_img_feat_joints = self.hand_decoder(hand_feats, coord_init=hand_coord_init.detach(), query_init=hand_img_feat_joints)
        # hand regression head
        hand_pose = self.hand_regressor(hand_img_feat_joints, hand_joint_img.detach())
        hand_pose = rot6d_to_axis_angle(hand_pose.reshape(-1, 6)).reshape(hand_img_feat_joints.shape[0], -1)  # (2N, J_R*3)
        # restore flipped left hand joint coordinates
        batch_size = hand_joint_img.shape[0] // 2
        lhand_joint_img = hand_joint_img[:batch_size, :, :]
        lhand_joint_img = torch.cat(
            (cfg.output_hand_hm_shape[2] - 1 - lhand_joint_img[:, :, 0:1], lhand_joint_img[:, :, 1:]), 2)
        rhand_joint_img = hand_joint_img[batch_size:, :, :]
        # restore flipped left hand joint rotations
        batch_size = hand_pose.shape[0] // 2
        lhand_pose = hand_pose[:batch_size, :].reshape(-1, len(smpl_x.orig_joint_part['lhand']), 3)
        lhand_pose = torch.cat((lhand_pose[:, :, 0:1], -lhand_pose[:, :, 1:3]), 2).view(batch_size, -1)
        rhand_pose = hand_pose[batch_size:, :]

        # face keypoint-guided deformable decoder
        _, face_joint_img, face_img_feat_joints = self.face_position_net(face_feats[-2])  # (N, J_P, 3) in (face_hm_shape[2], face_hm_shape[1], face_hm_shape[0]) space
        face_coord_init = self.heatmap2norm(face_joint_img, cfg.output_face_hm_shape)
        face_img_feat_joints = self.face_decoder(face_feats, coord_init=face_coord_init.detach(), query_init=face_img_feat_joints)
        # face regression head
        expr, jaw_pose = self.face_regressor(face_img_feat_joints, face_joint_img.detach(), face_feats[-1])
        jaw_pose = rot6d_to_axis_angle(jaw_pose)

        # final output
        joint_proj, joint_cam, mesh_cam = self.get_coord(root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_trans, mode)
        pose = torch.cat((root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose), 1)
        joint_img = torch.cat((body_joint_img, lhand_joint_img, rhand_joint_img), 1)

        if mode == 'test' and 'smplx_pose' in targets:
            mesh_pseudo_gt = self.generate_mesh_gt(targets, mode)

        if mode == 'train':
            # loss functions
            loss = {}
            loss['smplx_pose'] = self.param_loss(pose, targets['smplx_pose'], meta_info['smplx_pose_valid'])
            loss['smplx_shape'] = self.param_loss(shape, targets['smplx_shape'], meta_info['smplx_shape_valid'][:, None]) * cfg.smplx_loss_weight
            loss['smplx_expr'] = self.param_loss(expr, targets['smplx_expr'], meta_info['smplx_expr_valid'][:, None])
            loss['joint_cam'] = self.coord_loss(joint_cam, targets['joint_cam'], meta_info['joint_valid'] * meta_info['is_3D'][:, None, None])
            loss['smplx_joint_cam'] = self.coord_loss(joint_cam, targets['smplx_joint_cam'], meta_info['smplx_joint_valid'])
            loss['lhand_bbox'] = (self.coord_loss(lhand_bbox_center, targets['lhand_bbox_center'], meta_info['lhand_bbox_valid'][:, None]) +
                                  self.coord_loss(lhand_bbox_size, targets['lhand_bbox_size'], meta_info['lhand_bbox_valid'][:, None]))
            loss['rhand_bbox'] = (self.coord_loss(rhand_bbox_center, targets['rhand_bbox_center'], meta_info['rhand_bbox_valid'][:, None]) +
                                  self.coord_loss(rhand_bbox_size, targets['rhand_bbox_size'], meta_info['rhand_bbox_valid'][:, None]))
            loss['face_bbox'] = (self.coord_loss(face_bbox_center, targets['face_bbox_center'], meta_info['face_bbox_valid'][:, None]) +
                                 self.coord_loss(face_bbox_size, targets['face_bbox_size'], meta_info['face_bbox_valid'][:, None]))
            # change hand target joint_img and joint_trunc according to hand bbox (cfg.output_hm_shape -> downsampled hand bbox space)
            for part_name, bbox in (('lhand', lhand_bbox), ('rhand', rhand_bbox)):
                for coord_name, trunc_name in (('joint_img', 'joint_trunc'), ('smplx_joint_img', 'smplx_joint_trunc')):
                    x = targets[coord_name][:, smpl_x.joint_part[part_name], 0]
                    y = targets[coord_name][:, smpl_x.joint_part[part_name], 1]
                    z = targets[coord_name][:, smpl_x.joint_part[part_name], 2]
                    trunc = meta_info[trunc_name][:, smpl_x.joint_part[part_name], 0]

                    x -= (bbox[:, None, 0] / cfg.input_body_shape[1] * cfg.output_hm_shape[2])
                    x *= (cfg.output_hand_hm_shape[2] / (
                            (bbox[:, None, 2] - bbox[:, None, 0]) / cfg.input_body_shape[1] * cfg.output_hm_shape[
                        2]))
                    y -= (bbox[:, None, 1] / cfg.input_body_shape[0] * cfg.output_hm_shape[1])
                    y *= (cfg.output_hand_hm_shape[1] / (
                            (bbox[:, None, 3] - bbox[:, None, 1]) / cfg.input_body_shape[0] * cfg.output_hm_shape[
                        1]))
                    z *= cfg.output_hand_hm_shape[0] / cfg.output_hm_shape[0]
                    trunc *= ((x >= 0) * (x < cfg.output_hand_hm_shape[2]) * (y >= 0) * (
                            y < cfg.output_hand_hm_shape[1]))

                    coord = torch.stack((x, y, z), 2)
                    trunc = trunc[:, :, None]
                    targets[coord_name] = torch.cat((targets[coord_name][:, :smpl_x.joint_part[part_name][0], :], coord,
                                                     targets[coord_name][:, smpl_x.joint_part[part_name][-1] + 1:, :]),
                                                    1)
                    meta_info[trunc_name] = torch.cat((meta_info[trunc_name][:, :smpl_x.joint_part[part_name][0], :],
                                                       trunc,
                                                       meta_info[trunc_name][:, smpl_x.joint_part[part_name][-1] + 1:,
                                                       :]), 1)

            # change hand projected joint coordinates according to hand bbox (cfg.output_hm_shape -> hand bbox space)
            for part_name, bbox in (('lhand', lhand_bbox), ('rhand', rhand_bbox)):
                x = joint_proj[:, smpl_x.joint_part[part_name], 0]
                y = joint_proj[:, smpl_x.joint_part[part_name], 1]

                x -= (bbox[:, None, 0] / cfg.input_body_shape[1] * cfg.output_hm_shape[2])
                x *= (cfg.output_hand_hm_shape[2] / (
                        (bbox[:, None, 2] - bbox[:, None, 0]) / cfg.input_body_shape[1] * cfg.output_hm_shape[2]))
                y -= (bbox[:, None, 1] / cfg.input_body_shape[0] * cfg.output_hm_shape[1])
                y *= (cfg.output_hand_hm_shape[1] / (
                        (bbox[:, None, 3] - bbox[:, None, 1]) / cfg.input_body_shape[0] * cfg.output_hm_shape[1]))

                coord = torch.stack((x, y), 2)
                trans = []
                for bid in range(coord.shape[0]):
                    mask = meta_info['joint_trunc'][bid, smpl_x.joint_part[part_name], 0] == 1
                    if torch.sum(mask) == 0:
                        trans.append(torch.zeros((2)).float().cuda())
                    else:
                        trans.append((-coord[bid, mask, :2] + targets['joint_img'][:, smpl_x.joint_part[part_name], :][
                                                              bid, mask, :2]).mean(0))
                trans = torch.stack(trans)[:, None, :]
                coord = coord + trans  # global translation alignment
                joint_proj = torch.cat((joint_proj[:, :smpl_x.joint_part[part_name][0], :], coord,
                                        joint_proj[:, smpl_x.joint_part[part_name][-1] + 1:, :]), 1)

            # change face projected joint coordinates according to face bbox (cfg.output_hm_shape -> face bbox space)
            coord = joint_proj[:, smpl_x.joint_part['face'], :]
            trans = []
            for bid in range(coord.shape[0]):
                mask = meta_info['joint_trunc'][bid, smpl_x.joint_part['face'], 0] == 1
                if torch.sum(mask) == 0:
                    trans.append(torch.zeros((2)).float().cuda())
                else:
                    trans.append((-coord[bid, mask, :2] + targets['joint_img'][:, smpl_x.joint_part['face'], :][bid,
                                                          mask, :2]).mean(0))
            trans = torch.stack(trans)[:, None, :]
            coord = coord + trans  # global translation alignment
            joint_proj = torch.cat((joint_proj[:, :smpl_x.joint_part['face'][0], :], coord,
                                    joint_proj[:, smpl_x.joint_part['face'][-1] + 1:, :]), 1)


            loss['joint_proj'] = self.coord_loss(joint_proj, targets['joint_img'][:, :, :2], meta_info['joint_trunc'])
            loss['joint_img'] = self.coord_loss(joint_img, smpl_x.reduce_joint_set(targets['joint_img']),
                                                smpl_x.reduce_joint_set(meta_info['joint_trunc']), meta_info['is_3D'])
            loss['joint_img_face'] = self.coord_loss(face_joint_img, targets['joint_img'][:, smpl_x.joint_part['face']],
                                                meta_info['joint_trunc'][:, smpl_x.joint_part['face']], meta_info['is_3D'])
            loss['smplx_joint_img'] = self.coord_loss(joint_img, smpl_x.reduce_joint_set(targets['smplx_joint_img']),
                                                      smpl_x.reduce_joint_set(meta_info['smplx_joint_trunc']))
            return loss
        else:
            # change hand output joint_img according to hand bbox
            for part_name, bbox in (('lhand', lhand_bbox), ('rhand', rhand_bbox)):
                joint_img[:, smpl_x.pos_joint_part[part_name], 0] *= (
                        ((bbox[:, None, 2] - bbox[:, None, 0]) / cfg.input_body_shape[1] * cfg.output_hm_shape[2]) /
                        cfg.output_hand_hm_shape[2])
                joint_img[:, smpl_x.pos_joint_part[part_name], 0] += (
                        bbox[:, None, 0] / cfg.input_body_shape[1] * cfg.output_hm_shape[2])
                joint_img[:, smpl_x.pos_joint_part[part_name], 1] *= (
                        ((bbox[:, None, 3] - bbox[:, None, 1]) / cfg.input_body_shape[0] * cfg.output_hm_shape[1]) /
                        cfg.output_hand_hm_shape[1])
                joint_img[:, smpl_x.pos_joint_part[part_name], 1] += (
                        bbox[:, None, 1] / cfg.input_body_shape[0] * cfg.output_hm_shape[1])

            # change input_body_shape to input_img_shape
            for bbox in (lhand_bbox, rhand_bbox, face_bbox):
                bbox[:, 0] *= cfg.input_img_shape[1] / cfg.input_body_shape[1]
                bbox[:, 1] *= cfg.input_img_shape[0] / cfg.input_body_shape[0]
                bbox[:, 2] *= cfg.input_img_shape[1] / cfg.input_body_shape[1]
                bbox[:, 3] *= cfg.input_img_shape[0] / cfg.input_body_shape[0]

            # test output
            out = {}
            out['img'] = inputs['img']
            out['joint_img'] = joint_img
            out['smplx_joint_proj'] = joint_proj
            out['smplx_mesh_cam'] = mesh_cam
            out['smplx_root_pose'] = root_pose
            out['smplx_body_pose'] = body_pose
            out['smplx_lhand_pose'] = lhand_pose
            out['smplx_rhand_pose'] = rhand_pose
            out['smplx_jaw_pose'] = jaw_pose
            out['smplx_shape'] = shape
            out['smplx_expr'] = expr
            out['cam_trans'] = cam_trans
            out['lhand_bbox'] = lhand_bbox
            out['rhand_bbox'] = rhand_bbox
            out['face_bbox'] = face_bbox
            if 'smplx_pose' in targets:
                out['smplx_mesh_cam_pseudo_gt'] = mesh_pseudo_gt
            if 'smplx_mesh_cam' in targets:
                out['smplx_mesh_cam_target'] = targets['smplx_mesh_cam']
            if 'bb2img_trans' in meta_info:
                out['bb2img_trans'] = meta_info['bb2img_trans']
            return out

# ==============================================================================
# 4. Final get_model Function
# ==============================================================================
def get_model(mode):
    # ================== 逻辑计算 ==================
    # 动态计算 Neck Channels (输入到 Decoder 的特征维度)
    # 对应 config 里的 upscale 逻辑
    neck_channels = []
    # print(f"[Debug] cfg.upscale is: {getattr(cfg, 'upscale', 'Not Found')}")
    if cfg.upscale == 1:
        neck_channels = [cfg.feat_dim] # [1024]
    elif cfg.upscale == 2:
        neck_channels = [cfg.feat_dim // 2, cfg.feat_dim] # [512, 1024]
    elif cfg.upscale == 4:
        neck_channels = [cfg.feat_dim // 4, cfg.feat_dim // 2, cfg.feat_dim] # [256, 512, 1024]
    elif cfg.upscale == 8:
        neck_channels = [cfg.feat_dim // 8, cfg.feat_dim // 4, cfg.feat_dim // 2, cfg.feat_dim]
    
    # 默认使用 256 作为 embed_dim
    emb_dim = 256
    
    # 打印一下，方便 Debug
    print(f"Upscale: {cfg.upscale}, Feat Dim: {cfg.feat_dim}")
    print(f"Decoder Input Channels: {neck_channels}, Num Levels: {len(neck_channels)}")
    print(f"Face Queries: {cfg.face_pos_joint_num}, Hand Queries: 20 (Default)")

    # ================== 模型构建 ==================
    # 1. Body (ViT-Large)
    vit_backbone = StandardViT(img_size=(256, 192), embed_dim=1024, depth=24, num_heads=16)
    vit = ViTWrapper(vit_backbone)
    
    body_position_net = PositionNet('body', feat_dim=cfg.feat_dim)
    body_rotation_net = BodyRotationNet(feat_dim=cfg.feat_dim)
    box_net = BoxNet(feat_dim=cfg.feat_dim)

    # 2. Hand
    hand_roi_net = HandRoI(feat_dim=cfg.feat_dim, upscale=cfg.upscale)
    hand_position_net = PositionNet('hand', feat_dim=cfg.feat_dim//2)
    hand_rotation_net = HandRotationNet('hand', feat_dim=256)
    
    # Hand Decoder: 传入 neck_channels，内部自动处理 input_proj
    hand_decoder = PoseurDecoder(
        in_channels_list=neck_channels, # 如果用了 internal backbone，这个参数其实会被忽略/覆盖
        embed_dim=256, 
        num_heads=8, 
        num_layers=6, 
        num_queries=20,
        use_internal_backbone=True # <--- 开启这个
    )

    # 3. Face
    face_roi_net = FaceRoI(feat_dim=cfg.feat_dim, upscale=cfg.upscale)
    face_position_net = PositionNet('face', feat_dim=cfg.feat_dim//2)
    face_regressor = FaceRegressor(feat_dim=cfg.feat_dim, joint_feat_dim=256)
    
    # Face Decoder: 使用 cfg.face_pos_joint_num (72)
    face_decoder = PoseurDecoder(
        in_channels_list=neck_channels, 
        embed_dim=256, 
        num_heads=8, 
        num_layers=6, 
        num_queries=72,
        use_internal_backbone=True # <--- 开启这个
    )

    # 4. Initialization
    if mode == 'train':
        def init_weights(m):
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.LayerNorm): nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        
        for n in [body_position_net, body_rotation_net, box_net, 
                  hand_position_net, hand_roi_net, hand_rotation_net,
                  face_position_net, face_roi_net, face_regressor]:
            n.apply(init_weights)

        if os.path.exists(cfg.encoder_pretrained_model_path):
            ckpt = torch.load(cfg.encoder_pretrained_model_path, map_location='cpu')
            vit.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt, strict=False)
            print("ViT Backbone loaded.")
        else:
            print(f"Warning: {cfg.encoder_pretrained_model_path} not found.")

    encoder = vit.backbone
    model = Model(encoder, body_position_net, body_rotation_net, box_net, 
                  hand_position_net, hand_roi_net, hand_decoder, hand_rotation_net,
                  face_position_net, face_roi_net, face_decoder, face_regressor)
    return model