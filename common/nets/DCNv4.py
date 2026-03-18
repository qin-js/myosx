import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from common.nets.layer import make_conv_layers, make_linear_layers, make_deconv_layers
from common.utils.transforms import sample_joint_features, soft_argmax_2d, soft_argmax_3d
from common.utils.human_models import smpl_x
from DCNv4 import DCNv4


# ==================================================================
# DCNv4 Block
# ==================================================================
class DCNv4Block(nn.Module):
    """
    Pre-Norm → DCNv4 → Residual → Pre-Norm → FFN → Residual
    """
    def __init__(self, channels, dcnv4_group=4, kernel_size=3, ffn_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.dcnv4 = DCNv4(
            channels=channels,
            kernel_size=kernel_size,
            stride=1,
            pad=kernel_size // 2,
            dilation=1,
            group=dcnv4_group,
            offset_scale=1.0,
        )
        self.norm2 = nn.LayerNorm(channels)
        ffn_hidden = int(channels * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, channels),
        )
        # 加上这行：让残差块在初始化时输出为 0，即初始状态为 Identity
        nn.init.constant_(self.ffn[-1].weight, 0.0)
        nn.init.constant_(self.ffn[-1].bias, 0.0)

    def forward(self, x, H, W):
        """
        x: (B, H*W, C)
        """
        shortcut = x
        x = self.norm1(x)
        x = self.dcnv4(x, shape=(H, W))
        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = shortcut + x
        return x


# ==================================================================
# DCNv4 PositionNet — 替代原始 PositionNet
# ==================================================================
class DCNv4PositionNet(nn.Module):
    """
    使用 DCNv4 增强的 PositionNet。
    
    与原始 PositionNet 保持 **完全相同的输入输出接口**：
      - body:      返回 (joint_hm, joint_coord)
      - hand/face: 返回 (joint_hm, joint_coord, img_feat_joints)
    """

    def __init__(self, part, feat_dim=768, dcnv4_group=4, num_blocks=3, ffn_ratio=4):
        """
        Args:
            part (str):        'body', 'hand', or 'face'
            feat_dim (int):    输入特征通道数（与原始 PositionNet 一致）
            dcnv4_group (int): DCNv4 分组数
            num_blocks (int):  DCNv4 Block 数量
            ffn_ratio (int):   FFN 扩展倍率
        """
        super().__init__()
        self.part = part

        # ----- 与原始 PositionNet 完全一致的配置 -----
        if part == 'body':
            self.joint_num = len(smpl_x.pos_joint_part['body'])
            self.hm_shape = cfg.output_hm_shape  # (D, H, W)
        elif part == 'hand':
            self.joint_num = cfg.hand_pos_joint_num
            self.hm_shape = cfg.output_hand_hm_shape
        elif part == 'face':
            self.joint_num = cfg.face_pos_joint_num
            self.hm_shape = cfg.output_face_hm_shape

        # =============================================================
        # DCNv4 通道对齐
        # 要求: channels % group == 0 且 (channels // group) % 16 == 0
        # =============================================================
        min_unit = dcnv4_group * 16
        if feat_dim % min_unit == 0:
            self.dcn_channels = feat_dim
        else:
            self.dcn_channels = ((feat_dim + min_unit - 1) // min_unit) * min_unit

        # =============================================================
        # 输入投影（feat_dim → dcn_channels，如果需要对齐）
        # =============================================================
        if self.dcn_channels != feat_dim:
            self.input_proj = nn.Sequential(
                nn.Conv2d(feat_dim, self.dcn_channels, 1, bias=False),
                nn.BatchNorm2d(self.dcn_channels),
                nn.GELU(),
            )
        else:
            self.input_proj = nn.Identity()

        # =============================================================
        # DCNv4 特征增强模块
        # =============================================================
        self.dcnv4_blocks = nn.ModuleList([
            DCNv4Block(
                channels=self.dcn_channels,
                dcnv4_group=dcnv4_group,
                kernel_size=3,
                ffn_ratio=ffn_ratio,
            )
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(self.dcn_channels)

        # =============================================================
        # 通道恢复（dcn_channels → feat_dim，如果需要）
        # =============================================================
        if self.dcn_channels != feat_dim:
            self.output_proj = nn.Sequential(
                nn.Conv2d(self.dcn_channels, feat_dim, 1, bias=False),
                nn.BatchNorm2d(feat_dim),
                nn.GELU(),
            )
        else:
            self.output_proj = nn.Identity()

        # =============================================================
        # Heatmap Head: 与原始完全一致，1×1 conv
        # 输入: (B, feat_dim, H, W)
        # 输出: (B, joint_num * D, H, W)
        # =============================================================
        self.conv = make_conv_layers(
            [feat_dim, self.joint_num * self.hm_shape[0]],
            kernel=1, stride=1, padding=0, bnrelu_final=False
        )

        # =============================================================
        # Hand/Face 特征降维（与原始完全一致，降到 256 通道）
        # =============================================================
        if part == 'hand':
            self.hand_conv = make_conv_layers(
                [feat_dim, 256], kernel=1, stride=1, padding=0
            )
        elif part == 'face':
            self.face_conv = make_conv_layers(
                [feat_dim, 256], kernel=1, stride=1, padding=0
            )

    def forward(self, img_feat):
        """
        与原始 PositionNet.forward() 完全相同的接口。

        Args:
            img_feat: (B, feat_dim, H, W)
                      其中 (H, W) == (hm_shape[1], hm_shape[2])

        Returns:
            body 模式:
                joint_hm:   (B, J, D, H, W)  softmax 归一化的 3D 热力图
                joint_coord: (B, J, 3)        soft-argmax 关节坐标

            hand/face 模式:
                joint_hm:        (B, J, D, H, W)  softmax 归一化的 3D 热力图
                joint_coord:     (B, J, 3)         soft-argmax 关节坐标
                img_feat_joints: (B, J, 256)       关节位置处采样的特征
        """
        B, C, H, W = img_feat.shape
        assert (H, W) == (self.hm_shape[1], self.hm_shape[2]), \
            f"输入空间尺寸 ({H}, {W}) 与 hm_shape ({self.hm_shape[1]}, {self.hm_shape[2]}) 不匹配"

        # ---- Step 1: 输入投影 ----
        x = self.input_proj(img_feat)  # (B, dcn_channels, H, W)

        # ---- Step 2: DCNv4 特征增强 ----
        x_tokens = x.flatten(2).transpose(1, 2).contiguous()  # (B, H*W, dcn_channels)
        for block in self.dcnv4_blocks:
            x_tokens = block(x_tokens, H, W)
        x_tokens = self.final_norm(x_tokens)
        x_enhanced = x_tokens.transpose(1, 2).contiguous().reshape(B, self.dcn_channels, H, W)

        # ---- Step 3: 通道恢复 ----
        x_enhanced = self.output_proj(x_enhanced)  # (B, feat_dim, H, W)

        # ---- Step 4: 生成热力图（与原始完全一致）----
        joint_hm = self.conv(x_enhanced).view(
            -1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2]
        )  # (B, J, D, H, W)

        # ---- Step 5: Soft-Argmax（使用原始的外部函数）----
        joint_coord = soft_argmax_3d(joint_hm)  # (B, J, 3)

        # ---- Step 6: Softmax 归一化热力图（与原始完全一致）----
        joint_hm = F.softmax(
            joint_hm.view(-1, self.joint_num,
                          self.hm_shape[0] * self.hm_shape[1] * self.hm_shape[2]),
            dim=2
        )
        joint_hm = joint_hm.view(
            -1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2]
        )

        # ---- Step 7: 按 part 类型返回（与原始完全一致）----
        if self.part == 'hand':
            img_feat_down = self.hand_conv(x_enhanced)  # (B, 256, H, W)
            img_feat_joints = sample_joint_features(
                img_feat_down, joint_coord.detach()[:, :, :2]  # ⚠️ detach + 只取 x,y
            )
            return joint_hm, joint_coord, img_feat_joints

        elif self.part == 'face':
            img_feat_down = self.face_conv(x_enhanced)  # (B, 256, H, W)
            img_feat_joints = sample_joint_features(
                img_feat_down, joint_coord.detach()[:, :, :2]  # ⚠️ detach + 只取 x,y
            )
            return joint_hm, joint_coord, img_feat_joints

        # body
        return joint_hm, joint_coord