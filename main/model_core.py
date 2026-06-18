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
from common.nets.loss import CoordLoss, ParamLoss, CELoss, RotMatParamLoss
from common.utils.human_models import smpl_x
from common.utils.transforms import rot6d_to_axis_angle, restore_bbox
from common.nets.vit import *

from config import cfg 

from common.nets.my_decoder import HandDecoder, FaceDecoder
from common.nets.DCNv4 import DCNv4PositionNet


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
        self.rotmat_param_loss = RotMatParamLoss()

        self.body_num_joints = len(smpl_x.pos_joint_part['body'])
        self.hand_num_joints = len(smpl_x.pos_joint_part['rhand'])

        # self.trainable_modules = [self.encoder, self.body_position_net, self.body_regressor,
        #                           self.box_net, self.hand_position_net, self.hand_roi_net, self.hand_regressor,
        #                           self.face_regressor, self.face_roi_net, self.face_position_net]
        # self.special_trainable_modules = [self.hand_decoder, self.face_decoder]

        # ---- 定义哪些模块冻结 ----
        self.frozen_modules = [
            'encoder',
            'body_position_net',
            'body_regressor',
            'box_net',
            'hand_roi_net',
            'face_roi_net',
        ]


        # 新模块：使用正常学习率
        self.trainable_modules = [
            self.hand_position_net,
            self.hand_decoder,
            self.face_position_net,
            self.face_decoder,
        ]

        # 回归网络：使用较小学习率（从预训练微调）
        self.special_trainable_modules = [
            self.hand_regressor,
            self.face_regressor,
        ]

        # 所有需要训练的模块名称（用于 freeze/unfreeze 判断）
        self.trainable_module_names = [
            'hand_position_net',
            'hand_decoder',
            'hand_regressor',
            'face_position_net',
            'face_decoder',
            'face_regressor',
        ]

        self.training_phase = 1

    def set_training_phase(self, phase):
        """
        Phase 1: hand-focused finetuning. Optionally trains hand_regressor at
        a low learning rate and keeps face modules frozen by default.
        Phase 2: all configured trainable modules can be finetuned.
        """
        self.training_phase = phase
        
        if phase == 1:
            regressor_trainable = {
                'hand_regressor': getattr(cfg, 'phase1_train_hand_regressor', True),
                'face_regressor': getattr(cfg, 'train_face_modules', False),
            }
            for module_name, trainable in regressor_trainable.items():
                module = getattr(self, module_name)
                module.train(trainable)
                for param in module.parameters():
                    param.requires_grad = trainable
            print("Phase 1: hand-focused finetuning")
            
        elif phase == 2:
            # 解冻 regressor
            regressor_trainable = {
                'hand_regressor': True,
                'face_regressor': getattr(cfg, 'train_face_modules', False),
            }
            for module_name, trainable in regressor_trainable.items():
                module = getattr(self, module_name)
                module.train(trainable)
                for param in module.parameters():
                    param.requires_grad = trainable
            print("Phase 2: configured module finetuning")


    def freeze_modules(self):
        """
        冻结指定模块:
        1. 设置 requires_grad = False
        2. 设置 eval() 模式（BN 层使用 running stats 而非 batch stats）
        """
        frozen_param_count = 0
        trainable_param_count = 0

        for module_name in self.frozen_modules:
            if isinstance(module_name, str):
                if hasattr(self, module_name):
                    module = getattr(self, module_name)
                else:
                    print(f"⚠️ 警告: 模型中没有名为 '{module_name}' 的模块，跳过冻结。")
                    continue
            else:
                # 如果你不小心传了 self.encoder 进来，走这个分支
                module = module_name
            
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
                frozen_param_count += param.numel()

        for module_name in self.trainable_modules:
            if isinstance(module_name, str):
                if hasattr(self, module_name):
                    module = getattr(self, module_name)
                else:
                    print(f"⚠️ 警告: 模型中没有名为 '{module_name}' 的模块，跳过冻结。")
                    continue
            else:
                # 如果你不小心传了 self.encoder 进来，走这个分支
                module = module_name
            module.train()
            for param in module.parameters():
                param.requires_grad = True
                trainable_param_count += param.numel()

        print(f"\n{'=' * 60}")
        print(f"模块冻结完成:")
        print(f"  🧊 冻结参数量: {frozen_param_count:,} "
              f"({frozen_param_count / 1e6:.1f}M)")
        print(f"  🔥 可训练参数量: {trainable_param_count:,} "
              f"({trainable_param_count / 1e6:.1f}M)")
        print(f"  📊 可训练占比: "
              f"{trainable_param_count / (frozen_param_count + trainable_param_count) * 100:.1f}%")
        print(f"{'=' * 60}\n")

    def train(self, mode=True):
        """
        重写 train()，确保冻结模块始终处于 eval 模式。
        
        ⚠️ 这一步非常关键！
        PyTorch 默认 model.train() 会把所有子模块设为 train 模式，
        会导致冻结模块中的 BatchNorm 使用 batch statistics 而非 running statistics，
        产生训练/推理不一致。
        """
        # 先调用默认 train()
        super().train(mode)

        # 再把冻结模块强制设回 eval()
        if mode:
            for module_name in self.frozen_modules:
                module = getattr(self, module_name)
                module.eval()

        return self

    def get_trainable_params(self):
        """
        返回只包含可训练参数的列表，用于传给 optimizer。
        """
        trainable_params = []
        for module_name in self.trainable_modules:
            module = getattr(self, module_name)
            trainable_params.extend(
                [p for p in module.parameters() if p.requires_grad]
            )
        return trainable_params

    def _sanitize_hand_bbox(self, bbox):
        min_size = 1.0
        finite = torch.isfinite(bbox).all(dim=1)
        safe_bbox = torch.nan_to_num(bbox, nan=0.0, posinf=0.0, neginf=0.0)
        bbox_w = safe_bbox[:, 2] - safe_bbox[:, 0]
        bbox_h = safe_bbox[:, 3] - safe_bbox[:, 1]
        valid = finite & (bbox_w > min_size) & (bbox_h > min_size)
        default_bbox = safe_bbox.new_tensor([
            0.0,
            0.0,
            float(cfg.input_body_shape[1]),
            float(cfg.input_body_shape[0]),
        ]).view(1, 4)
        safe_bbox = torch.where(valid[:, None], safe_bbox, default_bbox.expand_as(safe_bbox))
        return safe_bbox, valid.float()

    def _hand_bbox_transform(self, bbox):
        min_size = 1.0
        bbox_w = (bbox[:, 2] - bbox[:, 0]).clamp(min=min_size)
        bbox_h = (bbox[:, 3] - bbox[:, 1]).clamp(min=min_size)
        x0 = bbox[:, None, 0] / cfg.input_body_shape[1] * cfg.output_hm_shape[2]
        y0 = bbox[:, None, 1] / cfg.input_body_shape[0] * cfg.output_hm_shape[1]
        x_scale = cfg.output_hand_hm_shape[2] / (
            bbox_w[:, None] / cfg.input_body_shape[1] * cfg.output_hm_shape[2]
        )
        y_scale = cfg.output_hand_hm_shape[1] / (
            bbox_h[:, None] / cfg.input_body_shape[0] * cfg.output_hm_shape[1]
        )
        return x0, y0, x_scale, y_scale

    def get_trainable_params_with_lr(self, base_lr):
        """
        支持不同模块使用不同学习率。
        
        策略:
        - 新模块 (position_net, decoder): 使用 base_lr
        - 回归网络 (如果用预训练初始化): 使用 base_lr * 0.1
        """
        param_groups = [
            # 新模块: 全量学习率
            {
                'params': list(self.hand_position_net.parameters()) +
                          list(self.hand_decoder.parameters()) +
                          list(self.face_position_net.parameters()) +
                          list(self.face_decoder.parameters()),
                'lr': base_lr,
                'name': 'new_modules'
            },
            # 回归网络: 如果是从预训练微调，可以用较小学习率
            {
                'params': list(self.hand_regressor.parameters()) +
                          list(self.face_regressor.parameters()),
                'lr': base_lr * 0.1,  # 微调用小学习率
                'name': 'regressors'
            },
        ]

        # 过滤掉没有梯度的参数
        param_groups = [
            {
                **group,
                'params': [p for p in group['params'] if p.requires_grad]
            }
            for group in param_groups
        ]

        for group in param_groups:
            print(f"  Param group '{group['name']}': "
                  f"{sum(p.numel() for p in group['params']):,} params, "
                  f"lr={group['lr']}")

        return param_groups

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
        body_img = body_img.to('cuda')

        # 1. Encoder
        img_feat, task_tokens = self.encoder(body_img)  # task_token:[bs, N, c]
        # print(img_feat.shape)
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

        # Boxes used for the differentiable ROI crop. By default these are the
        # box_net predictions, but for hand-only datasets (e.g. InterHand) the
        # frozen top-down box_net outputs degenerate whole-frame boxes, so we
        # crop with the GT hand box instead. The box_net predictions
        # (lhand_bbox_center/size, ...) are kept untouched for the bbox loss.
        roi_lhand_center, roi_lhand_size = lhand_bbox_center, lhand_bbox_size
        roi_rhand_center, roi_rhand_size = rhand_bbox_center, rhand_bbox_size
        if getattr(cfg, 'inject_gt_hand_bbox', False) and 'is_hand_only' in meta_info \
                and 'lhand_bbox_center' in targets:
            hand_only = meta_info['is_hand_only'].view(-1) > 0
            l_use = (hand_only & (meta_info['lhand_bbox_valid'].view(-1) > 0)).view(-1, 1)
            r_use = (hand_only & (meta_info['rhand_bbox_valid'].view(-1) > 0)).view(-1, 1)
            roi_lhand_center = torch.where(l_use, targets['lhand_bbox_center'], roi_lhand_center)
            roi_lhand_size = torch.where(l_use, targets['lhand_bbox_size'], roi_lhand_size)
            roi_rhand_center = torch.where(r_use, targets['rhand_bbox_center'], roi_rhand_center)
            roi_rhand_size = torch.where(r_use, targets['rhand_bbox_size'], roi_rhand_size)

        lhand_bbox = restore_bbox(roi_lhand_center, roi_lhand_size, cfg.input_hand_shape[1] / cfg.input_hand_shape[0], 2.0).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        rhand_bbox = restore_bbox(roi_rhand_center, roi_rhand_size, cfg.input_hand_shape[1] / cfg.input_hand_shape[0], 2.0).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        face_bbox = restore_bbox(face_bbox_center, face_bbox_size, cfg.input_face_shape[1] / cfg.input_face_shape[0], 1.5).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        lhand_bbox, lhand_bbox_safe_valid = self._sanitize_hand_bbox(lhand_bbox)
        rhand_bbox, rhand_bbox_safe_valid = self._sanitize_hand_bbox(rhand_bbox)

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
        # print(f"hand_coord_init: {hand_coord_init.shape}, hand_img_feat_joints: {hand_img_feat_joints.shape}, hand_feats[-2]: {hand_feats[-2].shape}")
        hand_img_feat_joints = self.hand_decoder(hand_feats, coord_init=hand_coord_init[:, :, :2].detach(), query_init=hand_img_feat_joints)

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
        face_img_feat_joints = self.face_decoder(face_feats, coord_init=face_coord_init[:, :, :2].detach(), query_init=face_img_feat_joints)
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
            # Optional rotation-matrix pose loss on the 30 hand finger joints
            # (cfg.use_hand_rotmat_pose_loss, default OFF). Raw axis-angle L1 is
            # discontinuous / non-unique for articulated fingers; a rotmat L1 is
            # rotation-aware. ADDITIVE to the axis-angle smplx_pose loss above
            # (not a replacement), masked by smplx_pose_valid so it only fires on
            # joints with valid GT (BEDLAM hands; InterHand 15+15 fingers). When
            # the flag is off the key is never created, so it has zero effect.
            if getattr(cfg, 'use_hand_rotmat_pose_loss', False):
                bs = pose.shape[0]
                hand_joint_idx = (list(smpl_x.orig_joint_part['lhand']) +
                                  list(smpl_x.orig_joint_part['rhand']))
                pose_hand = pose.reshape(bs, smpl_x.orig_joint_num, 3)[:, hand_joint_idx, :]
                gt_hand = targets['smplx_pose'].reshape(bs, smpl_x.orig_joint_num, 3)[:, hand_joint_idx, :]
                valid_hand = meta_info['smplx_pose_valid'].reshape(bs, smpl_x.orig_joint_num, 3)[:, hand_joint_idx, :1]
                loss['smplx_hand_pose_rotmat'] = self.rotmat_param_loss(
                    pose_hand, gt_hand, valid_hand) * getattr(cfg, 'hand_rotmat_pose_loss_weight', 1.0)
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
            # Dynamic BEDLAM hand-ROI quality gate setup (applied per-coord below).
            roi_gate_on = getattr(cfg, 'bedlam_use_hand_roi_quality', False)
            if roi_gate_on:
                roi_cov_thr = float(getattr(cfg, 'bedlam_hand_roi_coverage_thr', 0.6))
                roi_min_joints = int(getattr(cfg, 'bedlam_hand_roi_min_joints', 8))
                if 'is_bedlam' in meta_info:
                    is_bedlam = meta_info['is_bedlam'].to(lhand_bbox.device).view(-1).float()
                else:
                    is_bedlam = lhand_bbox.new_zeros(lhand_bbox.shape[0])
            for part_name, bbox, bbox_valid in (
                    ('lhand', lhand_bbox, lhand_bbox_safe_valid),
                    ('rhand', rhand_bbox, rhand_bbox_safe_valid)):
                x0, y0, x_scale, y_scale = self._hand_bbox_transform(bbox)
                for coord_name, trunc_name in (('joint_img', 'joint_trunc'), ('smplx_joint_img', 'smplx_joint_trunc')):
                    x = targets[coord_name][:, smpl_x.joint_part[part_name], 0].clone()
                    y = targets[coord_name][:, smpl_x.joint_part[part_name], 1].clone()
                    z = targets[coord_name][:, smpl_x.joint_part[part_name], 2].clone()
                    trunc = meta_info[trunc_name][:, smpl_x.joint_part[part_name], 0]
                    gt_valid = trunc > 0  # GT-valid hand joints (before bbox/in_bbox masking)

                    x = (x - x0) * x_scale
                    y = (y - y0) * y_scale
                    z *= cfg.output_hand_hm_shape[0] / cfg.output_hm_shape[0]
                    coord = torch.stack((x, y, z), 2)
                    coord_finite = torch.isfinite(coord).all(dim=2)
                    coord = torch.nan_to_num(coord, nan=0.0, posinf=0.0, neginf=0.0)
                    in_bbox = ((coord[:, :, 0] >= 0) *
                               (coord[:, :, 0] < cfg.output_hand_hm_shape[2]) *
                               (coord[:, :, 1] >= 0) *
                               (coord[:, :, 1] < cfg.output_hand_hm_shape[1]))
                    trunc = trunc * bbox_valid[:, None] * coord_finite.float() * in_bbox.float()
                    # Dynamic hand-ROI quality gate: coverage = (#GT-valid hand
                    # joints inside the ACTUAL augmented predicted bbox) / (#GT-valid
                    # hand joints). Low coverage => box_net framed the wrong person
                    # => veto this hand's whole 2D trunc. Computed live so it matches
                    # train-time augmentation exactly (no flip/scale/rot mismatch).
                    # BEDLAM-only; InterHand (GT bbox) and non-BEDLAM keep roi_ok=1.
                    if roi_gate_on:
                        gate_valid = gt_valid & coord_finite
                        n_valid = gate_valid.sum(dim=1)
                        coverage = (gate_valid & in_bbox.bool()).sum(dim=1).float() / n_valid.clamp(min=1).float()
                        roi_ok = ((bbox_valid > 0) & (n_valid >= roi_min_joints) & (coverage >= roi_cov_thr)).float()
                        roi_ok = torch.where(is_bedlam > 0, roi_ok, torch.ones_like(roi_ok))
                        trunc = trunc * roi_ok[:, None]
                    trunc = trunc[:, :, None]
                    targets[coord_name] = torch.cat((targets[coord_name][:, :smpl_x.joint_part[part_name][0], :], coord,
                                                     targets[coord_name][:, smpl_x.joint_part[part_name][-1] + 1:, :]),
                                                    1)
                    meta_info[trunc_name] = torch.cat((meta_info[trunc_name][:, :smpl_x.joint_part[part_name][0], :],
                                                       trunc,
                                                       meta_info[trunc_name][:, smpl_x.joint_part[part_name][-1] + 1:,
                                                       :]), 1)

            # change hand projected joint coordinates according to hand bbox (cfg.output_hm_shape -> hand bbox space)
            for part_name, bbox, bbox_valid in (
                    ('lhand', lhand_bbox, lhand_bbox_safe_valid),
                    ('rhand', rhand_bbox, rhand_bbox_safe_valid)):
                x0, y0, x_scale, y_scale = self._hand_bbox_transform(bbox)
                x = joint_proj[:, smpl_x.joint_part[part_name], 0].clone()
                y = joint_proj[:, smpl_x.joint_part[part_name], 1].clone()

                x = (x - x0) * x_scale
                y = (y - y0) * y_scale

                coord = torch.stack((x, y), 2)
                coord_finite = torch.isfinite(coord).all(dim=2)
                coord = torch.nan_to_num(coord, nan=0.0, posinf=0.0, neginf=0.0)
                coord = torch.where(
                    bbox_valid[:, None, None].bool(),
                    coord,
                    coord.new_zeros(coord.shape),
                )
                trans = []
                for bid in range(coord.shape[0]):
                    target_coord = targets['joint_img'][bid, smpl_x.joint_part[part_name], :2]
                    mask = meta_info['joint_trunc'][bid, smpl_x.joint_part[part_name], 0] == 1
                    mask = mask & coord_finite[bid] & torch.isfinite(target_coord).all(dim=1)
                    if torch.sum(mask) == 0:
                        trans.append(coord.new_zeros((2)))
                    else:
                        trans.append((-coord[bid, mask, :2] + target_coord[mask, :2]).mean(0))
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


            # B-class anti-pollution: drop the BEDLAM hand contribution to the two
            # heatmap-supervising losses (joint_img / smplx_joint_img). These are
            # the ONLY gradient path into the hand_position_net soft-argmax head
            # (hand_joint_img is detached everywhere else), and BEDLAM's ~1px hand
            # crops make that target ill-posed, collapsing the shared head. We build
            # a masked trunc used ONLY for these two losses; joint_proj keeps the
            # original coverage-gated joint_trunc (it is stable and is the only
            # real-image hand 2D-alignment signal). InterHand / non-BEDLAM untouched.
            joint_trunc_img = meta_info['joint_trunc']
            smplx_joint_trunc_img = meta_info['smplx_joint_trunc']
            if not getattr(cfg, 'bedlam_supervise_hand_img', True) and 'is_bedlam' in meta_info:
                is_bedlam = meta_info['is_bedlam'].to(joint_trunc_img.device).view(-1, 1, 1).float()
                hand_joint_mask = joint_trunc_img.new_zeros(joint_trunc_img.shape[1])
                hand_joint_mask[smpl_x.joint_part['lhand']] = 1.0
                hand_joint_mask[smpl_x.joint_part['rhand']] = 1.0
                keep = 1.0 - is_bedlam * hand_joint_mask.view(1, -1, 1)
                joint_trunc_img = joint_trunc_img * keep
                smplx_joint_trunc_img = smplx_joint_trunc_img * keep

            loss['joint_proj'] = self.coord_loss(joint_proj, targets['joint_img'][:, :, :2], meta_info['joint_trunc'])
            joint_img_loss = self.coord_loss(joint_img, smpl_x.reduce_joint_set(targets['joint_img']),
                                             smpl_x.reduce_joint_set(joint_trunc_img), meta_info['is_3D'])
            loss['joint_img'] = joint_img_loss
            loss['joint_img_face'] = self.coord_loss(face_joint_img, targets['joint_img'][:, smpl_x.joint_part['face']],
                                                meta_info['joint_trunc'][:, smpl_x.joint_part['face']], meta_info['is_3D'])
            smplx_joint_img_loss = self.coord_loss(joint_img, smpl_x.reduce_joint_set(targets['smplx_joint_img']),
                                                   smpl_x.reduce_joint_set(smplx_joint_trunc_img))
            loss['smplx_joint_img'] = smplx_joint_img_loss
            # Diagnostics only (keys prefixed '_' are excluded from the backward
            # sum in train.py): split the joint_img / smplx_joint_img coord loss
            # into xy vs z so a depth-channel blow-up is visible per data source.
            # Slicing the already-computed loss tensors keeps these consistent
            # with the supervised values (is_3D gating on z is already applied).
            loss['_joint_img_xy'] = joint_img_loss[:, :, :2]
            loss['_joint_img_z'] = joint_img_loss[:, :, 2:]
            loss['_smplx_joint_img_xy'] = smplx_joint_img_loss[:, :, :2]
            loss['_smplx_joint_img_z'] = smplx_joint_img_loss[:, :, 2:]
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
    if mode == 'test':
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
            use_internal_backbone=False # <--- 开启这个
        )

    elif mode == 'train' or mode == "test1":
        hand_roi_net = HandRoI(feat_dim=cfg.feat_dim, upscale=cfg.upscale)
        hand_position_net = DCNv4PositionNet('hand', feat_dim=cfg.feat_dim//2, dcnv4_group=4, num_blocks=3)
        hand_rotation_net = HandRotationNet('hand', feat_dim=256)
        
        # Hand Decoder
        # 手部解码器
        hand_decoder = HandDecoder(
            d_model=256,                          # 与 hand_conv 输出通道一致
            nhead=8,
            num_decoder_layers=3,
            dim_feedforward=1024,
            dropout=0.1,
            n_levels=3,             # 多尺度特征层数
            n_points=4,
            num_joints=cfg.hand_pos_joint_num,    # 通常 20
            feat_channels=neck_channels,               # ROI 特征通道 (如 768)
        )

    # 3. Face
    if mode == 'test':
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
            use_internal_backbone=False # <--- 开启这个
        )
    
    elif mode == 'train' or mode == "test1":
        face_roi_net = FaceRoI(feat_dim=cfg.feat_dim, upscale=cfg.upscale)
        face_position_net = DCNv4PositionNet('face', feat_dim=cfg.feat_dim//2, dcnv4_group=4, num_blocks=3)
        face_regressor = FaceRegressor(feat_dim=cfg.feat_dim, joint_feat_dim=256)
        
        # Face Decoder:
        # 脸部解码器
        face_decoder = FaceDecoder(
            d_model=256,
            nhead=8,
            num_decoder_layers=3,
            dim_feedforward=1024,
            dropout=0.1,
            n_levels=3,
            n_points=4,
            num_joints=cfg.face_pos_joint_num,    # 通常 68
            feat_channels=neck_channels,
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
