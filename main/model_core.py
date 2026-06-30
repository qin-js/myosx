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

        # Face-only staging: when cfg.train_hand_modules is False, freeze the hand
        # branch like the backbone (no grad + eval BN, handled by freeze_modules /
        # the train() override / _verify_freeze_status) by moving it into
        # frozen_modules and out of the trainable lists. Its NAMES stay in
        # trainable_module_names above, so save_model still writes the (warm) hand
        # tensors into the snapshot — letting a later joint stage warm-start hands.
        if not getattr(cfg, 'train_hand_modules', True):
            self.frozen_modules = self.frozen_modules + [
                'hand_position_net', 'hand_decoder', 'hand_regressor',
            ]
            self.trainable_modules = [
                m for m in self.trainable_modules
                if m not in (self.hand_position_net, self.hand_decoder)
            ]
            self.special_trainable_modules = [
                m for m in self.special_trainable_modules
                if m is not self.hand_regressor
            ]

        # WA2D polish stabilization: freeze only the DCNv4 hand_position_net
        # while still training hand_decoder/hand_regressor. Keep the name in
        # trainable_module_names so lightweight snapshots include its warm-start
        # tensors; freeze/status checks prioritize frozen_modules.
        if (getattr(cfg, 'freeze_hand_position_net', False) and
                getattr(cfg, 'train_hand_modules', True)):
            if 'hand_position_net' not in self.frozen_modules:
                self.frozen_modules = self.frozen_modules + ['hand_position_net']
            self.trainable_modules = [
                m for m in self.trainable_modules
                if m is not self.hand_position_net
            ]

        # Route C (default OFF): unfreeze hand_roi_net so the feature-level hand
        # crop+upsample can co-adapt with the hand decoder/regressor (small lr,
        # set in train.py). Moving the whole module from frozen_modules into the
        # trainable lists is sufficient: freeze_modules() / train() / save_model /
        # _verify_freeze_status / _check_gradient_flow all key off these lists, so
        # the two-step checkpoint stays intact (the frozen-backbone load warm-
        # inits hand_roi_net, then the lightweight overlay restores the trained
        # weights). box_net is intentionally left frozen (it relocates the crop,
        # a far harsher perturbation; staged separately). No-op when False.
        if getattr(cfg, 'train_hand_roi', False):
            self.frozen_modules = [m for m in self.frozen_modules if m != 'hand_roi_net']
            if self.hand_roi_net not in self.trainable_modules:
                self.trainable_modules = self.trainable_modules + [self.hand_roi_net]
            if 'hand_roi_net' not in self.trainable_module_names:
                self.trainable_module_names = self.trainable_module_names + ['hand_roi_net']

        # Body-shape T1 (default OFF): name prefixes of the two decoupled linear
        # heads inside the (otherwise frozen) body_regressor that we allow to
        # train. Empty when cfg.train_body_shape is False -> all related logic in
        # freeze_modules / _verify_freeze_status / save_model / optimizer is a
        # no-op, so the hand/face path is byte-for-byte unchanged.
        self.body_shape_trainable_prefixes = (
            ['body_regressor.shape_out', 'body_regressor.cam_out']
            if getattr(cfg, 'train_body_shape', False) else []
        )

        # End-tip 2D loss weighting (default OFF, no-op when both = 1.0). UBody
        # natural-hand degradation concentrates at the distal joints (tip=_4,
        # then j3=_3); these per-joint multipliers up-weight them in the three
        # hand-space 2D losses (joint_proj / joint_img / smplx_joint_img). Built
        # for the full (joint_num) and reduced/pos (pos_joint_num) joint sets;
        # name-based so non-finger joints always keep weight 1.0.
        _tip_w = float(getattr(cfg, 'hand_tip_loss_weight', 1.0))
        _j3_w = float(getattr(cfg, 'hand_j3_loss_weight', 1.0))
        if _tip_w != 1.0 or _j3_w != 1.0:
            def _mk_hand_w(n, names, hand_part):
                w = torch.ones(n)
                for j in hand_part:
                    nm = names[j]
                    if nm.endswith('_4'):
                        w[j] = _tip_w
                    elif nm.endswith('_3'):
                        w[j] = _j3_w
                return w
            self.register_buffer('hand_loss_weight_full',
                                 _mk_hand_w(smpl_x.joint_num, smpl_x.joints_name, smpl_x.joint_part['hand']))
            self.register_buffer('hand_loss_weight_pos',
                                 _mk_hand_w(smpl_x.pos_joint_num, smpl_x.pos_joints_name, smpl_x.pos_joint_part['hand']))
        else:
            self.hand_loss_weight_full = None
            self.hand_loss_weight_pos = None

        wa_level_weight = torch.tensor([
            float(getattr(cfg, 'hand_wa_2d_j1_weight', 1.0)),
            float(getattr(cfg, 'hand_wa_2d_j2_weight', 1.0)),
            float(getattr(cfg, 'hand_wa_2d_j3_weight', 1.0)),
            float(getattr(cfg, 'hand_wa_2d_tip_weight', 1.0)),
        ], dtype=torch.float32).repeat(5)
        self.register_buffer('hand_wa_2d_level_weight', wa_level_weight)

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
                'hand_regressor': getattr(cfg, 'train_hand_modules', True) and getattr(cfg, 'phase1_train_hand_regressor', True),
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
                'hand_regressor': getattr(cfg, 'train_hand_modules', True),
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

        # Body-shape T1 (gated): re-enable grad on the whitelisted sub-heads of
        # the otherwise-frozen body_regressor. BodyRotationNet has no BatchNorm,
        # so leaving them in eval() is harmless. No-op when the prefix list is empty.
        body_shape_param_count = 0
        if self.body_shape_trainable_prefixes:
            for name, param in self.named_parameters():
                if any(name.startswith(p) for p in self.body_shape_trainable_prefixes):
                    param.requires_grad = True
                    body_shape_param_count += param.numel()
                    frozen_param_count -= param.numel()
                    trainable_param_count += param.numel()
            print(f"  🔧 body-shape T1: 解冻 {body_shape_param_count:,} 参数 "
                  f"({', '.join(self.body_shape_trainable_prefixes)})")

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

    def _hand_wa_2d_source_mask(self, meta_info, device, batch_size):
        source_cfg = getattr(cfg, 'hand_wa_2d_loss_sources', 'ubody,mscoco')
        if isinstance(source_cfg, str):
            source_cfg = source_cfg.replace('，', ',').replace('、', ',')
            sources = {s.strip().lower() for s in source_cfg.split(',') if s.strip()}
        else:
            sources = {str(s).strip().lower() for s in source_cfg if str(s).strip()}

        mask = torch.zeros((batch_size,), device=device)
        if 'all' in sources:
            return mask + 1.0
        if 'ubody' in sources and 'is_ubody' in meta_info:
            mask = torch.maximum(mask, meta_info['is_ubody'].to(device).view(-1).float())
        if 'interhand' in sources and 'is_interhand' in meta_info:
            mask = torch.maximum(mask, meta_info['is_interhand'].to(device).view(-1).float())
        if ('mscoco' in sources or 'coco' in sources) and 'dataset_id' in meta_info:
            dataset_id = meta_info['dataset_id'].to(device).view(-1)
            mask = torch.maximum(mask, (dataset_id == 3).float())
        return mask.clamp(0.0, 1.0)

    def _direct_hand_wa_2d_loss(self, joint_proj, targets, meta_info):
        loss_weight = float(getattr(cfg, 'hand_wa_2d_loss_weight', 0.0))
        if loss_weight <= 0:
            return None

        device = joint_proj.device
        batch_size = joint_proj.shape[0]
        source_mask = self._hand_wa_2d_source_mask(meta_info, device, batch_size)
        min_joints = max(int(getattr(cfg, 'hand_wa_2d_loss_min_joints', 4)), 2)
        # Backward NaN guard: err = dist / diag has per-joint grad magnitude 1/diag
        # (diag is GT-only, no grad), so floor diag to keep the WA2D gradient O(1)
        # and stop it saturating the deformable-attention backward. err_clip then
        # hard-caps the per-joint normalized error against single blown-up joints.
        min_diag = float(getattr(cfg, 'hand_wa_2d_min_diag', 1.0))
        if not min_diag > 0:
            min_diag = 1e-4
        err_clip = float(getattr(cfg, 'hand_wa_2d_err_clip', 0.0))

        per_sample_sum = joint_proj.new_zeros((batch_size,))
        per_sample_weighted_sum = joint_proj.new_zeros((batch_size,))
        hand_count = joint_proj.new_zeros((batch_size,))
        level_weight = self.hand_wa_2d_level_weight.to(device).view(1, -1)

        lhand_idx = list(smpl_x.joint_part['lhand'])
        rhand_idx = list(smpl_x.joint_part['rhand'])
        pred_hand = torch.stack((
            torch.cat((joint_proj[:, smpl_x.lwrist_idx:smpl_x.lwrist_idx + 1, :2],
                       joint_proj[:, lhand_idx, :2]), dim=1),
            torch.cat((joint_proj[:, smpl_x.rwrist_idx:smpl_x.rwrist_idx + 1, :2],
                       joint_proj[:, rhand_idx, :2]), dim=1),
        ), dim=1)

        gt_hand_parts = []
        gt_valid_parts = []
        target_xy = targets['joint_img'][:, :, :2].to(device)
        trunc = meta_info['joint_trunc'].to(device)[:, :, 0] > 0
        for part_name, wrist_idx in (('lhand', smpl_x.lwrist_idx),
                                     ('rhand', smpl_x.rwrist_idx)):
            hand_idx = list(smpl_x.joint_part[part_name])
            gt_hand_parts.append(torch.cat((target_xy[:, wrist_idx:wrist_idx + 1, :],
                                            target_xy[:, hand_idx, :]), dim=1))
            gt_valid_parts.append(torch.cat((trunc[:, wrist_idx:wrist_idx + 1],
                                             trunc[:, hand_idx]), dim=1))
        gt_hand = torch.stack(gt_hand_parts, dim=1)
        gt_valid = torch.stack(gt_valid_parts, dim=1)

        if 'coco_hand_joint_img' in targets and 'coco_hand_joint_trunc' in targets:
            coco_gt_hand = targets['coco_hand_joint_img'].to(device)[:, :, :, :2]
            coco_gt_valid = targets['coco_hand_joint_trunc'].to(device)[:, :, :, 0] > 0
            use_coco = coco_gt_valid.sum(dim=2) > 0
            # UBody/MSCOCO have real COCO-WholeBody hand wrists here; datasets
            # with the dummy MultipleDatasets filler fall back to joint_img.
            gt_hand = torch.where(use_coco[:, :, None, None], coco_gt_hand, gt_hand)
            gt_valid = torch.where(use_coco[:, :, None], coco_gt_valid, gt_valid)

        for side_idx in range(2):
            pred = pred_hand[:, side_idx, 1:, :]
            gt = gt_hand[:, side_idx, 1:, :2]
            pred_wrist = pred_hand[:, side_idx, 0, :]
            gt_wrist = gt_hand[:, side_idx, 0, :2]

            finger_finite = torch.isfinite(pred).all(dim=2) & torch.isfinite(gt).all(dim=2)
            wrist_finite = torch.isfinite(pred_wrist).all(dim=1) & torch.isfinite(gt_wrist).all(dim=1)
            finger_valid = gt_valid[:, side_idx, 1:] & finger_finite
            wrist_valid = gt_valid[:, side_idx, 0] & wrist_finite

            all_gt = gt_hand[:, side_idx, :, :2]
            all_valid = torch.cat((wrist_valid[:, None], finger_valid), dim=1)
            valid_num = all_valid.sum(dim=1)
            gt_min = all_gt.masked_fill(~all_valid[:, :, None], 1e6).amin(dim=1)
            gt_max = all_gt.masked_fill(~all_valid[:, :, None], -1e6).amax(dim=1)
            diag = torch.linalg.norm(gt_max - gt_min, dim=1)

            hand_ok = (source_mask > 0) & wrist_valid & (valid_num >= min_joints) & (diag > 1e-4)
            pred_wa = pred + (gt_wrist - pred_wrist)[:, None, :]
            diff = pred_wa - gt
            diff = torch.where(finger_valid[:, :, None], diff, diff.new_zeros(diff.shape))
            err = torch.sqrt((diff * diff).sum(dim=2) + 1e-8) / diag.clamp(min=min_diag)[:, None]
            if err_clip > 0:
                err = err.clamp(max=err_clip)
            per_hand = (err * finger_valid.float()).sum(dim=1) / finger_valid.float().sum(dim=1).clamp(min=1.0)
            valid_weight = finger_valid.float() * level_weight
            per_hand_weighted = (err * valid_weight).sum(dim=1) / valid_weight.sum(dim=1).clamp(min=1.0)
            per_sample_sum = per_sample_sum + torch.where(hand_ok, per_hand, per_hand.new_zeros(per_hand.shape))
            per_sample_weighted_sum = per_sample_weighted_sum + torch.where(
                hand_ok, per_hand_weighted, per_hand_weighted.new_zeros(per_hand_weighted.shape))
            hand_count = hand_count + hand_ok.float()

        raw = per_sample_sum / hand_count.clamp(min=1.0)
        raw = torch.where(hand_count > 0, raw, raw.new_zeros(raw.shape))
        weighted_raw = per_sample_weighted_sum / hand_count.clamp(min=1.0)
        weighted_raw = torch.where(hand_count > 0, weighted_raw, weighted_raw.new_zeros(weighted_raw.shape))
        return {
            'hand_wa_2d': weighted_raw[:, None] * loss_weight,
            '_hand_wa_2d_raw': raw[:, None],
            '_hand_wa_2d_weighted_raw': weighted_raw[:, None],
            '_hand_wa_2d_active': (hand_count / 2.0)[:, None],
        }

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
        k_value = cam_param.new_tensor([math.sqrt(
            cfg.focal[0] * cfg.focal[1] * cfg.camera_3d_size * cfg.camera_3d_size /
            (cfg.input_body_shape[0] * cfg.input_body_shape[1])
        )]).view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:, None]), 1)
        return cam_trans

    def get_coord(self, root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_trans, mode):
        batch_size = root_pose.shape[0]
        zero_pose = root_pose.new_zeros((batch_size, 3))  # eye poses
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
            return targets['smplx_mesh_cam'].to(next(self.parameters()).device)
        nums = [3, 63, 45, 45, 3]
        accu = []
        temp = 0
        for num in nums:
            temp += num
            accu.append(temp)
        device = next(self.parameters()).device
        pose = targets['smplx_pose'].to(device)
        root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose = \
            pose[:, :accu[0]], pose[:, accu[0]:accu[1]], pose[:, accu[1]:accu[2]], pose[:, accu[2]:accu[3]], pose[:,accu[3]:accu[4]]
        shape = targets['smplx_shape'].to(device)
        expr = targets['smplx_expr'].to(device)
        cam_trans = targets['smplx_cam_trans'].to(device)

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
            box_device = roi_lhand_center.device
            hand_only = meta_info['is_hand_only'].to(box_device).view(-1) > 0
            l_valid = meta_info['lhand_bbox_valid'].to(box_device).view(-1) > 0
            r_valid = meta_info['rhand_bbox_valid'].to(box_device).view(-1) > 0
            l_use = (hand_only & l_valid).view(-1, 1)
            r_use = (hand_only & r_valid).view(-1, 1)
            gt_lhand_center = targets['lhand_bbox_center'].to(box_device)
            gt_lhand_size = targets['lhand_bbox_size'].to(box_device)
            gt_rhand_center = targets['rhand_bbox_center'].to(box_device)
            gt_rhand_size = targets['rhand_bbox_size'].to(box_device)
            roi_lhand_center = torch.where(l_use, gt_lhand_center, roi_lhand_center)
            roi_lhand_size = torch.where(l_use, gt_lhand_size, roi_lhand_size)
            roi_rhand_center = torch.where(r_use, gt_rhand_center, roi_rhand_center)
            roi_rhand_size = torch.where(r_use, gt_rhand_size, roi_rhand_size)

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
            hand_wa_2d_loss = self._direct_hand_wa_2d_loss(
                joint_proj,
                targets,
                meta_info,
            )
            if hand_wa_2d_loss is not None:
                loss.update(hand_wa_2d_loss)
            # change hand target joint_img and joint_trunc according to hand bbox (cfg.output_hm_shape -> downsampled hand bbox space)
            # Dynamic hand-ROI quality gate setup (applied per-coord below).
            roi_gate_on = (getattr(cfg, 'bedlam_use_hand_roi_quality', False) or
                           getattr(cfg, 'ubody_use_hand_roi_quality', False) or
                           getattr(cfg, 'mscoco_use_hand_roi_quality', False))
            if roi_gate_on:
                gate_source = lhand_bbox.new_zeros(lhand_bbox.shape[0])
                gate_cov_thr = lhand_bbox.new_full((lhand_bbox.shape[0],), float('inf'))
                gate_min_joints = lhand_bbox.new_full((lhand_bbox.shape[0],), float('inf'))
                if 'is_bedlam' in meta_info:
                    is_bedlam = meta_info['is_bedlam'].to(lhand_bbox.device).view(-1).float()
                    if getattr(cfg, 'bedlam_use_hand_roi_quality', False):
                        gate_source = torch.maximum(gate_source, is_bedlam)
                        gate_cov_thr = torch.where(
                            is_bedlam > 0,
                            gate_cov_thr.new_full(gate_cov_thr.shape, float(getattr(cfg, 'bedlam_hand_roi_coverage_thr', 0.6))),
                            gate_cov_thr,
                        )
                        gate_min_joints = torch.where(
                            is_bedlam > 0,
                            gate_min_joints.new_full(gate_min_joints.shape, float(getattr(cfg, 'bedlam_hand_roi_min_joints', 8))),
                            gate_min_joints,
                        )
                if 'is_ubody' in meta_info:
                    is_ubody = meta_info['is_ubody'].to(lhand_bbox.device).view(-1).float()
                    if getattr(cfg, 'ubody_use_hand_roi_quality', False):
                        gate_source = torch.maximum(gate_source, is_ubody)
                        gate_cov_thr = torch.where(
                            is_ubody > 0,
                            gate_cov_thr.new_full(gate_cov_thr.shape, float(getattr(cfg, 'ubody_hand_roi_coverage_thr', 0.6))),
                            gate_cov_thr,
                        )
                        gate_min_joints = torch.where(
                            is_ubody > 0,
                            gate_min_joints.new_full(gate_min_joints.shape, float(getattr(cfg, 'ubody_hand_roi_min_joints', 8))),
                            gate_min_joints,
                        )
                if 'dataset_id' in meta_info:
                    dataset_id = meta_info['dataset_id'].to(lhand_bbox.device).view(-1).float()
                    is_mscoco = (dataset_id == 3).float()
                    if getattr(cfg, 'mscoco_use_hand_roi_quality', False):
                        gate_source = torch.maximum(gate_source, is_mscoco)
                        gate_cov_thr = torch.where(
                            is_mscoco > 0,
                            gate_cov_thr.new_full(gate_cov_thr.shape, float(getattr(cfg, 'mscoco_hand_roi_coverage_thr', 0.6))),
                            gate_cov_thr,
                        )
                        gate_min_joints = torch.where(
                            is_mscoco > 0,
                            gate_min_joints.new_full(gate_min_joints.shape, float(getattr(cfg, 'mscoco_hand_roi_min_joints', 8))),
                            gate_min_joints,
                        )
                roi_gate_diag = {'coverage': [], 'ok': [], 'active': [], 'n_valid': []}
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
                    # Source-gated; InterHand (GT bbox) and disabled sources keep
                    # roi_ok=1.
                    if roi_gate_on:
                        gate_valid = gt_valid & coord_finite
                        n_valid = gate_valid.sum(dim=1)
                        coverage = (gate_valid & in_bbox.bool()).sum(dim=1).float() / n_valid.clamp(min=1).float()
                        roi_ok = ((bbox_valid > 0) &
                                  (n_valid.float() >= gate_min_joints) &
                                  (coverage >= gate_cov_thr)).float()
                        roi_ok = torch.where(gate_source > 0, roi_ok, torch.ones_like(roi_ok))
                        trunc = trunc * roi_ok[:, None]
                        roi_gate_diag['coverage'].append(coverage)
                        roi_gate_diag['ok'].append(roi_ok)
                        roi_gate_diag['active'].append(gate_source)
                        roi_gate_diag['n_valid'].append(n_valid.float())
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
            # End-tip 2D loss weighting (gated): boost distal hand joints in the
            # three hand-space 2D losses. No-op when the buffers are None. The
            # diagnostic _joint_img_* below slice the unweighted joint_img_loss /
            # smplx_joint_img_loss locals, so monitoring stays on raw values.
            if self.hand_loss_weight_full is not None:
                loss['joint_proj'] = loss['joint_proj'] * self.hand_loss_weight_full.view(1, -1, 1)
                loss['joint_img'] = loss['joint_img'] * self.hand_loss_weight_pos.view(1, -1, 1)
                loss['smplx_joint_img'] = loss['smplx_joint_img'] * self.hand_loss_weight_pos.view(1, -1, 1)
            if roi_gate_on:
                coverage = torch.stack(roi_gate_diag['coverage'], dim=1)
                roi_ok = torch.stack(roi_gate_diag['ok'], dim=1)
                active = torch.stack(roi_gate_diag['active'], dim=1)
                n_valid = torch.stack(roi_gate_diag['n_valid'], dim=1)
                active_denom = active.sum(dim=1, keepdim=True).clamp(min=1.0)
                loss['_hand_roi_gate_coverage'] = (coverage * active).sum(dim=1, keepdim=True) / active_denom
                loss['_hand_roi_gate_pass'] = (roi_ok * active).sum(dim=1, keepdim=True) / active_denom
                loss['_hand_roi_gate_active'] = active.mean(dim=1, keepdim=True)
                loss['_hand_roi_gate_n_valid'] = (n_valid * active).sum(dim=1, keepdim=True) / active_denom
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
            # Encoder-fidelity probe (gated): expose the encoder input + outputs so
            # the StandardViT port (this pytorch path) can be compared tensor-for-
            # tensor against the MMCV-ViT. No-op unless cfg.dump_encoder is True.
            if getattr(cfg, 'dump_encoder', False):
                out['_enc_input'] = body_img
                out['_enc_img_feat'] = img_feat
                out['_enc_task_tokens'] = task_tokens
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
