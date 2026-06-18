import torch
import torch.nn as nn

from common.utils.geometry import batch_rodrigues

class CoordLoss(nn.Module):
    def __init__(self):
        super(CoordLoss, self).__init__()

    def forward(self, coord_out, coord_gt, valid, is_3D=None):
        coord_out = torch.nan_to_num(coord_out, nan=0.0, posinf=0.0, neginf=0.0)
        coord_gt = torch.nan_to_num(coord_gt, nan=0.0, posinf=0.0, neginf=0.0)
        valid = torch.nan_to_num(valid, nan=0.0, posinf=0.0, neginf=0.0)
        loss = torch.abs(coord_out - coord_gt) * valid
        if is_3D is not None:
            loss_z = loss[:,:,2:] * is_3D[:,None,None].float()
            loss = torch.cat((loss[:,:,:2], loss_z),2)
        return loss

class ParamLoss(nn.Module):
    def __init__(self):
        super(ParamLoss, self).__init__()

    def forward(self, param_out, param_gt, valid):
        param_out = torch.nan_to_num(param_out, nan=0.0, posinf=0.0, neginf=0.0)
        param_gt = torch.nan_to_num(param_gt, nan=0.0, posinf=0.0, neginf=0.0)
        valid = torch.nan_to_num(valid, nan=0.0, posinf=0.0, neginf=0.0)
        loss = torch.abs(param_out - param_gt) * valid
        return loss

class RotMatParamLoss(nn.Module):
    """Rotation-matrix L1 loss for axis-angle pose parameters.

    Converts predicted and GT axis-angle to 3x3 rotation matrices (batch_rodrigues)
    and takes a masked L1 over the matrix entries. This is rotation-aware, unlike
    raw axis-angle L1 (which is discontinuous / non-unique near +-pi), so it gives
    cleaner gradients for articulated finger joints. Intended for the 30 SMPL-X
    hand finger joints only (see cfg.use_hand_rotmat_pose_loss); NOT a global
    ParamLoss replacement.

    Args:
        pose_out, pose_gt: (B, J, 3) axis-angle.
        valid:             (B, J, 1) or (B, J) per-joint validity mask.
    Returns:
        (B, J, 9) masked per-entry L1; reduce with .mean() like the other losses.
    """
    def __init__(self):
        super(RotMatParamLoss, self).__init__()

    def forward(self, pose_out, pose_gt, valid):
        batch, num_joints = pose_out.shape[0], pose_out.shape[1]
        pose_out = torch.nan_to_num(pose_out, nan=0.0, posinf=0.0, neginf=0.0)
        pose_gt = torch.nan_to_num(pose_gt, nan=0.0, posinf=0.0, neginf=0.0)
        rot_out = batch_rodrigues(pose_out.reshape(-1, 3)).reshape(batch, num_joints, 9)
        rot_gt = batch_rodrigues(pose_gt.reshape(-1, 3)).reshape(batch, num_joints, 9)
        if valid.dim() == 2:
            valid = valid[:, :, None]
        valid = torch.nan_to_num(valid, nan=0.0, posinf=0.0, neginf=0.0)
        loss = torch.abs(rot_out - rot_gt) * valid
        return loss

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, out, gt_index):
        loss = self.ce_loss(out, gt_index)
        return loss
