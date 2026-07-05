import os
import argparse
import torch.backends.cudnn as cudnn
from config import cfg
import torch

DEFAULT_PHASE1_EPOCHS = 10
HAND_WA_2D_VALID_SOURCES = {'ubody', 'mscoco', 'coco', 'interhand', 'all'}


def _split_csv_or_space(value):
    if value is None:
        return None
    if isinstance(value, str):
        raw_items = value.replace('，', ',').replace('、', ',').split(',')
    else:
        raw_items = []
        for item in value:
            raw_items.extend(str(item).replace('，', ',').replace('、', ',').split(','))
    return [item.strip() for item in raw_items if item.strip()]


def _parse_trainset_3d(value):
    names = _split_csv_or_space(value)
    if names is None:
        return None
    if not names:
        raise ValueError('--trainset_3d must contain at least one dataset name')
    invalid = [name for name in names if name not in cfg.dataset_list]
    if invalid:
        raise ValueError(
            'Invalid --trainset_3d dataset(s): %s; valid datasets are %s' % (
                ','.join(invalid), ','.join(cfg.dataset_list)))
    return names


def _parse_trainset_3d_sample_prob(value, trainset_3d):
    items = _split_csv_or_space(value)
    if items is None:
        return None
    if not items:
        raise ValueError('--trainset_3d_sample_prob must not be empty when provided')

    if any('=' in item for item in items):
        prob = {}
        for item in items:
            if '=' not in item:
                raise ValueError(
                    '--trainset_3d_sample_prob cannot mix name=value and positional values')
            name, raw_value = item.split('=', 1)
            name = name.strip()
            if name not in trainset_3d:
                raise ValueError(
                    '--trainset_3d_sample_prob includes %s, which is not in trainset_3d=%s' % (
                        name, ','.join(trainset_3d)))
            prob[name] = float(raw_value)
    else:
        if len(items) != len(trainset_3d):
            raise ValueError(
                '--trainset_3d_sample_prob positional values must match --trainset_3d length '
                '(%d values for %d datasets)' % (len(items), len(trainset_3d)))
        prob = {name: float(raw_value) for name, raw_value in zip(trainset_3d, items)}

    missing = [name for name in trainset_3d if name not in prob]
    if missing:
        raise ValueError(
            '--trainset_3d_sample_prob missing probabilities for: %s' % ','.join(missing))
    values = list(prob.values())
    if any(v < 0 for v in values) or sum(values) <= 0:
        raise ValueError('--trainset_3d_sample_prob values must be non-negative and sum to > 0')
    return prob


def _validate_trainset_3d_sample_prob(trainset_3d, prob_cfg, weighted_sampling):
    if not weighted_sampling or len(trainset_3d) <= 1:
        return
    if not prob_cfg:
        raise ValueError(
            'Weighted 3D dataset sampling is enabled for multiple trainset_3d entries, '
            'but trainset_3d_sample_prob is empty. Pass --trainset_3d_sample_prob or '
            '--disable_weighted_dataset_sampling.')
    missing = [name for name in trainset_3d if name not in prob_cfg]
    if missing:
        raise ValueError(
            'trainset_3d_sample_prob missing probabilities for %s. '
            'Pass --trainset_3d_sample_prob with all trainset_3d entries or '
            '--disable_weighted_dataset_sampling.' % ','.join(missing))


def _normalize_hand_wa_2d_sources(source_cfg):
    if isinstance(source_cfg, str):
        raw = source_cfg.replace('，', ',').replace('、', ',')
        sources = [s.strip().lower() for s in raw.split(',') if s.strip()]
    else:
        sources = [str(s).strip().lower() for s in source_cfg if str(s).strip()]

    invalid = sorted(set(sources) - HAND_WA_2D_VALID_SOURCES)
    if invalid:
        raise ValueError(
            'Invalid hand_wa_2d_loss_sources %s; valid sources are %s' % (
                ','.join(invalid), ','.join(sorted(HAND_WA_2D_VALID_SOURCES))))
    if not sources:
        raise ValueError('hand_wa_2d_loss_sources must contain at least one source')
    if 'all' in sources and len(set(sources)) > 1:
        raise ValueError("hand_wa_2d_loss_sources cannot combine 'all' with specific sources")
    return ','.join(sources)


def _source_masks(meta_info):
    if 'is_interhand' not in meta_info or 'is_bedlam' not in meta_info:
        return None

    masks = {
        'interhand': meta_info['is_interhand'].detach().view(-1).float(),
        'bedlam': meta_info['is_bedlam'].detach().view(-1).float(),
    }
    if 'is_ubody' in meta_info:
        masks['ubody'] = meta_info['is_ubody'].detach().view(-1).float()

    if 'dataset_id' in meta_info:
        dataset_id = meta_info['dataset_id'].detach().view(-1)
        masks['mscoco'] = (dataset_id == 3).float()
    else:
        known = torch.zeros_like(next(iter(masks.values())))
        for mask in masks.values():
            known = torch.maximum(known, mask.to(known.device))
        masks['mscoco'] = (1.0 - known.clamp(max=1.0)).clamp(min=0.0)

    return masks


def _per_source_losses(raw_loss, meta_info):
    """Split per-sample losses by dataset source for separate monitoring.

    raw_loss: dict of loss tensors still carrying the batch dim (dim 0), BEFORE
    any .mean(). Returns {'interhand': {...}, 'bedlam': {...}, 'ubody': {...}}
    plus {'mscoco': {...}} when present, where each value is the mean loss over
    the samples of that source in the batch. Returns None if the source flags are
    absent (e.g. single-dataset training).
    """
    masks = _source_masks(meta_info)
    if masks is None:
        return None
    batch_n = next(iter(masks.values())).numel()
    out = {src: {} for src in masks}
    for k, v in raw_loss.items():
        vd = v.detach()
        if vd.ndim == 0 or vd.shape[0] != batch_n:
            continue
        # collapse every dim except batch -> per-sample scalar
        per_sample = vd.reshape(vd.shape[0], -1).mean(dim=1)
        for src_name, mask in masks.items():
            # meta_info flags may live on CPU under DataParallel; align device.
            m = mask.to(per_sample.device)
            denom = m.sum()
            if denom.item() > 0:
                out[src_name][k] = ((per_sample * m).sum() / denom).item()
    return out


def _raise_nonfinite_loss(loss, total_loss, meta_info, logger, epoch, itr):
    if torch.isfinite(total_loss):
        return

    loss_state = []
    for k, v in loss.items():
        value = v.detach()
        if torch.isfinite(value):
            loss_state.append('%s=%.6g' % (k, value.item()))
        else:
            loss_state.append('%s=%s' % (k, value.detach().cpu().item()))

    batch_state = []
    source_masks = _source_masks(meta_info)
    if source_masks is not None:
        for src_name, mask in source_masks.items():
            batch_state.append('%s=%.0f' % (src_name, mask.sum().detach().cpu().item()))

    msg = 'Non-finite total_loss before backward at epoch %d itr %d. %s %s' % (
        epoch,
        itr,
        ' '.join(batch_state),
        ' '.join(loss_state),
    )
    logger.error(msg)
    raise RuntimeError(msg)


def _raise_nonfinite_trainable_params(model, logger, epoch, itr):
    bad = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        data = param.detach()
        if torch.isfinite(data).all():
            continue
        n_bad = (~torch.isfinite(data)).sum().item()
        bad.append((name, n_bad, data.numel()))
        if len(bad) >= 10:
            break

    if not bad:
        return

    msg = 'Non-finite trainable parameters after optimizer step at epoch %d itr %d. %s' % (
        epoch,
        itr,
        ' '.join('%s=%d/%d' % item for item in bad),
    )
    logger.error(msg)
    raise RuntimeError(msg)


def _collect_nonfinite_trainable_grads(model, limit=10):
    """Return [(name, n_bad, numel), ...] for trainable params with non-finite
    grads (capped at `limit`); empty list = all finite. Non-raising, so callers
    can either abort (strict) or skip the batch."""
    bad = []
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        grad = param.grad.detach()
        if torch.isfinite(grad).all():
            continue
        n_bad = (~torch.isfinite(grad)).sum().item()
        bad.append((name, n_bad, grad.numel()))
        if len(bad) >= limit:
            break
    return bad


def _raise_nonfinite_trainable_grads(model, logger, epoch, itr, stage):
    bad = _collect_nonfinite_trainable_grads(model)
    if not bad:
        return

    msg = 'Non-finite trainable gradients %s at epoch %d itr %d. %s' % (
        stage,
        epoch,
        itr,
        ' '.join('%s=%d/%d' % item for item in bad),
    )
    logger.error(msg)
    raise RuntimeError(msg)


def _clip_grad_norm_checked(params, max_norm, group_name, logger, epoch, itr):
    params = list(params)
    if not params:
        return None
    try:
        return torch.nn.utils.clip_grad_norm_(
            params,
            max_norm=max_norm,
            error_if_nonfinite=True,
        )
    except RuntimeError as exc:
        finite_stats = []
        for param in params:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if not torch.isfinite(grad).all():
                continue
            finite_stats.append(float(grad.abs().max().detach().cpu()))
        max_abs = max(finite_stats) if finite_stats else float('nan')
        msg = 'Non-finite grad norm while clipping %s at epoch %d itr %d. max_finite_abs_grad=%s. %s' % (
            group_name,
            epoch,
            itr,
            max_abs,
            str(exc),
        )
        logger.error(msg)
        raise RuntimeError(msg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, dest='gpu_ids')
    parser.add_argument('--lr', type=str, dest='lr', default="1e-4")
    parser.add_argument('--lr_mult', type=float, default=0.1,
                        help='回归网络的学习率倍率 (lr * lr_mult)')
    parser.add_argument('--continue_train', dest='continue_train', action='store_true')
    parser.add_argument('--exp_name', type=str, default='output/dcnv4')
    parser.add_argument('--num_thread', type=int, default=4)
    parser.add_argument('--end_epoch', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--trainset_3d', nargs='+', default=None,
                        help='覆盖 cfg.trainset_3d；支持空格或逗号分隔，如 '
                             '--trainset_3d BEDLAM InterHand26M UBody 或 '
                             '--trainset_3d BEDLAM,InterHand26M,UBody')
    parser.add_argument('--trainset_3d_sample_prob', nargs='+', default=None,
                        help='覆盖 cfg.trainset_3d_sample_prob；支持 name=value 或按 '
                             '--trainset_3d 顺序给值，如 '
                             '--trainset_3d_sample_prob BEDLAM=0.4,InterHand26M=0.4,UBody=0.2 '
                             '或 --trainset_3d_sample_prob 0.4 0.4 0.2')
    parser.add_argument('--encoder_setting', type=str, default='osx_l', choices=['osx_b', 'osx_l'])
    parser.add_argument('--decoder_setting', type=str, default='pytorch', choices=['normal', 'wo_face_decoder', 'wo_decoder', 'pytorch'])
    parser.add_argument('--agora_benchmark', action='store_true')
    parser.add_argument('--ubody_benchmark', action='store_true')
    parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_models/osx_l.pth.tar')
    parser.add_argument('--continue_train_path', type=str, default="")
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='梯度裁剪最大范数')
    parser.add_argument('--posnet_lr_mult', type=float, default=0.25,
                        help='hand_position_net 单独学习率倍率 (lr * posnet_lr_mult)，'
                             '稳定 soft-argmax 深度分支防止中途发散')
    parser.add_argument('--posnet_grad_clip', type=float, default=0.5,
                        help='hand_position_net 单独梯度裁剪范数 (<=0 关闭)')
    parser.add_argument('--skip_nonfinite_grad', action='store_true',
                        help='backward 后若出现非有限梯度，跳过该 iter（zero_grad、不 step）而非中止训练；'
                             '默认关闭=旧的严格 abort。用于绕过 DCNv4 采样反向偶发 NaN 的坏 batch')
    parser.add_argument('--phase1_epochs', type=int, default=DEFAULT_PHASE1_EPOCHS,
                        help='手部微调 Phase 1 的 epoch 数；默认覆盖 10 epoch 短程混训')
    parser.add_argument('--save_iters', type=int, default=0,
                        help='>0 时在每个 epoch 中途每 N 个 iter 额外保存一个 '
                             'snapshot_{epoch}_itr{N}.pth.tar（评估候选点，与 epoch 末快照'
                             '分开命名、互不覆盖）；0=关闭，保持原有每 epoch 保存')
    parser.add_argument('--train_body_shape', action='store_true',
                        help='Body-shape T1 微实验：解冻 body_regressor.shape_out+cam_out（其余 '
                             'body_regressor 仍冻结），在 BEDLAM GT betas 上微调体型/相机；'
                             '默认关闭，关闭时对手/脸训练零影响')
    parser.add_argument('--body_shape_lr', type=float, default=1e-5,
                        help='body-shape T1 的 shape_out/cam_out 学习率（仅 --train_body_shape 时生效）')
    parser.add_argument('--body_shape_mode', type=str, default='both',
                        choices=['both', 'shape', 'cam'],
                        help='T1 shape/cam 拆分：解冻哪个头（both=shape_out+cam_out 原始 T1；'
                             'shape=只 shape_out；cam=只 cam_out）；仅 --train_body_shape 时生效')
    parser.add_argument('--use_poseur_hand_decoder', action='store_true',
                        help='decoder 隔离实验：手部用 OSX 式 PoseurDecoder(6 层迭代精修) 替代 '
                             'HandDecoder；默认关闭=HandDecoder。train 与 test1 必须一致')
    parser.add_argument('--train_hand_roi', action='store_true',
                        help='路线 C：解冻 hand_roi_net，与 hand decoder/regressor 小 LR 共适应'
                             '（box_net 仍冻结）；默认关闭，关闭时对现有训练零影响')
    parser.add_argument('--hand_roi_lr', type=float, default=1e-6,
                        help='hand_roi_net 的学习率（仅 --train_hand_roi 时生效，建议 1e-6~5e-6）')
    parser.add_argument('--hand_tip_loss_weight', type=float, default=1.0,
                        help='末端 2D loss 加权：手指尖(_4)在 hand-space 2D loss(joint_proj/joint_img/'
                             'smplx_joint_img)的权重；默认 1.0=关闭')
    parser.add_argument('--hand_j3_loss_weight', type=float, default=1.0,
                        help='手指 j3(_3)在 hand-space 2D loss 的权重；默认 1.0=关闭')
    parser.add_argument('--hand_wa_2d_loss_weight', type=float, default=0.0,
                        help='direct wrist-aligned hand 2D loss 权重；在 full-body heatmap 坐标中'
                             '按 GT 手 bbox diag 归一化，默认 0=关闭')
    parser.add_argument('--hand_wa_2d_loss_sources', type=str, default='ubody,mscoco',
                        help='启用 direct wrist-aligned hand 2D loss 的数据源，逗号分隔；'
                             '可选 ubody,mscoco,interhand，默认 ubody,mscoco')
    parser.add_argument('--hand_wa_2d_loss_min_joints', type=int, default=4,
                        help='direct wrist-aligned hand 2D loss 每只手至少需要的可见关节点数')
    parser.add_argument('--hand_wa_2d_j1_weight', type=float, default=1.0,
                        help='direct wrist-aligned hand 2D loss 中每根手指 j1 的权重；默认 1.0')
    parser.add_argument('--hand_wa_2d_j2_weight', type=float, default=1.0,
                        help='direct wrist-aligned hand 2D loss 中每根手指 j2 的权重；默认 1.0')
    parser.add_argument('--hand_wa_2d_j3_weight', type=float, default=1.0,
                        help='direct wrist-aligned hand 2D loss 中每根手指 j3 的权重；默认 1.0')
    parser.add_argument('--hand_wa_2d_tip_weight', type=float, default=1.0,
                        help='direct wrist-aligned hand 2D loss 中每根手指 tip 的权重；默认 1.0')
    parser.add_argument('--hand_wa_2d_min_diag', type=float, default=1.0,
                        help='WA2D backward NaN 守卫：GT 手 bbox 对角线(output_hm_shape 单位)的下限，'
                             '每关节梯度=1/diag，太小会爆 NaN；默认 1.0(每指梯度<=1)，1e-4=旧危险行为')
    parser.add_argument('--hand_wa_2d_err_clip', type=float, default=5.0,
                        help='WA2D 每关节归一化误差上限(hard clamp)，防单个坏关节主导反向；默认 5.0，0=关闭')
    parser.add_argument('--train_face_modules', action='store_true',
                        help='默认冻结 face 分支；需要 UBody face/expression 微调时显式开启')
    parser.add_argument('--no_train_hand_modules', action='store_true',
                        help='冻结手部分支（face-only 阶段用）：手部三件套不进优化器、'
                             '保持已加载的手权重不动；其名字仍在 snapshot 里保存，'
                             '便于后续联合阶段暖启动')
    parser.add_argument('--freeze_hand_position_net', action='store_true',
                        help='WA2D polish 稳定项：冻结已 warm-start 的 hand_position_net，'
                             '只训练 hand_decoder/hand_regressor，避免 DCNv4 position head '
                             '在自然手 batch 上反传出非有限梯度')
    parser.add_argument('--init_trained_path', type=str, default='',
                        help='非续训(continue_train=False)时，额外暖加载一个 lightweight '
                             'snapshot 的已训练手/脸模块（不续 epoch/optimizer）；用于阶段衔接')
    parser.add_argument('--no_phase1_hand_regressor', action='store_true',
                        help='Phase 1 不训练 hand_regressor，仅用于对比旧热身策略')
    parser.add_argument('--bedlam_max_samples', type=int, default=None)
    parser.add_argument('--interhand_max_samples', type=int, default=None)
    parser.add_argument('--bedlam_use_hand_roi_quality', action='store_true',
                        help='对 BEDLAM 启用动态手部 ROI coverage 门控，低覆盖手只屏蔽 hand-space 2D loss')
    parser.add_argument('--ubody_use_hand_roi_quality', action='store_true',
                        help='对 UBody 启用动态手部 ROI coverage 门控；Stage 3 joint polish 的保护项，'
                             '不会像 BEDLAM B 类修复那样整源切断 hand img loss')
    parser.add_argument('--mscoco_use_hand_roi_quality', action='store_true',
                        help='对 MSCOCO/COCO-WholeBody 启用动态手部 ROI coverage 门控；'
                             '与 UBody 使用同一类自然图手部保护逻辑')
    parser.add_argument('--ubody_hand_roi_coverage_thr', type=float, default=None,
                        help='UBody 手部 ROI coverage 门控阈值；默认沿用 cfg，Stage 3 前需短跑确认')
    parser.add_argument('--ubody_hand_roi_min_joints', type=int, default=None,
                        help='UBody 手部 ROI 门控所需最少 GT-valid 手关节点数；默认沿用 cfg，Stage 3 前需短跑确认')
    parser.add_argument('--mscoco_hand_roi_coverage_thr', type=float, default=None,
                        help='MSCOCO 手部 ROI coverage 门控阈值；默认沿用 cfg')
    parser.add_argument('--mscoco_hand_roi_min_joints', type=int, default=None,
                        help='MSCOCO 手部 ROI 门控所需最少 GT-valid 手关节点数；默认沿用 cfg')
    parser.add_argument('--bedlam_no_hand_img_loss', action='store_true',
                        help='BEDLAM 不监督手部 joint_img/smplx_joint_img（soft-argmax 热图头的'
                             '唯一梯度源），切断对共享 hand_position_net 的梯度污染；'
                             '保留 BEDLAM 手部 joint_proj 与全部 SMPL-X/3D/pose 监督，InterHand 不受影响')
    parser.add_argument('--use_hand_rotmat_pose_loss', action='store_true',
                        help='额外对 30 个手指关节加 rotation-matrix pose loss（旋转感知，比 '
                             'axis-angle L1 更稳）；默认关闭，不影响现有训练')
    parser.add_argument('--hand_rotmat_pose_loss_weight', type=float, default=None,
                        help='rotation-matrix hand pose loss 权重（默认沿用 cfg 的 1.0）')
    parser.add_argument('--hand_aux_coord_loss_weight', type=float, default=None,
                        help='decoder 每层 aux 2D 坐标监督权重（默认沿用 cfg 的 1.0）；设 0 关闭 aux，'
                             '测试 2D 坐标监督是否与 3D hand pose 抢梯度/容量（见 experiment_log 7-03）')
    parser.add_argument('--no_refined_hand_coord', action='store_true',
                        help='关闭 L728：不把 decoder 末层精修坐标喂 hand_regressor，回退到 soft-argmax '
                             'hand_joint_img（默认 cfg.use_refined_hand_coord=True）；7-03 L728 判负后的复位开关')
    parser.add_argument('--disable_weighted_dataset_sampling', action='store_true')
    args = parser.parse_args()
    return args


def _set_module_trainable(core_model, module_names, trainable):
    for module_name in module_names:
        module = getattr(core_model, module_name)
        module.train(trainable)
        for param in module.parameters():
            param.requires_grad = trainable


def _grad_params_for_groups(optimizer, group_names):
    return [
        p
        for group in optimizer.param_groups
        if group.get('name') in group_names
        for p in group['params']
        if p.requires_grad and p.grad is not None
    ]


def _collect_module_params(core_model, module_names):
    params = []
    for module_name in module_names:
        module = getattr(core_model, module_name)
        params.extend([p for p in module.parameters() if p.requires_grad])
    return params


def _configure_training_phase(trainer, phase, remaining_epochs):
    core_model = trainer.model.module
    core_model.set_training_phase(phase)

    candidate_modules = [
        'hand_position_net',
        'hand_decoder',
        'hand_regressor',
        'face_position_net',
        'face_decoder',
        'face_regressor',
    ]
    # Route C (gated): hand_roi_net joins the managed set so it is reset/toggled
    # like the others and gets its own small-lr optimizer group below.
    hand_roi_modules = ['hand_roi_net'] if getattr(cfg, 'train_hand_roi', False) else []
    candidate_modules = candidate_modules + hand_roi_modules

    # hand_position_net is split into its own optimizer group at a reduced lr
    # (A-class stabilization): it is the soft-argmax module that diverges
    # mid-epoch, so a persistently smaller step protects it long after any
    # warmup window would have closed.
    posnet_modules = []
    normal_modules = []
    special_modules = []
    phase_scale = 1.0 if phase == 1 else 0.1
    eta_min = 1e-6 if phase == 1 else 1e-7
    # Hand branch is gated by cfg.train_hand_modules (default True). When False
    # (face-only stage) the hand modules are frozen elsewhere and simply never
    # enter the optimizer here. Default path is unchanged.
    if getattr(cfg, 'train_hand_modules', True):
        if not getattr(cfg, 'freeze_hand_position_net', False):
            posnet_modules.append('hand_position_net')
        normal_modules.append('hand_decoder')
        if phase == 1:
            if cfg.phase1_train_hand_regressor:
                special_modules.append('hand_regressor')
        else:
            special_modules.append('hand_regressor')
    normal_lr = cfg.lr * phase_scale
    special_lr = cfg.lr * cfg.lr_mult * phase_scale
    posnet_lr = cfg.lr * cfg.posnet_lr_mult * phase_scale

    if cfg.train_face_modules:
        posnet_modules.append('face_position_net')
        normal_modules.append('face_decoder')
        special_modules.append('face_regressor')

    trainable_modules = posnet_modules + normal_modules + special_modules + hand_roi_modules
    _set_module_trainable(core_model, candidate_modules, False)
    _set_module_trainable(core_model, trainable_modules, True)

    optim_params = []
    posnet_params = _collect_module_params(core_model, posnet_modules)
    normal_params = _collect_module_params(core_model, normal_modules)
    special_params = _collect_module_params(core_model, special_modules)
    if posnet_params:
        optim_params.append({'params': posnet_params, 'lr': posnet_lr, 'name': 'position_nets'})
    if normal_params:
        optim_params.append({'params': normal_params, 'lr': normal_lr, 'name': 'hand_face_new_modules'})
    if special_params:
        optim_params.append({'params': special_params, 'lr': special_lr, 'name': 'regressors'})

    # Body-shape T1 (gated): shape_out/cam_out (requires_grad set in
    # Model.freeze_modules) get their own optimizer group. No-op when the prefix
    # list is empty, so the hand/face optimizer layout is unchanged.
    body_shape_prefixes = getattr(core_model, 'body_shape_trainable_prefixes', [])
    if body_shape_prefixes:
        body_shape_params = [
            p for n, p in core_model.named_parameters()
            if p.requires_grad and any(n.startswith(pre) for pre in body_shape_prefixes)
        ]
        if body_shape_params:
            optim_params.append({'params': body_shape_params,
                                 'lr': getattr(cfg, 'body_shape_lr', 1e-5),
                                 'name': 'body_shape'})

    # Route C (gated): hand_roi_net co-adapts at its own small lr (cfg.hand_roi_lr).
    # _set_module_trainable above already re-enabled its grads. No-op when off.
    if hand_roi_modules:
        hand_roi_params = _collect_module_params(core_model, hand_roi_modules)
        if hand_roi_params:
            optim_params.append({'params': hand_roi_params,
                                 'lr': getattr(cfg, 'hand_roi_lr', 1e-6),
                                 'name': 'hand_roi'})

    if not optim_params:
        raise RuntimeError("No trainable parameters were selected for phase %d" % phase)

    trainer.optimizer = torch.optim.Adam(optim_params)
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer,
        max(1, remaining_epochs * trainer.itr_per_epoch),
        eta_min=eta_min,
    )

    # Restore resume optimizer state stashed by _load_resume_checkpoint, but only
    # when the checkpoint was saved in the SAME phase we are now entering — phases
    # have different param-group counts/structure, so a cross-phase load would
    # mismatch. Consumed once (cleared) so later phase switches start fresh.
    if getattr(trainer, 'resume_optimizer_state', None) is not None:
        phase1_end_epoch = min(max(int(cfg.phase1_epochs), 0), cfg.end_epoch)
        ckpt_phase = 1 if trainer.resume_epoch < phase1_end_epoch else 2
        if ckpt_phase == phase:
            try:
                trainer.optimizer.load_state_dict(trainer.resume_optimizer_state)
                trainer.logger.info("  ✅ 已恢复 phase %d 的 optimizer 状态" % phase)
            except Exception as exc:
                trainer.logger.warning("  ⚠️ optimizer 状态恢复失败（分组不匹配？）: %s" % exc)
        else:
            trainer.logger.info(
                "  ⏭️ checkpoint 属于 phase %d，当前进入 phase %d，跳过 optimizer 恢复" % (
                    ckpt_phase, phase))
        trainer.resume_optimizer_state = None
        trainer.resume_epoch = None

    trainer.logger.info("=== Phase %d: epoch %d, remaining_epochs=%d ===" % (
        phase,
        trainer.cur_epoch,
        remaining_epochs,
    ))
    trainer.logger.info("  trainable modules: %s" % ", ".join(trainable_modules))
    for group in trainer.optimizer.param_groups:
        trainer.logger.info(
            "  optimizer group %s: %d params, lr=%g" % (
                group.get('name', 'unnamed'),
                sum(p.numel() for p in group['params']),
                group['lr'],
            )
        )


def main():
    print('### Argument parse and create log ###')
    args = parse_args()
    trainset_3d = _parse_trainset_3d(args.trainset_3d)
    if trainset_3d is None:
        trainset_3d = cfg.trainset_3d
    trainset_3d_sample_prob = _parse_trainset_3d_sample_prob(
        args.trainset_3d_sample_prob,
        trainset_3d,
    )
    cfg.set_args(args.gpu_ids, args.lr, args.continue_train)
    cfg.set_additional_args(
        exp_name=args.exp_name,
        num_thread=args.num_thread,
        train_batch_size=args.train_batch_size,
        trainset_3d=trainset_3d,
        encoder_setting=args.encoder_setting,
        decoder_setting=args.decoder_setting,
        end_epoch=args.end_epoch,
        pretrained_model_path=args.pretrained_model_path,
        agora_benchmark=args.agora_benchmark,
        ubody_benchmark=args.ubody_benchmark,
        lr_mult=args.lr_mult,
        phase1_epochs=args.phase1_epochs,
        phase1_train_hand_regressor=not args.no_phase1_hand_regressor,
        train_face_modules=args.train_face_modules,
        train_hand_modules=not args.no_train_hand_modules,
        freeze_hand_position_net=args.freeze_hand_position_net,
        use_weighted_dataset_sampling=not args.disable_weighted_dataset_sampling,
        posnet_lr_mult=args.posnet_lr_mult,
        posnet_grad_clip=args.posnet_grad_clip,
    )
    if trainset_3d_sample_prob is not None:
        cfg.trainset_3d_sample_prob = trainset_3d_sample_prob
    _validate_trainset_3d_sample_prob(
        cfg.trainset_3d,
        getattr(cfg, 'trainset_3d_sample_prob', None),
        getattr(cfg, 'use_weighted_dataset_sampling', False),
    )
    print('>>> trainset_3d: %s' % ','.join(cfg.trainset_3d))
    if getattr(cfg, 'trainset_3d_sample_prob', None):
        prob_str = ','.join(
            '%s=%s' % (name, cfg.trainset_3d_sample_prob.get(name, 'NA'))
            for name in cfg.trainset_3d
        )
        print('>>> trainset_3d_sample_prob: %s' % prob_str)

    if cfg.decoder_setting not in ('pytorch', 'normal'):
        raise ValueError(
            "Training phase/freeze code supports --decoder_setting pytorch (main path) "
            "or normal (track-B OSX architecture-controlled fair baseline); got %s"
            % cfg.decoder_setting)

    if args.bedlam_max_samples is not None:
        cfg.bedlam_max_samples = args.bedlam_max_samples
    if args.interhand_max_samples is not None:
        cfg.interhand_max_samples = args.interhand_max_samples
    if args.bedlam_use_hand_roi_quality:
        cfg.bedlam_use_hand_roi_quality = True
    if args.ubody_use_hand_roi_quality:
        cfg.ubody_use_hand_roi_quality = True
    if args.mscoco_use_hand_roi_quality:
        cfg.mscoco_use_hand_roi_quality = True
    if args.ubody_hand_roi_coverage_thr is not None:
        cfg.ubody_hand_roi_coverage_thr = args.ubody_hand_roi_coverage_thr
    if args.ubody_hand_roi_min_joints is not None:
        cfg.ubody_hand_roi_min_joints = args.ubody_hand_roi_min_joints
    if args.mscoco_hand_roi_coverage_thr is not None:
        cfg.mscoco_hand_roi_coverage_thr = args.mscoco_hand_roi_coverage_thr
    if args.mscoco_hand_roi_min_joints is not None:
        cfg.mscoco_hand_roi_min_joints = args.mscoco_hand_roi_min_joints
    if args.bedlam_no_hand_img_loss:
        cfg.bedlam_supervise_hand_img = False
    if args.use_hand_rotmat_pose_loss:
        cfg.use_hand_rotmat_pose_loss = True
    if args.hand_rotmat_pose_loss_weight is not None:
        cfg.hand_rotmat_pose_loss_weight = args.hand_rotmat_pose_loss_weight
    if args.hand_aux_coord_loss_weight is not None:
        cfg.hand_aux_coord_loss_weight = args.hand_aux_coord_loss_weight
    if args.no_refined_hand_coord:
        cfg.use_refined_hand_coord = False
    if args.init_trained_path:
        cfg.init_trained_path = args.init_trained_path

    # Body-shape T1 (gated). Must be set BEFORE Trainer()/_make_model builds the
    # Model, which reads cfg.train_body_shape to populate the unfreeze prefixes.
    if args.train_body_shape:
        cfg.train_body_shape = True
    cfg.body_shape_lr = args.body_shape_lr
    cfg.body_shape_mode = args.body_shape_mode
    # Decoder isolation experiment: must be set BEFORE _make_model (get_model reads
    # cfg.use_poseur_hand_decoder to pick HandDecoder vs PoseurDecoder).
    cfg.use_poseur_hand_decoder = args.use_poseur_hand_decoder

    # Route C (gated). Must be set BEFORE _make_model builds the Model, which
    # reads cfg.train_hand_roi to move hand_roi_net into the trainable lists.
    if args.train_hand_roi:
        cfg.train_hand_roi = True
    cfg.hand_roi_lr = args.hand_roi_lr

    # End-tip 2D loss weighting (gated). Must be set BEFORE _make_model: the Model
    # reads these in __init__ to build the per-joint weight buffers.
    cfg.hand_tip_loss_weight = args.hand_tip_loss_weight
    cfg.hand_j3_loss_weight = args.hand_j3_loss_weight
    cfg.hand_wa_2d_loss_weight = args.hand_wa_2d_loss_weight
    cfg.hand_wa_2d_loss_sources = _normalize_hand_wa_2d_sources(args.hand_wa_2d_loss_sources)
    cfg.hand_wa_2d_loss_min_joints = args.hand_wa_2d_loss_min_joints
    cfg.hand_wa_2d_min_diag = args.hand_wa_2d_min_diag
    cfg.hand_wa_2d_err_clip = args.hand_wa_2d_err_clip
    cfg.skip_nonfinite_grad = args.skip_nonfinite_grad
    wa_level_weights = {
        'hand_wa_2d_j1_weight': args.hand_wa_2d_j1_weight,
        'hand_wa_2d_j2_weight': args.hand_wa_2d_j2_weight,
        'hand_wa_2d_j3_weight': args.hand_wa_2d_j3_weight,
        'hand_wa_2d_tip_weight': args.hand_wa_2d_tip_weight,
    }
    for name, value in wa_level_weights.items():
        if not value >= 0:
            raise ValueError('%s must be non-negative, got %s' % (name, value))
        setattr(cfg, name, value)
    if sum(wa_level_weights.values()) <= 0:
        raise ValueError('At least one hand_wa_2d level weight must be positive')

    if args.continue_train and args.continue_train_path:
        cfg.continue_train_path = args.continue_train_path

    print(f"cfg.lr_mult = {cfg.lr_mult}")
    

    cudnn.benchmark = True

    from common.base import Trainer
    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()

    print('### Set some hyper parameters ###')
    for k in cfg.__dict__:
        trainer.logger.info(f'set {k} to {cfg.__dict__[k]}')

    # ================================================================
    #  打印训练配置摘要
    # ================================================================
    trainer.logger.info('\n' + '=' * 60)
    trainer.logger.info('训练配置摘要:')
    trainer.logger.info(f'  🧊 冻结模块: {trainer.model.module.frozen_modules}')
    trainer.logger.info(f'  🔥 训练模块: {trainer.model.module.trainable_module_names}')
    trainer.logger.info(f'  📈 学习率: {cfg.lr} (新模块), '
                        f'{cfg.lr * cfg.lr_mult} (回归网络)')
    trainer.logger.info(f'  Phase 1 epochs: {cfg.phase1_epochs}, '
                        f'phase1 hand_regressor: {cfg.phase1_train_hand_regressor}, '
                        f'train face modules: {cfg.train_face_modules}, '
                        f'freeze hand position net: {cfg.freeze_hand_position_net}')
    trainer.logger.info(f'  📦 Batch size: {cfg.train_batch_size} x {cfg.num_gpus} GPU')
    trainer.logger.info(f'  🔄 Epochs: {trainer.start_epoch} → {cfg.end_epoch}')
    trainer.logger.info('=' * 60 + '\n')

    # ================================================================
    #  训练循环
    # ================================================================
    print('### Start training ###')
    phase1_end_epoch = min(max(int(cfg.phase1_epochs), 0), cfg.end_epoch)
    active_phase = None
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        trainer.cur_epoch = epoch
        phase = 1 if epoch < phase1_end_epoch else 2
        if phase != active_phase:
            remaining_epochs = (phase1_end_epoch - epoch) if phase == 1 else (cfg.end_epoch - epoch)
            _configure_training_phase(trainer, phase, remaining_epochs)
            active_phase = phase

        trainer.tot_timer.tic()
        trainer.read_timer.tic()

        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # ---- Forward ----
            trainer.optimizer.zero_grad()
            loss = trainer.model(inputs, targets, meta_info, 'train')
            # split per-sample losses by dataset source before reducing to scalars
            per_source = None
            if (itr + 1) % cfg.print_iters == 0:
                per_source = _per_source_losses(loss, meta_info)
            loss = {k: loss[k].mean() for k in loss}

            # ---- Backward ----
            # Keys prefixed '_' are diagnostics (e.g. xy/z splits) and must NOT
            # be added to the backward objective, or they would double-count.
            total_loss = sum(loss[k] for k in loss if not k.startswith('_'))
            _raise_nonfinite_loss(loss, total_loss, meta_info, trainer.logger, epoch, itr)
            total_loss.backward()

            # Non-finite grads after backward: abort (strict, default) or skip the
            # iter (--skip_nonfinite_grad) to survive the occasional DCNv4-backward
            # NaN batch without killing the whole run.
            skip_iter = False
            if getattr(cfg, 'skip_nonfinite_grad', False):
                bad_grads = _collect_nonfinite_trainable_grads(trainer.model)
                if bad_grads:
                    skip_iter = True
                    trainer.skip_nonfinite_count = getattr(trainer, 'skip_nonfinite_count', 0) + 1
                    trainer.logger.warning(
                        'Skip itr %d (epoch %d): non-finite grads after backward, '
                        'skipped=%d so far. %s' % (
                            itr, epoch, trainer.skip_nonfinite_count,
                            ' '.join('%s=%d/%d' % b for b in bad_grads[:4])))
                    trainer.optimizer.zero_grad(set_to_none=True)
            else:
                _raise_nonfinite_trainable_grads(trainer.model, trainer.logger, epoch, itr, 'after backward')

            if not skip_iter:
                # ---- 梯度裁剪（防止新模块初期梯度爆炸）----
                # position_nets (soft-argmax, diverges mid-epoch) get a separate,
                # tighter clip; everything else uses the global grad_clip. Clipping
                # the two disjoint sets independently avoids double-clipping posnet.
                if cfg.posnet_grad_clip > 0:
                    posnet_params = _grad_params_for_groups(trainer.optimizer, {'position_nets'})
                    if posnet_params:
                        _clip_grad_norm_checked(
                            posnet_params,
                            cfg.posnet_grad_clip,
                            'position_nets',
                            trainer.logger,
                            epoch,
                            itr,
                        )
                if args.grad_clip > 0:
                    other_params = _grad_params_for_groups(
                        trainer.optimizer, {'hand_face_new_modules', 'regressors', 'body_shape', 'hand_roi'})
                    if other_params:
                        _clip_grad_norm_checked(
                            other_params,
                            args.grad_clip,
                            'non_position_groups',
                            trainer.logger,
                            epoch,
                            itr,
                        )
                _raise_nonfinite_trainable_grads(trainer.model, trainer.logger, epoch, itr, 'after clipping')

                trainer.optimizer.step()
                _raise_nonfinite_trainable_params(trainer.model, trainer.logger, epoch, itr)
            trainer.scheduler.step()
            trainer.gpu_timer.toc()

            # ---- 日志 ----
            if (itr + 1) % cfg.print_iters == 0:
                screen = [
                    'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                    'lr: %g' % (trainer.get_lr()),
                    'speed: %.2f(%.2fs r%.2f)s/itr' % (
                        trainer.tot_timer.average_time,
                        trainer.gpu_timer.average_time,
                        trainer.read_timer.average_time,
                    ),
                    '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                ]
                source_masks = _source_masks(meta_info)
                if source_masks is not None:
                    screen += [
                        'batch_interhand: %.0f' % source_masks['interhand'].sum().detach().cpu().item(),
                        'batch_bedlam: %.0f' % source_masks['bedlam'].sum().detach().cpu().item(),
                    ]
                    if 'ubody' in source_masks:
                        screen += ['batch_ubody: %.0f' % source_masks['ubody'].sum().detach().cpu().item()]
                    screen += ['batch_mscoco: %.0f' % source_masks['mscoco'].sum().detach().cpu().item()]
                screen += ['%s: %.4f' % ('loss_' + k.lstrip('_'), v.detach()) for k, v in loss.items()]
                trainer.logger.info(' '.join(screen))
                if per_source is not None:
                    for src_name in ('interhand', 'bedlam', 'ubody', 'mscoco'):
                        src_losses = per_source.get(src_name, {})
                        if src_losses:
                            line = ['  [%s]' % src_name] + [
                                '%s: %.4f' % (k.lstrip('_'), val) for k, val in src_losses.items()
                            ]
                            trainer.logger.info(' '.join(line))

            # ---- 中途保存（评估候选点）----
            # save_iters>0 时每 N 个 iter 存一个 snapshot_{epoch}_itr{N}，与 epoch 末
            # 快照分开命名、互不覆盖；定位为评估候选点（state['epoch'] 仍为当前 epoch，
            # 故若拿来续训会从下一 epoch 起步，续训粒度仍是 epoch）。
            if args.save_iters > 0 and (itr + 1) % args.save_iters == 0 \
                    and (itr + 1) < trainer.itr_per_epoch:
                trainer.save_model({
                    'epoch': epoch,
                    'network': trainer.model.state_dict(),
                    'optimizer': trainer.optimizer.state_dict(),
                }, '%d_itr%d' % (epoch, itr + 1))

            # ---- 首个 iteration 检查梯度流 ----
            if epoch == trainer.start_epoch and itr == 0:
                _check_gradient_flow(trainer)

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        # ---- 每 epoch 可视化检查面板 (InterHand/BEDLAM/UBody, if present) ----
        if getattr(cfg, 'vis_epoch_panel', True) and getattr(trainer, 'trainset_by_name', None):
            try:
                from common.utils.train_vis import save_epoch_panels
                save_epoch_panels(
                    trainer.model,
                    trainer.trainset_by_name,
                    epoch,
                    os.path.join(cfg.vis_dir, 'epoch_panel'),
                    logger=trainer.logger,
                )
            except Exception as exc:
                trainer.logger.info('[epoch-vis] skipped: %s' % exc)

        # ---- 保存模型 ----
        if epoch % 1 == 0 or epoch == (cfg.end_epoch - 1):
            trainer.save_model({
                'epoch': epoch,
                'network': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
            }, epoch)


def _check_gradient_flow(trainer):
    """
    第一个 iteration 后检查梯度流向是否正确。
    """
    trainer.logger.info("\n--- 梯度流检查 (首个 iteration) ---")

    frozen_with_grad = 0
    trainable_no_grad = 0
    trainable_zero_grad = 0

    for name, param in trainer.model.named_parameters():
        clean_name = name.replace('module.', '', 1)
        module_name = clean_name.split('.')[0]

        # Body-shape T1 (gated): whitelisted body_regressor sub-heads are trainable
        # despite body_regressor being in frozen_modules. Skip them here so they are
        # not flagged as "frozen module with grad". No-op when the prefix list is empty.
        if any(clean_name.startswith(p)
               for p in getattr(trainer.model.module, 'body_shape_trainable_prefixes', [])):
            continue

        if module_name in trainer.model.module.frozen_modules:
            if param.grad is not None:
                frozen_with_grad += 1
                trainer.logger.error(f"  ❌ {name}: 冻结模块不应有梯度!")

        elif module_name in trainer.model.module.trainable_module_names and param.requires_grad:
            if param.grad is None:
                trainable_no_grad += 1
                trainer.logger.warning(f"  ⚠️  {name}: 训练模块梯度为 None")
            elif param.grad.norm().item() == 0:
                trainable_zero_grad += 1
                # 这可能是正常的（某些参数在当前 batch 未被使用）

    trainer.logger.info(f"  冻结模块有梯度: {frozen_with_grad} (应为 0)")
    trainer.logger.info(f"  训练模块无梯度: {trainable_no_grad} (应为 0)")
    trainer.logger.info(f"  训练模块零梯度: {trainable_zero_grad} (可能正常)")

    if frozen_with_grad == 0 and trainable_no_grad == 0:
        trainer.logger.info("  ✅ 梯度流正常!")
    else:
        trainer.logger.error("  ❌ 梯度流异常，请检查冻结设置!")

    trainer.logger.info("--- 梯度流检查结束 ---\n")


if __name__ == "__main__":
    main()
