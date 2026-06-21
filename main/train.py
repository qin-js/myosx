import os
import argparse
import torch.backends.cudnn as cudnn
from config import cfg
import torch

DEFAULT_PHASE1_EPOCHS = 10


def _per_source_losses(raw_loss, meta_info):
    """Split per-sample losses by dataset source for separate monitoring.

    raw_loss: dict of loss tensors still carrying the batch dim (dim 0), BEFORE
    any .mean(). Returns {'interhand': {...}, 'bedlam': {...}, 'ubody': {...}}
    ('ubody' only when meta_info carries is_ubody) where each value is the mean
    loss over the samples of that source in the batch. Returns None if the
    interhand/bedlam source flags are absent (e.g. single-dataset training).
    """
    if 'is_interhand' not in meta_info or 'is_bedlam' not in meta_info:
        return None
    masks = {
        'interhand': meta_info['is_interhand'].detach().view(-1).float(),
        'bedlam': meta_info['is_bedlam'].detach().view(-1).float(),
    }
    if 'is_ubody' in meta_info:
        masks['ubody'] = meta_info['is_ubody'].detach().view(-1).float()
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
    if 'is_interhand' in meta_info:
        batch_state.append('interhand=%.0f' % meta_info['is_interhand'].sum().detach().cpu().item())
    if 'is_bedlam' in meta_info:
        batch_state.append('bedlam=%.0f' % meta_info['is_bedlam'].sum().detach().cpu().item())
    if 'is_ubody' in meta_info:
        batch_state.append('ubody=%.0f' % meta_info['is_ubody'].sum().detach().cpu().item())

    msg = 'Non-finite total_loss before backward at epoch %d itr %d. %s %s' % (
        epoch,
        itr,
        ' '.join(batch_state),
        ' '.join(loss_state),
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
    parser.add_argument('--phase1_epochs', type=int, default=DEFAULT_PHASE1_EPOCHS,
                        help='手部微调 Phase 1 的 epoch 数；默认覆盖 10 epoch 短程混训')
    parser.add_argument('--train_face_modules', action='store_true',
                        help='默认冻结 face 分支；需要 UBody face/expression 微调时显式开启')
    parser.add_argument('--no_train_hand_modules', action='store_true',
                        help='冻结手部分支（face-only 阶段用）：手部三件套不进优化器、'
                             '保持已加载的手权重不动；其名字仍在 snapshot 里保存，'
                             '便于后续联合阶段暖启动')
    parser.add_argument('--init_trained_path', type=str, default='',
                        help='非续训(continue_train=False)时，额外暖加载一个 lightweight '
                             'snapshot 的已训练手/脸模块（不续 epoch/optimizer）；用于阶段衔接')
    parser.add_argument('--no_phase1_hand_regressor', action='store_true',
                        help='Phase 1 不训练 hand_regressor，仅用于对比旧热身策略')
    parser.add_argument('--bedlam_max_samples', type=int, default=None)
    parser.add_argument('--interhand_max_samples', type=int, default=None)
    parser.add_argument('--bedlam_use_hand_roi_quality', action='store_true')
    parser.add_argument('--bedlam_no_hand_img_loss', action='store_true',
                        help='BEDLAM 不监督手部 joint_img/smplx_joint_img（soft-argmax 热图头的'
                             '唯一梯度源），切断对共享 hand_position_net 的梯度污染；'
                             '保留 BEDLAM 手部 joint_proj 与全部 SMPL-X/3D/pose 监督，InterHand 不受影响')
    parser.add_argument('--use_hand_rotmat_pose_loss', action='store_true',
                        help='额外对 30 个手指关节加 rotation-matrix pose loss（旋转感知，比 '
                             'axis-angle L1 更稳）；默认关闭，不影响现有训练')
    parser.add_argument('--hand_rotmat_pose_loss_weight', type=float, default=None,
                        help='rotation-matrix hand pose loss 权重（默认沿用 cfg 的 1.0）')
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

    trainable_modules = posnet_modules + normal_modules + special_modules
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
    cfg.set_args(args.gpu_ids, args.lr, args.continue_train)
    cfg.set_additional_args(
        exp_name=args.exp_name,
        num_thread=args.num_thread,
        train_batch_size=args.train_batch_size,
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
        use_weighted_dataset_sampling=not args.disable_weighted_dataset_sampling,
        posnet_lr_mult=args.posnet_lr_mult,
        posnet_grad_clip=args.posnet_grad_clip,
    )

    if cfg.decoder_setting != 'pytorch':
        raise ValueError("Current training phase/freeze code requires --decoder_setting pytorch")

    if args.bedlam_max_samples is not None:
        cfg.bedlam_max_samples = args.bedlam_max_samples
    if args.interhand_max_samples is not None:
        cfg.interhand_max_samples = args.interhand_max_samples
    if args.bedlam_use_hand_roi_quality:
        cfg.bedlam_use_hand_roi_quality = True
    if args.bedlam_no_hand_img_loss:
        cfg.bedlam_supervise_hand_img = False
    if args.use_hand_rotmat_pose_loss:
        cfg.use_hand_rotmat_pose_loss = True
    if args.hand_rotmat_pose_loss_weight is not None:
        cfg.hand_rotmat_pose_loss_weight = args.hand_rotmat_pose_loss_weight
    if args.init_trained_path:
        cfg.init_trained_path = args.init_trained_path

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
                        f'train face modules: {cfg.train_face_modules}')
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

            # ---- 梯度裁剪（防止新模块初期梯度爆炸）----
            # position_nets (soft-argmax, diverges mid-epoch) get a separate,
            # tighter clip; everything else uses the global grad_clip. Clipping
            # the two disjoint sets independently avoids double-clipping posnet.
            if cfg.posnet_grad_clip > 0:
                posnet_params = _grad_params_for_groups(trainer.optimizer, {'position_nets'})
                if posnet_params:
                    torch.nn.utils.clip_grad_norm_(posnet_params, max_norm=cfg.posnet_grad_clip)
            if args.grad_clip > 0:
                other_params = _grad_params_for_groups(
                    trainer.optimizer, {'hand_face_new_modules', 'regressors'})
                if other_params:
                    torch.nn.utils.clip_grad_norm_(other_params, max_norm=args.grad_clip)

            trainer.optimizer.step()
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
                if 'is_interhand' in meta_info and 'is_bedlam' in meta_info:
                    screen += [
                        'batch_interhand: %.0f' % meta_info['is_interhand'].sum().detach().cpu().item(),
                        'batch_bedlam: %.0f' % meta_info['is_bedlam'].sum().detach().cpu().item(),
                    ]
                    if 'is_ubody' in meta_info:
                        screen += ['batch_ubody: %.0f' % meta_info['is_ubody'].sum().detach().cpu().item()]
                screen += ['%s: %.4f' % ('loss_' + k.lstrip('_'), v.detach()) for k, v in loss.items()]
                trainer.logger.info(' '.join(screen))
                if per_source is not None:
                    for src_name in ('interhand', 'bedlam', 'ubody'):
                        src_losses = per_source.get(src_name, {})
                        if src_losses:
                            line = ['  [%s]' % src_name] + [
                                '%s: %.4f' % (k.lstrip('_'), val) for k, val in src_losses.items()
                            ]
                            trainer.logger.info(' '.join(line))

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
