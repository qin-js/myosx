import os
import argparse
import torch.backends.cudnn as cudnn
from config import cfg
import torch

PHASE1_EPOCHS = 17  # 前 10 个 epoch 只训练新模块
PHASE2_EPOCHS = 40  # 后 20 个 epoch 全部微调


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
    parser.add_argument('--decoder_setting', type=str, default='normal', choices=['normal', 'wo_face_decoder', 'wo_decoder', 'pytorch'])
    parser.add_argument('--agora_benchmark', action='store_true')
    parser.add_argument('--ubody_benchmark', action='store_true')
    parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_models/osx_l.pth.tar')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='梯度裁剪最大范数')
    args = parser.parse_args()
    return args


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
    )

    # 如果 cfg 中没有 lr_mult，手动设置
    if not hasattr(cfg, 'lr_mult'):
        cfg.lr_mult = args.lr_mult

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
    trainer.logger.info(f'  📦 Batch size: {cfg.train_batch_size} x {cfg.num_gpus} GPU')
    trainer.logger.info(f'  🔄 Epochs: {trainer.start_epoch} → {cfg.end_epoch}')
    trainer.logger.info('=' * 60 + '\n')

    # ================================================================
    #  训练循环
    # ================================================================
    print('### Start training ###')
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        # ---- 切换训练阶段 ----
        if epoch < PHASE1_EPOCHS:
            if trainer.model.module.training_phase != 1:
                trainer.model.module.set_training_phase(1)
                # Phase 1 的 optimizer：只包含 position_net + decoder
                trainer.optimizer = torch.optim.Adam([
                    {'params': list(trainer.model.module.hand_position_net.parameters()) +
                            list(trainer.model.module.hand_decoder.parameters()) +
                            list(trainer.model.module.face_position_net.parameters()) +
                            list(trainer.model.module.face_decoder.parameters()),
                    'lr': cfg.lr}
                ])
                trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    trainer.optimizer, 
                    PHASE1_EPOCHS * trainer.itr_per_epoch,
                    eta_min=1e-6
                )
                trainer.logger.info(f"=== Phase 1: epoch {epoch} ===")
        else:
            if trainer.model.module.training_phase != 2:
                trainer.model.module.set_training_phase(2)
                # Phase 2 的 optimizer：新模块 + regressor（小学习率）
                trainer.optimizer = torch.optim.Adam([
                    {'params': list(trainer.model.module.hand_position_net.parameters()) +
                            list(trainer.model.module.hand_decoder.parameters()) +
                            list(trainer.model.module.face_position_net.parameters()) +
                            list(trainer.model.module.face_decoder.parameters()),
                    'lr': cfg.lr * 0.1},  # Phase 2 新模块也降低学习率
                    {'params': list(trainer.model.module.hand_regressor.parameters()) +
                            list(trainer.model.module.face_regressor.parameters()),
                    'lr': cfg.lr * 0.01},  # regressor 用更小的学习率
                ])
                trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    trainer.optimizer,
                    PHASE2_EPOCHS * trainer.itr_per_epoch,
                    eta_min=1e-7
                )
                trainer.logger.info(f"=== Phase 2: epoch {epoch} ===")

        trainer.tot_timer.tic()
        trainer.read_timer.tic()

        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # ---- Forward ----
            trainer.optimizer.zero_grad()
            loss = trainer.model(inputs, targets, meta_info, 'train')
            loss = {k: loss[k].mean() for k in loss}

            # ---- Backward ----
            total_loss = sum(loss[k] for k in loss)
            total_loss.backward()

            # ---- 梯度裁剪（防止新模块初期梯度爆炸）----
            if args.grad_clip > 0:
                trainable_params = []
                for module in trainer.model.module.trainable_modules:
                    trainable_params.extend(
                        [p for p in module.parameters() if p.requires_grad and p.grad is not None]
                    )
                for module in trainer.model.module.special_trainable_modules:
                    trainable_params.extend(
                        [p for p in module.parameters() if p.requires_grad and p.grad is not None]
                    )
                if trainable_params:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)

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
                screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k, v in loss.items()]
                trainer.logger.info(' '.join(screen))

            # ---- 首个 iteration 检查梯度流 ----
            if epoch == trainer.start_epoch and itr == 0:
                _check_gradient_flow(trainer)

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

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

        elif module_name in trainer.model.module.trainable_module_names:
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