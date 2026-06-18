# 首次训练（从原始 OSX 预训练模型开始，InterHand26M + BEDLAM 微调）
python train.py \
      --gpu_ids 0 --lr 2e-4 --lr_mult 0.1 --train_batch_size 64 --num_thread 8 \
      --end_epoch 14 --phase1_epochs 10 \
      --exp_name output/interhand_bedlam_b \
      --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
      --encoder_setting osx_l --grad_clip 1.0 --decoder_setting pytorch \
      --bedlam_use_hand_roi_quality --bedlam_no_hand_img_loss

# 恢复训练
# python train.py \
#     --gpu_ids 0 \
#     --lr 2e-4 \
#     --lr_mult 0.1 \
#     --continue_train \
#     --train_batch_size 64 \
#     --num_thread 8 \
#     --end_epoch 14 \
#     --exp_name output/interhand_bedlam \
#     --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
#     --encoder_setting osx_l \
#     --grad_clip 1.0 \
#     --decoder_setting pytorch \
#     --continue_train_path /workspace/myosx/output/interhand_bedlam/model_dump/snapshot_XX.pth.tar
