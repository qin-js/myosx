# 首次训练（从原始 OSX 预训练模型开始，InterHand26M + BEDLAM 微调）
# python train.py \
#       --gpu_ids 0 --lr 2e-4 --lr_mult 0.1 --train_batch_size 64 --num_thread 8 \
#       --end_epoch 14 --phase1_epochs 10 \
#       --exp_name output/interhand_bedlam_c \
#       --posnet_lr_mult 1.0 \
#       --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
#       --encoder_setting osx_l --grad_clip 1.0 --decoder_setting pytorch \
#       --bedlam_use_hand_roi_quality --bedlam_no_hand_img_loss

# 恢复训练
python train.py \
      --gpu_ids 0 --lr 2e-4 --lr_mult 0.1 --train_batch_size 64 --num_thread 8 \
      --end_epoch 14 --phase1_epochs 10 \
      --exp_name output/interhand_bedlam_c \
      --posnet_lr_mult 1.0 \
      --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
      --encoder_setting osx_l --grad_clip 1.0 --decoder_setting pytorch \
      --bedlam_use_hand_roi_quality --bedlam_no_hand_img_loss \
      --continue_train \
      --continue_train_path /workspace/myosx/output/interhand_bedlam_c/model_dump/snapshot_5.pth.tar

python train.py --gpu_ids 0 --lr 2e-4 --lr_mult 0.1 \
    --train_batch_size 64 --num_thread 8 --end_epoch 12 --phase1_epochs 8 \
    --exp_name output/face_ubody_e --decoder_setting pytorch --encoder_setting osx_l --grad_clip 1.0 \
    --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
    --init_trained_path ../output/interhand_bedlam_c/model_dump/snapshot_8.pth.tar \
    --train_face_modules --no_train_hand_modules --posnet_lr_mult 1.0
