export UBODY_ANNOTATION_DIR=/workspace/myosx/dataset/UBody/annotations_filtered
python train.py \
  --gpu_ids 0 \
  --lr 5e-5 \
  --lr_mult 0.1 \
  --train_batch_size 64 \
  --num_thread 8 \
  --end_epoch 4 \
  --phase1_epochs 2 \
  --save_iters 1000 \
  --exp_name output/coco_pilot \
  --decoder_setting pytorch \
  --encoder_setting osx_l \
  --grad_clip 1.0 \
  --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
  --continue_train \
  --continue_train_path ../output/coco_pilot/model_dump/snapshot_0.pth.tar \
  --train_face_modules \
  --posnet_lr_mult 0.5 \
  --ubody_use_hand_roi_quality \
  #--mscoco_use_hand_roi_quality

# ─────────────── 历史命令（保留备查，勿直接跑） ───────────────

# Stage 3 联合 polish：修末端自然手泛化（当前活动命令）
# 诊断结论：UBody 自然手退化沿指节深度递增、tip 最差、同构于随机手头（区间 72%）。
# 目标=低 lr + UBody 用 box_net 预测框（不注入 GT 框）让 hand decoder 见真实裁剪分布。
# 先短跑 2-4 epoch，别直接 12；成功判据见 docs/continue.txt「下一步：Stage 3」。
#
# 前置（必须在 config.py 改，CLI 不覆盖数据集选择）：
#   trainset_3d = ['BEDLAM', 'InterHand26M', 'UBody']
#   trainset_3d_sample_prob = {'BEDLAM': 0.20, 'InterHand26M': 0.35, 'UBody': 0.45}
#   （不补 UBody 概率会报 Missing sample probabilities）
# UBody ROI gate 阈值用 config.py 默认 0.6/8（诊断起点）；正式跑前先 --end_epoch 1
#   短跑看日志 [ubody] hand_roi_gate_coverage/pass/n_valid 校准，再按需改 config.py。
# export UBODY_ANNOTATION_DIR=/workspace/myosx/dataset/UBody/annotations_filtered
# python train.py --gpu_ids 0 --lr 5e-5 --lr_mult 0.1 \
#     --train_batch_size 64 --num_thread 8 --end_epoch 4 --phase1_epochs 2 \
#     --exp_name output/joint_polish_f --decoder_setting pytorch --encoder_setting osx_l --grad_clip 1.0 \
#     --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
#     --init_trained_path ../output/face_ubody_e/model_dump/snapshot_4.pth.tar \
#     --train_face_modules --posnet_lr_mult 0.5 \
#     --bedlam_use_hand_roi_quality --bedlam_no_hand_img_loss \
#     --ubody_use_hand_roi_quality

# python train.py --gpu_ids 0 --lr 5e-5 --lr_mult 0.1 \
#     --train_batch_size 64 --num_thread 8 --end_epoch 4 --phase1_epochs 2 \
#     --exp_name output/joint_polish_f --decoder_setting pytorch --encoder_setting osx_l --grad_clip 1.0 \
#     --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
#     --continue_train \
#     --continue_train_path ../output/joint_polish_f/model_dump/snapshot_1.pth.tar \
#     --train_face_modules --posnet_lr_mult 0.5 \
#     --bedlam_use_hand_roi_quality --bedlam_no_hand_img_loss \
#     --ubody_use_hand_roi_quality --save_iters 1280


# Stage 2 冻手训脸（face_ubody_e）：从 _c snapshot_8 暖启，只训脸
# python train.py --gpu_ids 0 --lr 2e-4 --lr_mult 0.1 \
#     --train_batch_size 64 --num_thread 8 --end_epoch 12 --phase1_epochs 8 \
#     --exp_name output/face_ubody_e --decoder_setting pytorch --encoder_setting osx_l --grad_clip 1.0 \
#     --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
#     --train_face_modules --no_train_hand_modules --posnet_lr_mult 1.0 \
#     --continue_train \
#     --continue_train_path /workspace/myosx/output/face_ubody_e/model_dump/snapshot_5.pth.tar

# Stage 1 手部微调（interhand_bedlam_c）：从原始 OSX 预训练开始，InterHand26M + BEDLAM
# python train.py \
#       --gpu_ids 0 --lr 2e-4 --lr_mult 0.1 --train_batch_size 64 --num_thread 8 \
#       --end_epoch 14 --phase1_epochs 10 \
#       --exp_name output/interhand_bedlam_c \
#       --posnet_lr_mult 1.0 \
#       --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
#       --encoder_setting osx_l --grad_clip 1.0 --decoder_setting pytorch \
#       --bedlam_use_hand_roi_quality --bedlam_no_hand_img_loss
