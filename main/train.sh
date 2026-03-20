# 首次训练（从原始 OSX 预训练模型开始）
python train.py \
    --gpu_ids 0 \
    --lr 1e-4 \
    --lr_mult 0.1 \
    --train_batch_size 64 \
    --num_thread 8 \
    --end_epoch 20 \
    --exp_name output/dcnv4_hand_face \
    --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
    --encoder_setting osx_l \
    --grad_clip 1.0 \
    --decoder_setting pytorch

# 恢复训练
# python train.py \
#     --gpu_ids 0,1 \
#     --lr 1e-4 \
#     --lr_mult 0.1 \
#     --continue_train \
#     --train_batch_size 32 \
#     --end_epoch 100 \
#     --exp_name output/dcnv4_hand_face \
#     --pretrained_model_path output/dcnv4_hand_face/model/snapshot_50.pth.tar \
#     --encoder_setting osx_l