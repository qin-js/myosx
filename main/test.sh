export UBODY_ANNOTATION_DIR=/workspace/myosx/dataset/UBody/annotations_filtered
# python test.py \
#     --gpu 0 \
#     --test_batch_size 32 \
#     --decoder_setting normal \
#     --testset InterHand26M \
#     --interhand_eval_split test \
#     --max_eval_iters -1 \
#     --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
#     --exp_name output/eval_interhand_bedlam_c \
#     --continue_train_paths ../output/face_ubody_e/model_dump/snapshot_*.pth.tar  
# --continue_train_paths ../output/interhand_bedlam_c/model_dump/snapshot_*.pth.tar

python test.py --gpu 0 --testset HInt --exp_name output/eval_hint \
      --max_eval_iters -1 \
      --interhand_eval_split test \
      --decoder_setting pytorch --encoder_setting osx_l \
      --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
      --continue_train_paths  ../output/face_ubody_e/model_dump/snapshot_4.pth.tar ../output/joint_polish_f/model_dump/snapshot_0.pth.tar ../output/joint_polish_f/model_dump/snapshot_1.pth.tar 
      #--continue_train_paths '../output/coco_pilot/model_dump/snapshot_1*.pth.tar' 