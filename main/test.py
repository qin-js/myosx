import argparse
from config import cfg
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import os.path as osp
import os

def visualize_debug(inputs, targets, meta_info, save_dir='debug_vis', sample_idx=0):
    """
    可视化调试函数：将 keypoint 和 mesh 投影到原图
    
    Args:
        inputs: batch inputs dict
        targets: batch targets dict  
        meta_info: batch meta_info dict
        save_dir: 保存目录
        sample_idx: batch 中的样本索引
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取单个样本的数据
    img = inputs['img'][sample_idx].cpu().numpy()  # (3, H, W)
    img = (img * 255).astype(np.uint8).transpose(1, 2, 0)  # (H, W, 3) BGR
    img = img.copy()  # make writable
    
    bb2img_trans = meta_info['bb2img_trans'][sample_idx].cpu().numpy()  # (3, 3)
    
    # ========== 可视化 joint_img (2D keypoints) ==========
    if 'joint_img' in targets:
        joint_img = targets['joint_img'][sample_idx].cpu().numpy()  # (J, 3)
        joint_valid = meta_info.get('joint_valid', None)
        if joint_valid is not None:
            joint_valid = joint_valid[sample_idx].cpu().numpy()  # (J, 1)
        
        # 将 joint_img 从 normalized 坐标转换回原图坐标
        # joint_img 是在 input_img_shape 空间中的坐标
        joint_img_hm = joint_img[:, :2].copy()  # (J, 2)
        
        # 从 heatmap 空间转换到 input_img 空间
        joint_img_input = joint_img_hm.copy()
        joint_img_input[:, 0] = joint_img_hm[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
        joint_img_input[:, 1] = joint_img_hm[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
        
        # 从 input_img 空间转换到原图空间
        joint_img_orig = np.concatenate([joint_img_input, np.ones((joint_img_input.shape[0], 1))], axis=1)
        joint_img_orig = np.dot(bb2img_trans, joint_img_orig.T).T[:, :2]
        
        # 绘制关键点
        for j in range(joint_img_orig.shape[0]):
            x, y = joint_img_orig[j]
            if joint_valid is not None and joint_valid[j, 0] > 0:
                cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
            else:
                cv2.circle(img, (int(x), int(y)), 2, (128, 128, 128), 1)
    
    # ========== 可视化 smplx_joint_img ==========
    if 'smplx_joint_img' in targets:
        smplx_joint_img = targets['smplx_joint_img'][sample_idx].cpu().numpy()  # (J, 3)
        smplx_joint_valid = meta_info.get('smplx_joint_valid', None)
        if smplx_joint_valid is not None:
            smplx_joint_valid = smplx_joint_valid[sample_idx].cpu().numpy()  # (J, 1)
        
        # 转换坐标
        smplx_joint_hm = smplx_joint_img[:, :2].copy()
        smplx_joint_input = smplx_joint_hm.copy()
        smplx_joint_input[:, 0] = smplx_joint_hm[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
        smplx_joint_input[:, 1] = smplx_joint_hm[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
        
        smplx_joint_orig = np.concatenate([smplx_joint_input, np.ones((smplx_joint_input.shape[0], 1))], axis=1)
        smplx_joint_orig = np.dot(bb2img_trans, smplx_joint_orig.T).T[:, :2]
        
        # 绘制 SMPLX 关键点（红色）
        for j in range(smplx_joint_orig.shape[0]):
            x, y = smplx_joint_orig[j]
            if smplx_joint_valid is not None and smplx_joint_valid[j, 0] > 0:
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
    
    # ========== 可视化 bbox ==========
    if 'lhand_bbox_center' in targets:
        # 左手 bbox (蓝色)
        lhand_center = targets['lhand_bbox_center'][sample_idx].cpu().numpy()
        lhand_size = targets['lhand_bbox_size'][sample_idx].cpu().numpy()
        lhand_valid = meta_info.get('lhand_bbox_valid', None)
        if lhand_valid is not None and lhand_valid[sample_idx].item() > 0:
            # 转换到原图
            center_orig = np.dot(bb2img_trans, [lhand_center[0], lhand_center[1], 1])[:2]
            size_orig = lhand_size * (bb2img_trans[0, 0] + bb2img_trans[1, 1]) / 2
            xmin = center_orig[0] - size_orig[0] / 2
            ymin = center_orig[1] - size_orig[1] / 2
            cv2.rectangle(img, (int(xmin), int(ymin)), 
                         (int(xmin + size_orig[0]), int(ymin + size_orig[1])), (255, 0, 0), 2)
    
    if 'rhand_bbox_center' in targets:
        # 右手 bbox (绿色)
        rhand_center = targets['rhand_bbox_center'][sample_idx].cpu().numpy()
        rhand_size = targets['rhand_bbox_size'][sample_idx].cpu().numpy()
        rhand_valid = meta_info.get('rhand_bbox_valid', None)
        if rhand_valid is not None and rhand_valid[sample_idx].item() > 0:
            center_orig = np.dot(bb2img_trans, [rhand_center[0], rhand_center[1], 1])[:2]
            size_orig = rhand_size * (bb2img_trans[0, 0] + bb2img_trans[1, 1]) / 2
            xmin = center_orig[0] - size_orig[0] / 2
            ymin = center_orig[1] - size_orig[1] / 2
            cv2.rectangle(img, (int(xmin), int(ymin)), 
                         (int(xmin + size_orig[0]), int(ymin + size_orig[1])), (0, 255, 0), 2)
    
    if 'face_bbox_center' in targets:
        # face bbox (黄色)
        face_center = targets['face_bbox_center'][sample_idx].cpu().numpy()
        face_size = targets['face_bbox_size'][sample_idx].cpu().numpy()
        face_valid = meta_info.get('face_bbox_valid', None)
        if face_valid is not None and face_valid[sample_idx].item() > 0:
            center_orig = np.dot(bb2img_trans, [face_center[0], face_center[1], 1])[:2]
            size_orig = face_size * (bb2img_trans[0, 0] + bb2img_trans[1, 1]) / 2
            xmin = center_orig[0] - size_orig[0] / 2
            ymin = center_orig[1] - size_orig[1] / 2
            cv2.rectangle(img, (int(xmin), int(ymin)), 
                         (int(xmin + size_orig[0]), int(ymin + size_orig[1])), (0, 255, 255), 2)
    
    # 保存图像
    save_path = osp.join(save_dir, f'debug_sample_{sample_idx}.png')
    cv2.imwrite(save_path, img)
    print(f"[Debug] Saved visualization to {save_path}")
    
    # 打印调试信息
    print(f"[Debug] === Sample {sample_idx} Info ===")
    print(f"[Debug] img shape: {inputs['img'][sample_idx].shape}")
    if 'joint_img' in targets:
        print(f"[Debug] joint_img shape: {targets['joint_img'][sample_idx].shape}")
    if 'smplx_joint_img' in targets:
        print(f"[Debug] smplx_joint_img shape: {targets['smplx_joint_img'][sample_idx].shape}")
    if 'smplx_shape' in targets:
        shape = targets['smplx_shape'][sample_idx].cpu().numpy()
        print(f"[Debug] smplx_shape: {shape[:5]}... (first 5)")
    if 'smplx_pose' in targets:
        pose = targets['smplx_pose'][sample_idx].cpu().numpy()
        print(f"[Debug] smplx_pose shape: {pose.shape}, mean: {np.abs(pose).mean():.4f}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--exp_name', type=str, default='output/test')
    parser.add_argument('--test_batch_size', type=int, default=16 )
    parser.add_argument('--encoder_setting', type=str, default='osx_l', choices=['osx_b', 'osx_l'])
    parser.add_argument('--decoder_setting', type=str, default='wo_face_decoder', choices=['normal', 'wo_face_decoder', 'wo_decoder', 'pytorch'])
    parser.add_argument('--testset', type=str, default='EHF')
    parser.add_argument('--agora_benchmark', action='store_true')
    parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_models/osx_l_wo_face_decoder.pth.tar')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set proper gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def main():
    print('### Argument parse and create log ###')
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cfg.set_additional_args(exp_name=args.exp_name,
                            test_batch_size=args.test_batch_size,
                            encoder_setting=args.encoder_setting,
                            decoder_setting=args.decoder_setting,
                            pretrained_model_path=args.pretrained_model_path,
                            agora_benchmark=args.agora_benchmark,
                            testset=args.testset,
                            )
    cudnn.benchmark = True
    from common.base import Tester
    tester = Tester()
    tester._make_batch_generator()
    tester._make_model()

    eval_result = {}
    cur_sample_idx = 0
    debug_vis_done = False  # 只可视化第一个 batch
    
    for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
        
        
        # ========== 调试可视化：只对第一个 batch ==========
        if not debug_vis_done:
            print("\n[Debug] Visualizing first batch...")
            # 可视化 batch 中的前几个样本
            batch_size = inputs['img'].shape[0]
            for i in range(min(3, batch_size)):  # 最多可视化前3个样本
                visualize_debug(inputs, targets, meta_info, 
                              save_dir=osp.join(cfg.output_dir, 'debug_vis'),
                              sample_idx=i)
            debug_vis_done = True


        # forward
        with torch.no_grad():
            out = tester.model(inputs, targets, meta_info, 'test')

        # save output
        out = {k: v.cpu().numpy() for k, v in out.items()}
        for k, v in out.items(): batch_size = out[k].shape[0]
        out = [{k: v[bid] for k, v in out.items()} for bid in range(batch_size)]

        # evaluate
        cur_eval_result = tester._evaluate(out, cur_sample_idx)
        for k, v in cur_eval_result.items():
            if k in eval_result:
                eval_result[k] += v
            else:
                eval_result[k] = v
        cur_sample_idx += len(out)

        if itr >= 50:
            break

    tester._print_eval_result(eval_result)

if __name__ == "__main__":
    main()
