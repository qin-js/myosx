import argparse
from config import cfg
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import os.path as osp
import os
import logging
from glob import glob


def _save_bootstrap_npz(path, eval_result):
    """Dump each per-sample / per-hand metric list as a 1D float array for
    offline bootstrap CI. Non-numeric or ragged keys are skipped. Safe no-op
    when nothing usable is present."""
    arrs = {}
    for k, v in eval_result.items():
        try:
            a = np.asarray(v, dtype=np.float64)
        except (ValueError, TypeError):
            continue
        if a.ndim == 1 and a.size > 0:
            arrs[k] = a
    if arrs:
        np.savez(path, **arrs)
        print(f"[dump] bootstrap arrays -> {path}  ({len(arrs)} keys)")


def _save_crosspath_npz(path, records):
    """Dump per-sample frozen-body params (+ bb2img_trans alignment anchor) for
    the OSX-normal vs pytorch cross-path comparison. No-op when empty."""
    arrs = {}
    for k, v in records.items():
        if len(v) == 0:
            continue
        try:
            arrs[k] = np.stack([np.asarray(x, dtype=np.float64) for x in v], axis=0)
        except (ValueError, TypeError):
            continue
    if arrs:
        np.savez(path, **arrs)
        print(f"[dump] crosspath arrays -> {path}  (N={next(iter(arrs.values())).shape[0]})")


# Per-hand dump keys (UBody grouped-attribution; see docs/wa2d_group_attribution.md).
# These are (N,21) float arrays + scalar lists + an img_path str list. Kept separate
# from bootstrap npz so the bootstrap CI tool's 1D-array contract is unchanged.
_PERHAND_NUMERIC_KEYS = ('hand2d_wa_joints', 'hand2d_abs_joints', 'hand2d_joint_valid',
                         'hand2d_side', 'hand2d_hand_size', 'hand2d_n_visible')


def _save_perhand_npz(path, eval_result):
    """Dump per-hand (N,21) joint error + hand-attribute arrays for offline grouped
    attribution. Stacks each list into a 2D/1D array; img_path saved as a string
    array. Safe no-op when the per-hand keys are absent (non-UBody testsets)."""
    arrs = {}
    for k in _PERHAND_NUMERIC_KEYS:
        v = eval_result.get(k, [])
        if not v:
            continue
        try:
            arrs[k] = np.stack([np.asarray(x, dtype=np.float64) for x in v], axis=0)
        except (ValueError, TypeError):
            continue
    img_paths = eval_result.get('hand2d_img_path', [])
    if img_paths:
        try:
            arrs['hand2d_img_path'] = np.asarray(img_paths, dtype=str)
        except (ValueError, TypeError):
            pass
    if arrs:
        lens = {k: v.shape[0] for k, v in arrs.items()}
        if len(set(lens.values())) != 1:
            print(f"[dump] skip per-hand arrays with inconsistent lengths: {lens}")
            return
        np.savez(path, **arrs)
        n = next(iter(arrs.values())).shape[0]
        print(f"[dump] per-hand arrays -> {path}  (N={n}, keys={list(arrs.keys())})")


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
    parser.add_argument('--testset', type=str, default='EHF', choices=['EHF', 'HAA500', 'InterHand26M', 'UBody', 'HInt'])
    parser.add_argument('--interhand_eval_split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='InterHand26M: which annotation split to load for eval (files only; '
                             'augmentation stays identity). "train" = contaminated upper bound '
                             'when the eval model trained on train.')
    parser.add_argument('--agora_benchmark', action='store_true')
    parser.add_argument('--debug_vis', action='store_true')
    parser.add_argument('--hint_root', type=str, default='/workspace/HInt_annotation_partial',
                        help='HInt root folder. Defaults to HINT_ROOT, then dataset/HInt/HInt_annotation(_partial).')
    parser.add_argument('--hint_split', type=str, default='test', choices=['train', 'val', 'test', 'TRAIN', 'VAL', 'TEST'],
                        help='HInt split folders to evaluate. Default: test.')
    parser.add_argument('--hint_sources', type=str, default='epick,newdays',
                        help='Comma-separated HInt source filter matched against folder names, e.g. epick,newdays.')
    parser.add_argument('--hint_max_samples', type=int, default=None,
                        help='Cap HInt loaded hand instances for quick checks.')
    parser.add_argument('--hint_sample_interval', type=int, default=None,
                        help='Subsample HInt json files by this interval.')
    parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_models/osx_l_wo_face_decoder.pth.tar')
    parser.add_argument('--continue_train_path', type=str, default="")
    parser.add_argument('--continue_train_paths', type=str, nargs='+', default=None,
                        help='一次评估多个训练 checkpoint。支持空格分隔多个路径，也支持通配符；'
                             '每个 checkpoint 的结果会保存到 result/<checkpoint_name>/')
    parser.add_argument('--max_eval_iters', type=int, default=-1,
                        help='只评估前 N 个 batch（快速查看用）；<=0 表示评估整个 testset')
    parser.add_argument('--use_poseur_hand_decoder', action='store_true',
                        help='decoder 隔离实验：手部用 PoseurDecoder 评测（必须与训练该 checkpoint 时一致）')
    parser.add_argument('--dump_analysis', action='store_true',
                        help='额外 dump 逐样本/逐手指标到 result_dir 下的 '
                             'bootstrap_<testset>.npz 与 crosspath_<testset>.npz，'
                             '供 tool/analysis/ 做 bootstrap CI 与跨路径校验；默认关闭、不改变评测行为')
    parser.add_argument('--dump_encoder_n', type=int, default=0,
                        help='编码器忠实度探针：dump 前 N 个样本的 body_img/img_feat/task_tokens 到 '
                             'result_dir/encoder_<testset>.npz，供 tool/analysis/encoder_compare.py 比对 '
                             'MMCV-ViT(normal) 与 StandardViT(pytorch)；0=关闭（默认）')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set proper gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args


def _expand_checkpoint_paths(args):
    raw_paths = []
    if args.continue_train_paths:
        raw_paths.extend(args.continue_train_paths)
    elif args.continue_train_path:
        raw_paths.append(args.continue_train_path)

    paths = []
    for raw_path in raw_paths:
        matches = sorted(glob(raw_path))
        if matches:
            paths.extend(matches)
        else:
            paths.append(raw_path)

    seen = set()
    unique_paths = []
    for path in paths:
        path = osp.normpath(path)
        if path not in seen:
            unique_paths.append(path)
            seen.add(path)
    return unique_paths


def _checkpoint_run_name(checkpoint_path):
    if not checkpoint_path:
        return 'pretrained'
    name = osp.basename(checkpoint_path)
    for suffix in ('.pth.tar', '.pth', '.tar', '.ckpt'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name.replace(' ', '_')


def _testset_result_name(args):
    if args.testset == 'InterHand26M':
        return f'{args.testset}_{args.interhand_eval_split}_result.txt'
    return f'{args.testset}_result.txt'


def _set_eval_output_dirs(base_result_dir, base_vis_dir, base_log_dir,
                          run_name):
    result_dir = osp.join(base_result_dir, run_name)
    vis_dir = osp.join(base_vis_dir, run_name)
    log_dir = osp.join(base_log_dir, run_name)

    cfg.result_dir = result_dir
    cfg.vis_dir = vis_dir
    cfg.log_dir = log_dir

    os.makedirs(cfg.result_dir, exist_ok=True)
    os.makedirs(cfg.vis_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)


def _set_base_output_dirs(base_result_dir, base_vis_dir, base_log_dir):
    cfg.result_dir = base_result_dir
    cfg.vis_dir = base_vis_dir
    cfg.log_dir = base_log_dir

    os.makedirs(cfg.result_dir, exist_ok=True)
    os.makedirs(cfg.vis_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)


def _rename_result_file(result_filename):
    target_path = osp.join(cfg.result_dir, result_filename)
    candidate_names = ['result.txt', 'HAA500_result.txt']
    for name in candidate_names:
        src_path = osp.join(cfg.result_dir, name)
        if osp.isfile(src_path):
            if osp.abspath(src_path) != osp.abspath(target_path):
                os.replace(src_path, target_path)
            return target_path
    return None


def _reset_test_logger_handlers():
    logger = logging.getLogger('test_logs.txt')
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _run_eval_for_current_checkpoint(args, Tester, result_filename):
    _reset_test_logger_handlers()
    tester = Tester()
    tester._make_batch_generator()
    tester._make_model()

    eval_result = {}
    cur_sample_idx = 0
    debug_vis_done = False  # 只可视化第一个 batch
    dump_on = getattr(args, 'dump_analysis', False)
    cp_keys = ('smplx_root_pose', 'smplx_body_pose', 'smplx_shape',
               'cam_trans', 'bb2img_trans')
    cp_records = {k: [] for k in cp_keys}
    # Encoder-fidelity probe: collect the (heavy) encoder I/O for the first N
    # samples only. cfg.dump_encoder gates the stash inside the model forward.
    enc_n = int(getattr(args, 'dump_encoder_n', 0) or 0)
    cfg.dump_encoder = enc_n > 0
    enc_keys = ('_enc_input', '_enc_img_feat', '_enc_task_tokens', 'bb2img_trans')
    enc_records = {k: [] for k in enc_keys}
    enc_collected = 0
    if args.max_eval_iters > 0:
        print(f"[eval] 限制只跑前 {args.max_eval_iters} 个 batch（部分评估，指标仅供快速参考）")
    
    for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
        
        
        # ========== 调试可视化：只对第一个 batch ==========
        if args.debug_vis and not debug_vis_done:
            print("\n[Debug] Visualizing first batch...")
            # 可视化 batch 中的前几个样本
            batch_size = inputs['img'].shape[0]
            for i in range(min(3, batch_size)):  # 最多可视化前3个样本
                visualize_debug(inputs, targets, meta_info, 
                              save_dir=osp.join(cfg.vis_dir, 'debug_vis'),
                              sample_idx=i)
            debug_vis_done = True


        # forward
        with torch.no_grad():
            out = tester.model(inputs, targets, meta_info, 'test')

        # Encoder probe: keep the heavy encoder tensors only until N samples are
        # collected, then drop them so the rest of eval is unaffected (and the
        # cpu().numpy() below stays cheap).
        if cfg.dump_encoder and enc_collected >= enc_n:
            for kk in [k for k in list(out.keys()) if k.startswith('_enc_')]:
                out.pop(kk)

        # save output
        out = {k: v.cpu().numpy() for k, v in out.items()}
        for k, v in out.items(): batch_size = out[k].shape[0]
        out = [{k: v[bid] for k, v in out.items()} for bid in range(batch_size)]

        # Optional per-sample dump (frozen-body params for the cross-path check).
        # Cheap: only a few small arrays per sample; off unless --dump_analysis.
        if dump_on:
            for d in out:
                for k in cp_keys:
                    if k in d:
                        cp_records[k].append(np.asarray(d[k]))

        # Encoder-fidelity probe: stash the first N samples' encoder I/O.
        if cfg.dump_encoder and enc_collected < enc_n:
            for d in out:
                if enc_collected >= enc_n:
                    break
                if all(k in d for k in enc_keys):
                    for k in enc_keys:
                        enc_records[k].append(np.asarray(d[k]))
                    enc_collected += 1

        # evaluate
        cur_eval_result = tester._evaluate(out, cur_sample_idx)
        for k, v in cur_eval_result.items():
            if k in eval_result:
                eval_result[k] += v
            else:
                eval_result[k] = v
        cur_sample_idx += len(out)

        if args.max_eval_iters > 0 and itr + 1 >= args.max_eval_iters:
            break

    tester._print_eval_result(eval_result)
    tag = args.testset
    if dump_on:
        _save_bootstrap_npz(osp.join(cfg.result_dir, f'bootstrap_{tag}.npz'), eval_result)
        _save_crosspath_npz(osp.join(cfg.result_dir, f'crosspath_{tag}.npz'), cp_records)
        _save_perhand_npz(osp.join(cfg.result_dir, f'perhand_{tag}.npz'), eval_result)
    if cfg.dump_encoder:
        _save_crosspath_npz(osp.join(cfg.result_dir, f'encoder_{tag}.npz'), enc_records)
    result_path = _rename_result_file(result_filename)
    if result_path:
        print(f"[eval] result_file: {result_path}")
    return eval_result


def main():
    print('### Argument parse and create log ###')
    args = parse_args()
    use_pytorch_decoder = args.decoder_setting == 'pytorch'
    checkpoint_paths = _expand_checkpoint_paths(args) if use_pytorch_decoder else []

    cfg.set_args(args.gpu_ids)
    cfg.set_additional_args(exp_name=args.exp_name,
                            test_batch_size=args.test_batch_size,
                            encoder_setting=args.encoder_setting,
                            decoder_setting=args.decoder_setting,
                            pretrained_model_path=args.pretrained_model_path,
                            agora_benchmark=args.agora_benchmark,
                            testset=args.testset,
                            )
    hint_kwargs = {}
    for name in ('hint_root', 'hint_split', 'hint_sources',
                 'hint_max_samples', 'hint_sample_interval'):
        value = getattr(args, name)
        if value is not None:
            hint_kwargs[name] = value
    if hint_kwargs:
        cfg.set_additional_args(**hint_kwargs)
    cfg.interhand_eval_split = args.interhand_eval_split
    # Decoder isolation experiment: must match how the checkpoint was trained, set
    # before Tester._make_model (get_model('test1') reads it).
    cfg.use_poseur_hand_decoder = args.use_poseur_hand_decoder
    cudnn.benchmark = True
    from common.base import Tester

    base_result_dir = cfg.result_dir
    base_vis_dir = cfg.vis_dir
    base_log_dir = cfg.log_dir
    result_filename = _testset_result_name(args)

    if not use_pytorch_decoder:
        # Track B: normal/OSX eval overlays the trained hand modules from a single
        # continue_train_path (Tester._make_model). The multi-checkpoint sweep
        # (continue_train_paths) stays pytorch-only. Empty path -> stock OSX (the
        # 19.58 baseline); non-empty -> the fine-tuned OSX fair baseline.
        if args.continue_train_paths:
            print("[eval] decoder_setting != pytorch: continue_train_paths(多 ckpt 扫描)仅 pytorch 支持，忽略")
        cfg.continue_train_path = args.continue_train_path or ""
        if cfg.continue_train_path:
            print(f"[eval] track B: normal 评估叠加微调手模块 {cfg.continue_train_path}")
        _set_base_output_dirs(base_result_dir, base_vis_dir, base_log_dir)
        _run_eval_for_current_checkpoint(args, Tester, result_filename)
        return

    if not checkpoint_paths:
        cfg.continue_train_path = ""
        _set_eval_output_dirs(base_result_dir, base_vis_dir, base_log_dir,
                              'pretrained')
        _run_eval_for_current_checkpoint(args, Tester, result_filename)
        return

    for checkpoint_path in checkpoint_paths:
        run_name = _checkpoint_run_name(checkpoint_path)
        cfg.continue_train_path = checkpoint_path
        _set_eval_output_dirs(base_result_dir, base_vis_dir, base_log_dir,
                              run_name)

        print("=" * 80)
        print(f"[eval] checkpoint: {checkpoint_path}")
        print(f"[eval] result_dir: {cfg.result_dir}")
        _run_eval_for_current_checkpoint(args, Tester, result_filename)

if __name__ == "__main__":
    main()
