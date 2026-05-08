import os
import os.path as osp
import numpy as np
from config import cfg
import copy
import json
import cv2
import torch
from common.utils.human_models import smpl_x
from common.utils.preprocessing import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output
from common.utils.transforms import rigid_align
import random

class HAA500(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.MAX_RETRY = 10
        self.retry = 0
        self.read_success = 0
        self.transform = transform
        self.data_split = data_split
        
        self.data_path = osp.join(cfg.data_dir, 'Haa500')
        self.img_path = osp.join(self.data_path, 'images/haa500')
        self.annot_path = osp.join(self.data_path, 'data/HAA500/annotations')
        
        # 134 关键点标准定义
        self.joint_set = {
            'joint_num': 134,
            'joints_name': \
                (
                'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
                'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'L_Big_toe',
                'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel',  # body part (24)
                'L_Wrist_Hand', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2',
                'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1',
                'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4',  # left hand (21)
                'R_Wrist_Hand', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2',
                'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1',
                'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4',  # right hand (21)
                *['Face_' + str(i) for i in range(56, 73)],  # face contour
                *['Face_' + str(i) for i in range(5, 56)]  # face
                ),
            'flip_pairs': \
                ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (18, 21), (19, 22), (20, 23),
                 (24, 45), (25, 46), (26, 47), (27, 48), (28, 49), (29, 50), (30, 51), (31, 52), (32, 53), (33, 54),
                 (34, 55), (35, 56), (36, 57), (37, 58), (38, 59), (39, 60), (40, 61), (41, 62), (42, 63), (43, 64),
                 (44, 65),  # hand part
                 (66, 82), (67, 81), (68, 80), (69, 79), (70, 78), (71, 77), (72, 76), (73, 75),  # face contour
                 (83, 92), (84, 91), (85, 90), (86, 89), (87, 88),  # face eyebrow
                 (97, 101), (98, 100),  # face below nose
                 (102, 111), (103, 110), (104, 109), (105, 108), (106, 113), (107, 112),  # face eyes
                 (114, 120), (115, 119), (116, 118), (121, 125), (122, 124),  # face mouth
                 (126, 130), (127, 129), (131, 133)  # face lip
                 )
        }
        self.datalist = self.load_data()

    def merge_joint(self, body_img, foot_img, lhand_img, rhand_img, face_img):
        lhip_idx = self.joint_set['joints_name'].index('L_Hip')
        rhip_idx = self.joint_set['joints_name'].index('R_Hip')
        
        pelvis = (body_img[lhip_idx, :] + body_img[rhip_idx, :]) * 0.5
        pelvis[2] = body_img[lhip_idx, 2] * body_img[rhip_idx, 2] 
        pelvis = pelvis.reshape(1, 3)

        joint_img = np.concatenate((body_img, pelvis, foot_img, lhand_img, rhand_img, face_img)).astype(np.float32)
        return joint_img

    def load_data(self):
        import pickle
        # 此处统一加载 pkl 文件
        pkl_file = osp.join(self.annot_path, f'HAA500_{self.data_split}.pkl')
        print(f"Loading HAA500 {self.data_split} data from {pkl_file} ...")
        
        with open(pkl_file, 'rb') as f:
            raw_datalist = pickle.load(f)
            
        formatted_datalist =[]
        
        for data in raw_datalist:
            img_path = osp.join(self.img_path, data['file_name'])
            
            # 动态过滤没有被抽取到的帧图片
            if not os.path.exists(img_path):
                continue
                
            img_shape = data['img_shape']
            
            joint_img = self.merge_joint(
                data['body_kpts'], data['foot_kpts'], 
                data['lefthand_kpts'], data['righthand_kpts'], data['face_kpts']
            )
            joint_valid = (joint_img[:, 2].copy().reshape(-1, 1) > 0).astype(np.float32)
            joint_img[:, 2] = 0

            # 填充 Wrist / Face_18
            for body_name, part_name in (('L_Wrist', 'L_Wrist_Hand'), ('R_Wrist', 'R_Wrist_Hand'), ('Nose', 'Face_18')):
                b_idx = self.joint_set['joints_name'].index(body_name)
                p_idx = self.joint_set['joints_name'].index(part_name)
                if joint_valid[p_idx, 0] == 0:
                    joint_img[p_idx] = joint_img[b_idx]
                    joint_valid[p_idx] = joint_valid[b_idx]

            bbox = process_bbox(data['bbox'], img_shape[1], img_shape[0])
            if bbox is None: continue

            lhand_bbox = data['lhand_bbox']
            if lhand_bbox is not None:
                lhand_bbox = np.array(lhand_bbox).reshape(4); lhand_bbox[2:] += lhand_bbox[:2]
            rhand_bbox = data['rhand_bbox']
            if rhand_bbox is not None:
                rhand_bbox = np.array(rhand_bbox).reshape(4); rhand_bbox[2:] += rhand_bbox[:2]
            face_bbox = data['face_bbox']
            if face_bbox is not None:
                face_bbox = np.array(face_bbox).reshape(4); face_bbox[2:] += face_bbox[:2]

            # ========================================================
            # [核心修复]：向 smplx_param 强行注入 OSX 框架必需的有效性标志位
            # ========================================================
            smplx_param = data.get('smplx_param', None)
            if smplx_param is not None:
                smplx_param.setdefault('lhand_valid', True)
                smplx_param.setdefault('rhand_valid', True)
                smplx_param.setdefault('face_valid', True)
            # ========================================================

            formatted_datalist.append({
                'ann_id': data['ann_id'], 'img_path': img_path, 'img_shape': img_shape,
                'bbox': bbox, 'orig_bbox': data['bbox'],
                'joint_img': joint_img, 'joint_valid': joint_valid,
                'smplx_param': smplx_param,
                'lhand_bbox': lhand_bbox, 'rhand_bbox': rhand_bbox, 'face_bbox': face_bbox,
                'camera_params': data['camera_params']
            })

        return formatted_datalist

    def process_hand_face_bbox(self, bbox, do_flip, img_shape, img2bb_trans):
        if bbox is None:
            bbox = np.array([0, 0, 1, 1], dtype=np.float32).reshape(2, 2)
            bbox_valid = float(False)
        else:
            bbox = bbox.reshape(2, 2)
            if do_flip:
                bbox[:, 0] = img_shape[1] - bbox[:, 0] - 1
                bbox[0, 0], bbox[1, 0] = bbox[1, 0].copy(), bbox[0, 0].copy()

            bbox = bbox.reshape(4).tolist()
            xmin, ymin, xmax, ymax = bbox
            bbox = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32).reshape(4, 2)

            bbox_xy1 = np.concatenate((bbox, np.ones_like(bbox[:, :1])), 1)
            bbox = np.dot(img2bb_trans, bbox_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            bbox[:, 0] = bbox[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            bbox[:, 1] = bbox[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

            xmin = np.min(bbox[:, 0])
            xmax = np.max(bbox[:, 0])
            ymin = np.min(bbox[:, 1])
            ymax = np.max(bbox[:, 1])
            bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

            bbox_valid = float(True)
            bbox = bbox.reshape(2, 2)

        return bbox, bbox_valid

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape = data['img_path'], data['img_shape']

        try:
            img = load_img(img_path)
        except (IOError, OSError) as e:
            if self.retry < self.MAX_RETRY:
                # print(f"[HAA500] Warning: Cannot read {img_path}, skip to random sample")
                new_idx = random.randint(0, len(self) - 1)
                self.retry += 1
                return self.__getitem__(new_idx)
            else:
                print(f"[HAA500] Warning: 连续 {self.MAX_RETRY} 次读取失败")
                self.retry = 0
                new_idx = random.randint(0, len(self) - 1)
                return self.__getitem__(new_idx)
        bbox = data['bbox']
        
        # augmentation 内部会根据 self.data_split 是 'train' 还是 'test' 自动选择随机裁剪还是中心裁剪，do_flip 测试时一定是 False
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32)) / 255.

        lhand_bbox, lhand_bbox_valid = self.process_hand_face_bbox(data['lhand_bbox'], do_flip, img_shape, img2bb_trans)
        rhand_bbox, rhand_bbox_valid = self.process_hand_face_bbox(data['rhand_bbox'], do_flip, img_shape, img2bb_trans)
        face_bbox, face_bbox_valid = self.process_hand_face_bbox(data['face_bbox'], do_flip, img_shape, img2bb_trans)
        
        if do_flip:
            lhand_bbox, rhand_bbox = rhand_bbox, lhand_bbox
            lhand_bbox_valid, rhand_bbox_valid = rhand_bbox_valid, lhand_bbox_valid
            
        lhand_bbox_center = (lhand_bbox[0] + lhand_bbox[1]) / 2.
        rhand_bbox_center = (rhand_bbox[0] + rhand_bbox[1]) / 2.
        face_bbox_center = (face_bbox[0] + face_bbox[1]) / 2.
        lhand_bbox_size = lhand_bbox[1] - lhand_bbox[0]
        rhand_bbox_size = rhand_bbox[1] - rhand_bbox[0]
        face_bbox_size = face_bbox[1] - face_bbox[0]

        dummy_coord = np.zeros((self.joint_set['joint_num'], 3), dtype=np.float32)
        joint_img = data['joint_img']
        joint_img = np.concatenate((joint_img[:, :2], np.zeros_like(joint_img[:, :1])), 1)
        
        joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(
            joint_img, dummy_coord, data['joint_valid'], do_flip, img_shape, 
            self.joint_set['flip_pairs'], img2bb_trans, rot, self.joint_set['joints_name'], smpl_x.joints_name)

        smplx_param = data['smplx_param']
        smplx_cam_trans = np.zeros(3, dtype=np.float32)
        if smplx_param is not None:
            # --->[新增这行] 提取全局平移参数
            smplx_cam_trans = np.array(smplx_param.get('trans', [0, 0, 0]), dtype=np.float32).reshape(3)

             # [核心修复]：解析并重组相机参数，满足 OSX 底层函数对 R, t, focal, princpt 的需求
            # =========================================================
            raw_cam = data.get('camera_params', None)
            if raw_cam is not None:
                # 从 cameraMatrix 中提取焦距和光心
                fx = raw_cam['cameraMatrix'][0][0]
                fy = raw_cam['cameraMatrix'][1][1]
                cx = raw_cam['cameraMatrix'][0][2]
                cy = raw_cam['cameraMatrix'][1][2]
                
                cam_param = {
                    # 'R': np.array(raw_cam['rmat'], dtype=np.float32),
                    # 't': np.array(raw_cam['tvec'], dtype=np.float32).reshape(1, 3),
                    'focal': [fx, fy],
                    'princpt': [cx, cy]
                }
            else:
                # 兜底操作：如果这条数据没有相机参数，传入默认值字典，防止 'R' in cam_param 报 NoneType 错误
                cam_param = {'focal': cfg.focal, 'princpt': cfg.princpt}
                print(f"[Warning] HAA500 {self.data_split} data {data['ann_id']} has no camera_params, using default values.")
            smplx_joint_img, smplx_joint_cam, smplx_joint_trunc, smplx_pose, smplx_shape, smplx_expr, smplx_pose_valid, smplx_joint_valid, smplx_expr_valid, smplx_mesh_cam_orig \
                = process_human_model_output(smplx_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx')
            is_valid_fit = True
        else:
            smplx_joint_img = np.zeros((smpl_x.joint_num, 3), dtype=np.float32)
            smplx_joint_cam = np.zeros((smpl_x.joint_num, 3), dtype=np.float32)
            smplx_joint_trunc = np.zeros((smpl_x.joint_num, 1), dtype=np.float32)
            smplx_joint_valid = np.zeros((smpl_x.joint_num), dtype=np.float32)
            smplx_pose = np.zeros((smpl_x.orig_joint_num * 3), dtype=np.float32)
            smplx_shape = np.zeros((smpl_x.shape_param_dim), dtype=np.float32)
            smplx_expr = np.zeros((smpl_x.expr_code_dim), dtype=np.float32)
            smplx_pose_valid = np.zeros((smpl_x.orig_joint_num), dtype=np.float32)
            smplx_expr_valid = False
            is_valid_fit = False
            smplx_mesh_cam_orig = np.zeros((smpl_x.vertex_num, 3), dtype=np.float32)

        smplx_pose_valid = np.tile(smplx_pose_valid[:, None], (1, 3)).reshape(-1)
        smplx_joint_valid = smplx_joint_valid[:, None]
        smplx_joint_trunc = smplx_joint_valid * smplx_joint_trunc

        if not is_valid_fit:
            smplx_pose_valid[:] = 0
            smplx_joint_valid[:] = 0
            smplx_joint_trunc[:] = 0
            smplx_shape_valid = False
        else:
            smplx_shape_valid = True

        inputs = {'img': img}
        
        # 不论 Train/Test，全部向外输送 targets 真实标注，以供 Test 时进行指标评测
        targets = {
            'joint_img': joint_img, 
            'joint_cam': joint_cam, 
            'smplx_joint_img': smplx_joint_img,
            'smplx_joint_cam': smplx_joint_cam,
            'smplx_pose': smplx_pose, 
            'smplx_shape': smplx_shape, 
            'smplx_expr': smplx_expr,
            'smplx_cam_trans': smplx_cam_trans,
            'smplx_mesh_cam': smplx_mesh_cam_orig,
            'lhand_bbox_center': lhand_bbox_center,
            'lhand_bbox_size': lhand_bbox_size, 
            'rhand_bbox_center': rhand_bbox_center,
            'rhand_bbox_size': rhand_bbox_size,
            'face_bbox_center': face_bbox_center, 
            'face_bbox_size': face_bbox_size
        }
        
        meta_info = {
            'bb2img_trans': bb2img_trans,
            'joint_valid': joint_valid, 
            'joint_trunc': joint_trunc, 
            'smplx_joint_valid': smplx_joint_valid,
            'smplx_joint_trunc': smplx_joint_trunc,
            'smplx_pose_valid': smplx_pose_valid, 
            'smplx_shape_valid': float(smplx_shape_valid),
            'smplx_expr_valid': float(smplx_expr_valid), 
            'is_3D': float(is_valid_fit),
            'lhand_bbox_valid': lhand_bbox_valid,
            'rhand_bbox_valid': rhand_bbox_valid, 
            'face_bbox_valid': face_bbox_valid
        }
        
        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'pa_mpvpe_all':[], 'pa_mpvpe_hand':[], 'pa_mpvpe_face': [], 'mpvpe_all':[], 'mpvpe_hand':[],
                       'mpvpe_face':[], 'pa_mpjpe_body':[], 'pa_mpjpe_hand':[]}
                       
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            ann_id = annot['ann_id']
            out = outs[n]

            # 计算评价指标 (仅当框架输出了 Target Mesh 时)
            # print(f"out keys: {list(out.keys())}")
            if 'smplx_mesh_cam_target' in out and out['smplx_mesh_cam_target'] is not None:
                mesh_gt = out['smplx_mesh_cam_target']
                mesh_out = out['smplx_mesh_cam']

                # PA-MPVPE from all vertices
                mesh_out_align = rigid_align(mesh_out, mesh_gt)
                eval_result['pa_mpvpe_all'].append(np.sqrt(np.sum((mesh_out_align - mesh_gt) ** 2, 1)).mean() * 1000)
                
                # MPVPE from all vertices
                mesh_out_align = mesh_out - np.dot(smpl_x.J_regressor, mesh_out)[smpl_x.J_regressor_idx['pelvis'], None, :] + np.dot(smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['pelvis'], None, :]
                eval_result['mpvpe_all'].append(np.sqrt(np.sum((mesh_out_align - mesh_gt) ** 2, 1)).mean() * 1000)

                # MPVPE from hand vertices
                mesh_gt_lhand = mesh_gt[smpl_x.hand_vertex_idx['left_hand'], :]
                mesh_out_lhand = mesh_out[smpl_x.hand_vertex_idx['left_hand'], :]
                mesh_out_lhand_align = rigid_align(mesh_out_lhand, mesh_gt_lhand)
                
                mesh_gt_rhand = mesh_gt[smpl_x.hand_vertex_idx['right_hand'], :]
                mesh_out_rhand = mesh_out[smpl_x.hand_vertex_idx['right_hand'], :]
                mesh_out_rhand_align = rigid_align(mesh_out_rhand, mesh_gt_rhand)
                
                eval_result['pa_mpvpe_hand'].append((np.sqrt(np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000 + np.sqrt(np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000) / 2.)

                # MPVPE from face vertices
                mesh_gt_face = mesh_gt[smpl_x.face_vertex_idx, :]
                mesh_out_face = mesh_out[smpl_x.face_vertex_idx, :]
                mesh_out_face_align = rigid_align(mesh_out_face, mesh_gt_face)
                eval_result['pa_mpvpe_face'].append(np.sqrt(np.sum((mesh_out_face_align - mesh_gt_face) ** 2, 1)).mean() * 1000)

                # PA-MPJPE from body joints
                joint_gt_body = np.dot(smpl_x.j14_regressor, mesh_gt)
                joint_out_body = np.dot(smpl_x.j14_regressor, mesh_out)
                joint_out_body_align = rigid_align(joint_out_body, joint_gt_body)
                eval_result['pa_mpjpe_body'].append(np.sqrt(np.sum((joint_out_body_align - joint_gt_body) ** 2, 1)).mean() * 1000)

                # PA-MPJPE from hand joints
                joint_gt_lhand = np.dot(smpl_x.orig_hand_regressor['left'], mesh_gt)
                joint_out_lhand = np.dot(smpl_x.orig_hand_regressor['left'], mesh_out)
                joint_out_lhand_align = rigid_align(joint_out_lhand, joint_gt_lhand)
                
                joint_gt_rhand = np.dot(smpl_x.orig_hand_regressor['right'], mesh_gt)
                joint_out_rhand = np.dot(smpl_x.orig_hand_regressor['right'], mesh_out)
                joint_out_rhand_align = rigid_align(joint_out_rhand, joint_gt_rhand)
                eval_result['pa_mpjpe_hand'].append((np.sqrt(np.sum((joint_out_lhand_align - joint_gt_lhand) ** 2, 1)).mean() * 1000 + np.sqrt(np.sum((joint_out_rhand_align - joint_gt_rhand) ** 2, 1)).mean() * 1000) / 2.)

        return eval_result

    def print_eval_result(self, eval_result):
        if len(eval_result['pa_mpvpe_all']) > 0:
            print('--- HAA500 Evaluation Results ---')
            print('PA MPVPE (All): %.2f mm' % np.mean(eval_result['pa_mpvpe_all']))
            print('PA MPVPE (Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_hand']))
            print('PA MPVPE (Face): %.2f mm' % np.mean(eval_result['pa_mpvpe_face']))
            print()
            print('MPVPE (All): %.2f mm' % np.mean(eval_result['mpvpe_all']))
            print('PA MPJPE (Body): %.2f mm' % np.mean(eval_result['pa_mpjpe_body']))
            print('PA MPJPE (Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_hand']))

            # 写入结果文件
            f = open(os.path.join(cfg.result_dir, 'HAA500_result.txt'), 'w')
            f.write(f'HAA500 dataset: \n')
            f.write('PA MPVPE (All): %.2f mm\n' % np.mean(eval_result['pa_mpvpe_all']))
            f.write('PA MPVPE (Hands): %.2f mm\n' % np.mean(eval_result['pa_mpvpe_hand']))
            f.write('PA MPVPE (Face): %.2f mm\n' % np.mean(eval_result['pa_mpvpe_face']))
            f.write('MPVPE (All): %.2f mm\n' % np.mean(eval_result['mpvpe_all']))
            f.write('PA MPJPE (Body): %.2f mm\n' % np.mean(eval_result['pa_mpjpe_body']))
            f.write('PA MPJPE (Hands): %.2f mm\n' % np.mean(eval_result['pa_mpjpe_hand']))
            f.close()
        else:
            print("Evaluation finished. No Ground Truth 3D Mesh output found for distance metrics.")