import os
import os.path as osp
import numpy as np
from config import cfg
import copy
import json
import pickle
import cv2
import torch
from pycocotools.coco import COCO
from common.utils.human_models import smpl_x
from common.utils.preprocessing import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output, resize_bbox
from common.utils.transforms import rigid_align
from torch.utils.data.dataset import Dataset
import random

_UBODY_ANNOTATION_CACHE_VERSION = 1


def _resolve_annotation_cache_path(json_path, scene):
    cache_dir = getattr(cfg, 'ubody_annotation_cache_dir', '')
    if cache_dir:
        if not osp.isabs(cache_dir):
            cache_dir = osp.join(cfg.root_dir, cache_dir)
        return osp.join(cache_dir, scene, osp.splitext(osp.basename(json_path))[0] + '.pkl')
    return osp.splitext(json_path)[0] + '.pkl'


def _unwrap_annotation_cache(payload, json_path):
    if not (
        isinstance(payload, dict) and
        payload.get('_ubody_annotation_cache_version') == _UBODY_ANNOTATION_CACHE_VERSION and
        'data' in payload
    ):
        # Also accept plain pickle dumps of the JSON object for compatibility
        # with ad-hoc caches.
        return payload, None

    meta = payload.get('json', {})
    try:
        stat = os.stat(json_path)
    except OSError as e:
        if payload.get('allow_missing_json', False):
            return payload['data'], None
        return None, 'cannot stat source JSON: %s' % e

    if meta.get('size') != stat.st_size or meta.get('mtime_ns') != stat.st_mtime_ns:
        return None, 'source JSON changed'
    return payload['data'], None


def _load_annotation_data(json_path, scene, name):
    pkl_path = _resolve_annotation_cache_path(json_path, scene)
    if getattr(cfg, 'ubody_use_pkl_annotation', True) and osp.isfile(pkl_path):
        with open(pkl_path, 'rb') as f:
            payload = pickle.load(f)
        data, stale_reason = _unwrap_annotation_cache(payload, json_path)
        if data is not None:
            print('load UBody %s annotation from %s.' % (name, pkl_path))
            return data
        print('ignore stale UBody %s annotation cache %s (%s).' % (name, pkl_path, stale_reason))

    with open(json_path, 'r') as f:
        print('load UBody %s annotation from %s.' % (name, json_path))
        return json.load(f)


def _load_coco_from_annotation(json_path, scene):
    dataset = _load_annotation_data(json_path, scene, 'keypoint')
    db = COCO()
    db.dataset = dataset
    db.createIndex()
    return db


class UBody_Part(torch.utils.data.Dataset):
    def __init__(self, transform, data_split, scene):
        self.transform = transform
        self.data_split = data_split
        self.scene = scene
        self.img_path = osp.join(cfg.data_dir, 'UBody', 'images', scene)
        annot_root = getattr(cfg, 'ubody_annotation_dir', '')
        if annot_root:
            if not osp.isabs(annot_root):
                annot_root = osp.join(cfg.root_dir, annot_root)
        else:
            annot_root = osp.join(cfg.data_dir, 'UBody', 'annotations')
        self.annot_path = osp.join(annot_root, scene, 'keypoint_annotation.json')
        self.smplx_annot_path = osp.join(annot_root, scene, 'smplx_annotation.json')
        self.test_video_list_path = osp.join(cfg.data_dir, 'UBody', 'splits', 'intra_scene_test_list.npy')
        # Some local UBody image files can exist but fail cv2 decoding
        # (truncated / corrupt). During training, replace that whole sample with
        # another readable sample from the same scene instead of killing the
        # DataLoader worker. Test remains strict to avoid corrupting metrics.
        self._fallback_max_tries = 200
        self._warned_bad_imgs = set()

        # mscoco joint set
        self.joint_set = {
            'joint_num': 134,  # body 24 (23 + pelvis), lhand 21, rhand 21, face 68
            'joints_name': \
                (
                'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
                'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'L_Big_toe',
                'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel',  # body part
                'L_Wrist_Hand', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2',
                'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1',
                'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4',  # left hand
                'R_Wrist_Hand', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2',
                'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1',
                'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4',  # right hand
                *['Face_' + str(i) for i in range(56, 73)],  # face contour
                *['Face_' + str(i) for i in range(5, 56)]  # face
                ),
            'flip_pairs': \
                ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (18, 21), (19, 22), (20, 23),
                 # body part
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

    def merge_joint(self, joint_img, feet_img, lhand_img, rhand_img, face_img):
        # pelvis
        lhip_idx = self.joint_set['joints_name'].index('L_Hip')
        rhip_idx = self.joint_set['joints_name'].index('R_Hip')
        pelvis = (joint_img[lhip_idx, :] + joint_img[rhip_idx, :]) * 0.5
        pelvis[2] = joint_img[lhip_idx, 2] * joint_img[rhip_idx, 2]  # joint_valid
        pelvis = pelvis.reshape(1, 3)

        # feet
        lfoot = feet_img[:3, :]
        rfoot = feet_img[3:, :]

        joint_img = np.concatenate((joint_img, pelvis, lfoot, rfoot, lhand_img, rhand_img, face_img)).astype(
            np.float32)  # self.joint_set['joint_num'], 3
        return joint_img

    def load_data(self):
        db = _load_coco_from_annotation(self.annot_path, self.scene)
        smplx_params = _load_annotation_data(self.smplx_annot_path, self.scene, 'smplx')
        test_video_list = np.load(self.test_video_list_path)
        print(f'load test_video_list from {self.test_video_list_path}.')
        test_video_list = test_video_list.tolist()
        # train mode
        if self.data_split == 'train':
            datalist = []
            i = 0
            for aid in db.anns.keys():
                i = i + 1
                ann = db.anns[aid]
                sample_i = ann.get('_ubody_orig_ann_order', i)
                if sample_i % cfg.train_sample_interval != 0:
                    continue
                img = db.loadImgs(ann['image_id'])[0]
                if img['file_name'].startswith('/'):
                    file_name = img['file_name'][1:]   # [1:] means delete '/'
                else:
                    file_name = img['file_name']
                video_name = file_name.split('/')[-2]
                if 'Trim' in video_name:
                    video_name = video_name.split('_Trim')[0]
                if video_name in test_video_list: continue   # exclude the test video
                img_path = osp.join(self.img_path, file_name)
                if not os.path.exists(img_path): continue

                # exclude the samples that are crowd or have few visible keypoints
                if ann['iscrowd'] or (ann['num_keypoints']==0): continue

                # bbox
                bbox = process_bbox(ann['bbox'], img['width'], img['height'])
                if bbox is None: continue

                # joint coordinates
                joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
                foot_img = np.array(ann['foot_kpts'], dtype=np.float32).reshape(-1, 3)
                lhand_img = np.array(ann['lefthand_kpts'], dtype=np.float32).reshape(-1, 3)
                rhand_img = np.array(ann['righthand_kpts'], dtype=np.float32).reshape(-1, 3)
                face_img = np.array(ann['face_kpts'], dtype=np.float32).reshape(-1, 3)
                joint_img = self.merge_joint(joint_img, foot_img, lhand_img, rhand_img, face_img)

                joint_valid = (joint_img[:, 2].copy().reshape(-1, 1) > 0).astype(np.float32)
                joint_img[:, 2] = 0

                # use body annotation to fill hand/face annotation
                for body_name, part_name in (
                ('L_Wrist', 'L_Wrist_Hand'), ('R_Wrist', 'R_Wrist_Hand'), ('Nose', 'Face_18')):
                    if joint_valid[self.joint_set['joints_name'].index(part_name), 0] == 0:
                        joint_img[self.joint_set['joints_name'].index(part_name)] = joint_img[
                            self.joint_set['joints_name'].index(body_name)]
                        joint_valid[self.joint_set['joints_name'].index(part_name)] = joint_valid[
                            self.joint_set['joints_name'].index(body_name)]

                # hand/face bbox
                if ann['lefthand_valid']:
                    lhand_bbox = np.array(ann['lefthand_box']).reshape(4)
                    lhand_bbox[2:] += lhand_bbox[:2]  # xywh -> xyxy
                    lhand_bbox = resize_bbox(lhand_bbox, scale=0.65)  # 1.5*0.8=1.2
                else:
                    lhand_bbox = None
                if ann['righthand_valid']:
                    rhand_bbox = np.array(ann['righthand_box']).reshape(4)
                    rhand_bbox[2:] += rhand_bbox[:2]  # xywh -> xyxy
                    rhand_bbox = resize_bbox(rhand_bbox, scale=0.65)  # 1.5*0.8=1.2
                else:
                    rhand_bbox = None
                if ann['face_valid']:
                    # rough face bbox
                    face_bbox = [min(face_img[:, 0]), min(face_img[:, 1]),
                            max(face_img[:, 0]), max(face_img[:, 1])]
                    face_bbox = resize_bbox(face_bbox, scale=1.2)
                    face_bbox = np.array(face_bbox)
                    # face_bbox = np.array(ann['face_box']).reshape(4)
                    # face_bbox[2:] += face_bbox[:2]  # xywh -> xyxy
                else:
                    face_bbox = None

                if ann['valid_label'] == 0 or str(aid) not in smplx_params: continue

                smplx_param = smplx_params[str(aid)]
                if 'lhand_valid' not in smplx_param['smplx_param']:
                    smplx_param['smplx_param']['lhand_valid'] = ann['lefthand_valid']
                    smplx_param['smplx_param']['rhand_valid'] = ann['righthand_valid']
                    smplx_param['smplx_param']['face_valid'] = ann['face_valid']


                data_dict = {'img_path': img_path, 'img_shape': (img['height'], img['width']), 'bbox': bbox,
                             'joint_img': joint_img, 'joint_valid': joint_valid, 'smplx_param': smplx_param,
                             'lhand_bbox': lhand_bbox, 'rhand_bbox': rhand_bbox, 'face_bbox': face_bbox}
                datalist.append(data_dict)

            return datalist

        # test mode
        else:
            datalist = []
            i = 0
            for aid in db.anns.keys():
                i = i + 1
                ann = db.anns[aid]
                sample_i = ann.get('_ubody_orig_ann_order', i)
                if sample_i % cfg.test_sample_interval != 0:
                    continue
                img = db.loadImgs(ann['image_id'])[0]
                if img['file_name'].startswith('/'):
                    file_name = img['file_name'][1:]  # [1:] means delete '/'
                else:
                    file_name = img['file_name']
                video_name = file_name.split('/')[-2]
                if 'Trim' in video_name:
                    video_name = video_name.split('_Trim')[0]
                if video_name not in test_video_list: continue  # exclude the train video
                img_path = osp.join(self.img_path, file_name)
                if not os.path.exists(img_path): continue

                # exclude the samples that are crowd or have few visible keypoints
                if ann['iscrowd'] or (ann['num_keypoints']==0): continue

                if ann['valid_label'] == 0 or str(aid) not in smplx_params: continue

                # bbox
                bbox = process_bbox(ann['bbox'], img['width'], img['height'])
                if bbox is None: continue

                # joint coordinates
                joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
                foot_img = np.array(ann['foot_kpts'], dtype=np.float32).reshape(-1, 3)
                lhand_img = np.array(ann['lefthand_kpts'], dtype=np.float32).reshape(-1, 3)
                rhand_img = np.array(ann['righthand_kpts'], dtype=np.float32).reshape(-1, 3)
                face_img = np.array(ann['face_kpts'], dtype=np.float32).reshape(-1, 3)
                joint_img = self.merge_joint(joint_img, foot_img, lhand_img, rhand_img, face_img)

                joint_valid = (joint_img[:, 2].copy().reshape(-1, 1) > 0).astype(np.float32)
                joint_img[:, 2] = 0

                # use body annotation to fill hand/face annotation
                for body_name, part_name in (
                        ('L_Wrist', 'L_Wrist_Hand'), ('R_Wrist', 'R_Wrist_Hand'), ('Nose', 'Face_18')):
                    if joint_valid[self.joint_set['joints_name'].index(part_name), 0] == 0:
                        joint_img[self.joint_set['joints_name'].index(part_name)] = joint_img[
                            self.joint_set['joints_name'].index(body_name)]
                        joint_valid[self.joint_set['joints_name'].index(part_name)] = joint_valid[
                            self.joint_set['joints_name'].index(body_name)]

                # hand/face bbox
                if ann['lefthand_valid']:
                    lhand_bbox = np.array(ann['lefthand_box']).reshape(4)
                    lhand_bbox[2:] += lhand_bbox[:2]  # xywh -> xyxy
                    lhand_bbox = resize_bbox(lhand_bbox, scale=0.65)  # 1.5*0.8=1.2
                else:
                    lhand_bbox = None
                if ann['righthand_valid']:
                    rhand_bbox = np.array(ann['righthand_box']).reshape(4)
                    rhand_bbox[2:] += rhand_bbox[:2]  # xywh -> xyxy
                    rhand_bbox = resize_bbox(rhand_bbox, scale=0.65)  # 1.5*0.8=1.2
                else:
                    rhand_bbox = None
                if ann['face_valid']:
                    face_img = np.array(ann['face_kpts'], dtype=np.float32).reshape(-1, 3)
                    face_bbox = [min(face_img[:, 0]), min(face_img[:, 1]),
                                 max(face_img[:, 0]), max(face_img[:, 1])]
                    face_bbox = resize_bbox(face_bbox, scale=1.2)
                    face_bbox = np.array(face_bbox)
                else:
                    face_bbox = None

                if str(aid) in smplx_params:
                    smplx_param = smplx_params[str(aid)]
                    if 'lhand_valid' not in smplx_param['smplx_param']:
                        smplx_param['smplx_param']['lhand_valid'] = ann['lefthand_valid']
                        smplx_param['smplx_param']['rhand_valid'] = ann['righthand_valid']
                        smplx_param['smplx_param']['face_valid'] = ann['face_valid']
                else:
                    smplx_param = None

                data_dict = {'img_path': img_path, 'ann_id': aid, 'img_shape': (img['height'],img['width']), 'bbox': bbox,
                             'joint_img': joint_img, 'joint_valid': joint_valid, 'smplx_param': smplx_param,
                             'lhand_bbox': lhand_bbox, 'rhand_bbox': rhand_bbox, 'face_bbox': face_bbox}
                datalist.append(data_dict)
            return datalist

    def process_hand_face_bbox(self, bbox, do_flip, img_shape, img2bb_trans):
        if bbox is None:
            bbox = np.array([0, 0, 1, 1], dtype=np.float32).reshape(2, 2)  # dummy value
            bbox_valid = float(False)  # dummy value
        else:
            # reshape to top-left (x,y) and bottom-right (x,y)
            bbox = bbox.reshape(2, 2)

            # flip augmentation
            if do_flip:
                bbox[:, 0] = img_shape[1] - bbox[:, 0] - 1
                bbox[0, 0], bbox[1, 0] = bbox[1, 0].copy(), bbox[0, 0].copy()  # xmin <-> xmax swap

            # make four points of the bbox
            bbox = bbox.reshape(4).tolist()
            xmin, ymin, xmax, ymax = bbox
            bbox = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32).reshape(4, 2)

            # affine transformation (crop, rotation, scale)
            bbox_xy1 = np.concatenate((bbox, np.ones_like(bbox[:, :1])), 1)
            bbox = np.dot(img2bb_trans, bbox_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            bbox[:, 0] = bbox[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            bbox[:, 1] = bbox[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

            # make box a rectangle without rotation
            xmin = np.min(bbox[:, 0]);
            xmax = np.max(bbox[:, 0]);
            ymin = np.min(bbox[:, 1]);
            ymax = np.max(bbox[:, 1]);
            bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

            bbox_valid = float(True)
            bbox = bbox.reshape(2, 2)

        return bbox, bbox_valid

    def process_coco_hand_joint(self, joint_img, joint_valid, do_flip, img_shape, img2bb_trans, rot):
        hand_names = (
            'L_Wrist_Hand',
            'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4',
            'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4',
            'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4',
            'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4',
            'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4',
            'R_Wrist_Hand',
            'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4',
            'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4',
            'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4',
            'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4',
            'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4',
        )
        coord = np.concatenate((joint_img[:, :2], np.zeros_like(joint_img[:, :1])), 1)
        dummy_cam = np.zeros_like(coord)
        hand_joint_img, _, _, hand_joint_trunc = process_db_coord(
            coord, dummy_cam, joint_valid, do_flip, img_shape,
            self.joint_set['flip_pairs'], img2bb_trans, rot,
            self.joint_set['joints_name'], hand_names)
        return hand_joint_img.reshape(2, 21, 3), hand_joint_trunc.reshape(2, 21, 1)

    def __len__(self):
        return len(self.datalist)

    def _load_train_img_with_fallback(self, idx):
        img_path = self.datalist[idx]['img_path']
        try:
            return load_img(img_path), copy.deepcopy(self.datalist[idx])
        except (IOError, OSError) as exc:
            first_error = exc
            if img_path not in self._warned_bad_imgs:
                self._warned_bad_imgs.add(img_path)
                print("[UBody] unreadable image, resampling another sample: %s" % img_path)

        n = len(self.datalist)
        fallback_num = min(n, self._fallback_max_tries)
        for cand_idx in random.sample(range(n), fallback_num):
            if cand_idx == idx:
                continue
            cand = self.datalist[cand_idx]
            cand_path = cand['img_path']
            if cand_path == img_path:
                continue
            try:
                img = load_img(cand_path)
            except (IOError, OSError):
                continue
            cand_shape = cand.get('img_shape')
            if cand_shape is not None and (img.shape[0] != cand_shape[0] or img.shape[1] != cand_shape[1]):
                continue
            return img, copy.deepcopy(cand)

        raise IOError(
            "Fail to read %s and %d random fallback samples were also unreadable"
            % (img_path, fallback_num)
        ) from first_error

    def __getitem__(self, idx):

        # train mode
        if self.data_split == 'train':
            img, data = self._load_train_img_with_fallback(idx)
            img_path, img_shape = data['img_path'], data['img_shape']

            bbox = data['bbox']
            img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
            img = self.transform(img.astype(np.float32)) / 255.

            # hand and face bbox transform
            lhand_bbox, lhand_bbox_valid = self.process_hand_face_bbox(data['lhand_bbox'], do_flip, img_shape,
                                                                       img2bb_trans)
            rhand_bbox, rhand_bbox_valid = self.process_hand_face_bbox(data['rhand_bbox'], do_flip, img_shape,
                                                                       img2bb_trans)
            face_bbox, face_bbox_valid = self.process_hand_face_bbox(data['face_bbox'], do_flip, img_shape,
                                                                     img2bb_trans)
            if do_flip:
                lhand_bbox, rhand_bbox = rhand_bbox, lhand_bbox
                lhand_bbox_valid, rhand_bbox_valid = rhand_bbox_valid, lhand_bbox_valid
            lhand_bbox_center = (lhand_bbox[0] + lhand_bbox[1]) / 2.;
            rhand_bbox_center = (rhand_bbox[0] + rhand_bbox[1]) / 2.;
            face_bbox_center = (face_bbox[0] + face_bbox[1]) / 2.
            lhand_bbox_size = lhand_bbox[1] - lhand_bbox[0];
            rhand_bbox_size = rhand_bbox[1] - rhand_bbox[0];
            face_bbox_size = face_bbox[1] - face_bbox[0];

            # coco gt
            dummy_coord = np.zeros((self.joint_set['joint_num'], 3), dtype=np.float32)
            raw_joint_img = data['joint_img']
            raw_joint_valid = data['joint_valid']
            coco_hand_joint_img, coco_hand_joint_trunc = self.process_coco_hand_joint(
                raw_joint_img, raw_joint_valid, do_flip, img_shape, img2bb_trans, rot)
            joint_img = raw_joint_img
            joint_img = np.concatenate((joint_img[:, :2], np.zeros_like(joint_img[:, :1])), 1)  # x, y, dummy depth
            joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(joint_img, dummy_coord,
                                                                              raw_joint_valid, do_flip, img_shape,
                                                                              self.joint_set['flip_pairs'],
                                                                              img2bb_trans, rot,
                                                                              self.joint_set['joints_name'],
                                                                              smpl_x.joints_name)

            # smplx coordinates and parameters
            smplx_param = data['smplx_param']
            if smplx_param is not None:
                smplx_joint_img, smplx_joint_cam, smplx_joint_trunc, smplx_pose, smplx_shape, smplx_expr, smplx_pose_valid, smplx_joint_valid, smplx_expr_valid, smplx_mesh_cam_orig = process_human_model_output(
                    smplx_param['smplx_param'], smplx_param['cam_param'], do_flip, img_shape, img2bb_trans, rot,
                    'smplx')
                is_valid_fit = True
                smplx_cam_trans = np.array(smplx_param['smplx_param']['trans'], dtype=np.float32)

                """
                # for debug
                _tmp = joint_img.copy()
                _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
                _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
                _img = vis_keypoints(_img, _tmp)
                cv2.imwrite('coco_' + str(idx) + '.jpg', _img)
                """

            else:
                # dummy values
                smplx_joint_img = np.zeros((smpl_x.joint_num, 3), dtype=np.float32)
                smplx_joint_cam = np.zeros((smpl_x.joint_num, 3), dtype=np.float32)
                smplx_joint_trunc = np.zeros((smpl_x.joint_num, 1), dtype=np.float32)
                smplx_joint_valid = np.zeros((smpl_x.joint_num), dtype=np.float32)
                smplx_pose = np.zeros((smpl_x.orig_joint_num * 3), dtype=np.float32)
                smplx_shape = np.zeros((smpl_x.shape_param_dim), dtype=np.float32)
                smplx_expr = np.zeros((smpl_x.expr_code_dim), dtype=np.float32)
                smplx_mesh_cam_orig = np.zeros((smpl_x.vertex_num, 3), dtype=np.float32)
                smplx_pose_valid = np.zeros((smpl_x.orig_joint_num), dtype=np.float32)
                smplx_cam_trans = np.zeros((3), dtype=np.float32)
                smplx_expr_valid = False
                is_valid_fit = False

            # SMPLX pose parameter validity
            smplx_pose_valid = np.tile(smplx_pose_valid[:, None], (1, 3)).reshape(-1)
            # SMPLX joint coordinate validity
            smplx_joint_valid = smplx_joint_valid[:, None]
            smplx_joint_trunc = smplx_joint_valid * smplx_joint_trunc

            # make zero mask for invalid fit
            if not is_valid_fit:
                smplx_pose_valid[:] = 0
                smplx_joint_valid[:] = 0
                smplx_joint_trunc[:] = 0
                smplx_shape_valid = False
            else:
                smplx_shape_valid = True

            inputs = {'img': img, }
            targets = {'joint_img': joint_img, 'joint_cam': joint_cam, 'smplx_joint_img': smplx_joint_img,
                       'smplx_joint_cam': smplx_joint_cam, 'smplx_pose': smplx_pose, 'smplx_shape': smplx_shape,
                       'smplx_expr': smplx_expr, 'smplx_cam_trans': smplx_cam_trans,
                       'coco_hand_joint_img': coco_hand_joint_img,
                       'coco_hand_joint_trunc': coco_hand_joint_trunc,
                       'smplx_mesh_cam': smplx_mesh_cam_orig, 'lhand_bbox_center': lhand_bbox_center,
                       'lhand_bbox_size': lhand_bbox_size, 'rhand_bbox_center': rhand_bbox_center,
                       'rhand_bbox_size': rhand_bbox_size, 'face_bbox_center': face_bbox_center,
                       'face_bbox_size': face_bbox_size}
            meta_info = {'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'smplx_joint_valid': smplx_joint_valid,
                         'smplx_joint_trunc': smplx_joint_trunc, 'smplx_pose_valid': smplx_pose_valid,
                         'smplx_shape_valid': float(smplx_shape_valid), 'smplx_expr_valid': float(smplx_expr_valid),
                         'is_3D': float(False), 'lhand_bbox_valid': lhand_bbox_valid,
                         'rhand_bbox_valid': rhand_bbox_valid, 'face_bbox_valid': face_bbox_valid,
                         'dataset_id': float(2), 'is_interhand': float(False), 'is_bedlam': float(False),
                         'is_ubody': float(True), 'is_hand_only': float(False), 'bb2img_trans': bb2img_trans}
            return inputs, targets, meta_info

        # test mode
        else:
            data = copy.deepcopy(self.datalist[idx])
            img_path, img_shape = data['img_path'], data['img_shape']

            # image load
            img = load_img(img_path)
            bbox = data['bbox']
            img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
            img = self.transform(img.astype(np.float32)) / 255.

            # hand and face bbox transform
            lhand_bbox, lhand_bbox_valid = self.process_hand_face_bbox(data['lhand_bbox'], do_flip, img_shape,
                                                                       img2bb_trans)
            rhand_bbox, rhand_bbox_valid = self.process_hand_face_bbox(data['rhand_bbox'], do_flip, img_shape,
                                                                       img2bb_trans)
            face_bbox, face_bbox_valid = self.process_hand_face_bbox(data['face_bbox'], do_flip, img_shape,
                                                                     img2bb_trans)
            if do_flip:
                lhand_bbox, rhand_bbox = rhand_bbox, lhand_bbox
                lhand_bbox_valid, rhand_bbox_valid = rhand_bbox_valid, lhand_bbox_valid
            lhand_bbox_center = (lhand_bbox[0] + lhand_bbox[1]) / 2.;
            rhand_bbox_center = (rhand_bbox[0] + rhand_bbox[1]) / 2.;
            face_bbox_center = (face_bbox[0] + face_bbox[1]) / 2.
            lhand_bbox_size = lhand_bbox[1] - lhand_bbox[0];
            rhand_bbox_size = rhand_bbox[1] - rhand_bbox[0];
            face_bbox_size = face_bbox[1] - face_bbox[0];

            # coco gt
            dummy_coord = np.zeros((self.joint_set['joint_num'], 3), dtype=np.float32)
            joint_img = data['joint_img']
            joint_img = np.concatenate((joint_img[:, :2], np.zeros_like(joint_img[:, :1])), 1)  # x, y, dummy depth
            joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(joint_img, dummy_coord,
                                                                              data['joint_valid'], do_flip, img_shape,
                                                                              self.joint_set['flip_pairs'],
                                                                              img2bb_trans, rot,
                                                                              self.joint_set['joints_name'],
                                                                              smpl_x.joints_name)


            # smplx coordinates and parameters
            smplx_param = data['smplx_param']
            # if str(data['ann_id'])=='184516':
            #     print(data['ann_id'], smplx_param)
            if smplx_param is not None:
                smplx_joint_img, smplx_joint_cam, smplx_joint_trunc, smplx_pose, smplx_shape, smplx_expr, smplx_pose_valid, smplx_joint_valid, smplx_expr_valid, smplx_mesh_cam_orig = process_human_model_output(
                    smplx_param['smplx_param'], smplx_param['cam_param'], do_flip, img_shape, img2bb_trans, rot,
                    'smplx')
                is_valid_fit = True
                smplx_cam_trans = np.array(smplx_param['smplx_param']['trans'])
                # if str(data['ann_id']) == '184516':
                #     print(data['ann_id'], smplx_pose)
                """
                # for debug
                _tmp = joint_img.copy()
                _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
                _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
                _img = vis_keypoints(_img, _tmp)
                cv2.imwrite('coco_' + str(idx) + '.jpg', _img)
                """

            else:
                # dummy values
                smplx_joint_img = np.zeros((smpl_x.joint_num, 3), dtype=np.float32)
                smplx_joint_cam = np.zeros((smpl_x.joint_num, 3), dtype=np.float32)
                smplx_joint_trunc = np.zeros((smpl_x.joint_num, 1), dtype=np.float32)
                smplx_joint_valid = np.zeros((smpl_x.joint_num), dtype=np.float32)
                smplx_pose = np.zeros((smpl_x.orig_joint_num * 3), dtype=np.float32)
                smplx_shape = np.zeros((smpl_x.shape_param_dim), dtype=np.float32)
                smplx_expr = np.zeros((smpl_x.expr_code_dim), dtype=np.float32)
                smplx_pose_valid = np.zeros((smpl_x.orig_joint_num), dtype=np.float32)
                cam_param_num = 3
                smplx_cam_trans = np.zeros((cam_param_num), dtype=np.float32)
                smplx_expr_valid = False
                is_valid_fit = False

            # SMPLX pose parameter validity
            smplx_pose_valid = np.tile(smplx_pose_valid[:, None], (1, 3)).reshape(-1)
            # SMPLX joint coordinate validity
            smplx_joint_valid = smplx_joint_valid[:, None]
            smplx_joint_trunc = smplx_joint_valid * smplx_joint_trunc

            # make zero mask for invalid fit
            if not is_valid_fit:
                smplx_pose_valid[:] = 0
                smplx_joint_valid[:] = 0
                smplx_joint_trunc[:] = 0
                smplx_shape_valid = False
            else:
                smplx_shape_valid = True

            inputs = {'img': img, }
            targets = {'joint_img': joint_img, 'joint_cam': joint_cam, 'smplx_joint_img': smplx_joint_img,
                       'smplx_joint_cam': smplx_joint_cam, 'smplx_pose': smplx_pose, 'smplx_shape': smplx_shape,
                       'smplx_expr': smplx_expr, 'smplx_cam_trans': smplx_cam_trans,
                       'lhand_bbox_center': lhand_bbox_center,
                       'lhand_bbox_size': lhand_bbox_size, 'rhand_bbox_center': rhand_bbox_center,
                       'rhand_bbox_size': rhand_bbox_size, 'face_bbox_center': face_bbox_center,
                       'face_bbox_size': face_bbox_size}
            meta_info = {'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'smplx_joint_valid': smplx_joint_valid,
                         'smplx_joint_trunc': smplx_joint_trunc, 'smplx_pose_valid': smplx_pose_valid,
                         'smplx_shape_valid': float(smplx_shape_valid), 'smplx_expr_valid': float(smplx_expr_valid),
                         'is_3D': float(False), 'lhand_bbox_valid': lhand_bbox_valid,
                         'rhand_bbox_valid': rhand_bbox_valid, 'face_bbox_valid': face_bbox_valid,
                         'is_interhand': float(False), 'is_bedlam': float(False), 'is_ubody': float(True),
                         'bb2img_trans': bb2img_trans}
            return inputs, targets, meta_info

class UBody(Dataset):
    def __init__(self, transform, mode='test'):
        self.dbs = []

        self.aids = []
        # self.img_paths = []
        self.parts = []
        self.datalist = []
        folder = osp.join(cfg.data_dir, 'UBody', 'images')
        for scene in os.listdir(folder):
            db = UBody_Part(transform, mode, scene=scene)
            self.dbs.append(db)
            self.datalist += db.datalist

        self.db_num = len(self.dbs)
        if self.db_num == 0:
            raise RuntimeError('No UBody scene datasets were found under %s' % folder)
        self.joint_set = self.dbs[0].joint_set
        self.max_db_data_num = max([len(db) for db in self.dbs])
        self.db_len_cumsum = np.cumsum([len(db) for db in self.dbs])
        self.make_same_len = cfg.make_same_len
        print(f'Number of images: {sum([len(db) for db in self.dbs])}')

    def __len__(self):
        # all dbs have the same length
        if self.make_same_len:
            return self.max_db_data_num * self.db_num
        # each db has different length
        else:
            return sum([len(db) for db in self.dbs])

    def __getitem__(self, index):
        # print(self.__len__(), len(self.parts))
        if self.make_same_len:
            db_idx = index // self.max_db_data_num
            data_idx = index % self.max_db_data_num
            if data_idx >= len(self.dbs[db_idx]) * (self.max_db_data_num // len(self.dbs[db_idx])): # last batch: random sampling
                data_idx = random.randint(0,len(self.dbs[db_idx])-1)
            else: # before last batch: use modular
                data_idx = data_idx % len(self.dbs[db_idx])
        else:
            for i in range(self.db_num):
                if index < self.db_len_cumsum[i]:
                    db_idx = i
                    break
            if db_idx == 0:
                data_idx = index
            else:
                data_idx = index - self.db_len_cumsum[db_idx-1]

        return self.dbs[db_idx][data_idx]

    def perspective_transform(self, cam_trans, bb2img_trans, inputs):
        # inputs: [num_points, 3], world coordinate
        x = (inputs[:, 0] + cam_trans[None, 0]) / (
                    inputs[:, 2] + cam_trans[None, 2] + 1e-4) * \
            cfg.focal[0] + cfg.princpt[0]
        y = (inputs[:, 1] + cam_trans[None, 1]) / (
                    inputs[:, 2] + cam_trans[None, 2] + 1e-4) * \
            cfg.focal[1] + cfg.princpt[1]
        x = x / cfg.input_body_shape[1] * cfg.input_img_shape[1]  # input_img_shape
        y = y / cfg.input_body_shape[0] * cfg.input_img_shape[0]  # input_img_shape
        out = np.stack((x, y, np.ones_like(x)), 1)  # [num_points, 3], input_img_shape
        out = np.dot(bb2img_trans, out.transpose(1, 0)).transpose(1, 0)  # [num_points, 2], original image space
        return out

    def validate_within_img(self, img, points):  # check whether the points is within the image
        # img: (h, w, c), points: (num_points, 2)
        h, w, c = img.shape
        valid_mask = np.logical_and(np.logical_and(0 < points[:, 0], points[:, 0] < w),
                                    np.logical_and(0 < points[:, 1], points[:, 1] < h))

        return valid_mask

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'pa_mpvpe_all': [], 'pa_mpvpe_hand': [], 'pa_mpvpe_face': [], 'mpvpe_all': [], 'mpvpe_hand': [],
                       'mpvpe_face': [], 'pa_mpjpe_body': [], 'pa_mpjpe_hand': [],
                       # --- Natural-hand 2D keypoint metrics (vs the REAL COCO-WholeBody
                       # hand annotations, NOT the SMPL-X pseudo-GT mesh). This is the
                       # in-the-wild hand-quality signal the pseudo-GT MPJPE above cannot
                       # capture (OSX was fit to that pseudo-GT, so it is biased). Errors
                       # are normalized per hand by the GT hand-keypoint bbox diagonal, so
                       # PCK/NME are scale-invariant. 'abs' = absolute placement included;
                       # 'wa' = wrist-aligned (single-point translation) to isolate finger
                       # articulation from global hand placement.
                       'hand2d_pck01_abs': [], 'hand2d_pck02_abs': [], 'hand2d_nme_abs': [],
                       'hand2d_pck01_wa': [], 'hand2d_pck02_wa': [], 'hand2d_nme_wa': [],
                       'hand2d_pck02_abs_l': [], 'hand2d_pck02_abs_r': [],
                       'hand2d_nme_wa_l': [], 'hand2d_nme_wa_r': [], 'hand2d_n': [],
                       # --- Per-hand, per-joint arrays for grouped attribution
                       # (see docs/wa2d_group_attribution.md). Each entry is appended
                       # inside the `if v[0]:` block (wrist valid), so these stay
                       # 1:1 aligned with hand2d_nme_wa / hand2d_nme_wa_{l,r}.
                       # hand2d_*_joints / hand2d_joint_valid are (21,) per hand;
                       # the rest are scalars. hand2d_img_path is a str per hand.
                       'hand2d_wa_joints': [], 'hand2d_abs_joints': [],
                       'hand2d_joint_valid': [], 'hand2d_side': [],
                       'hand2d_hand_size': [], 'hand2d_n_visible': [],
                       'hand2d_img_path': []}
        for name in ('thumb', 'index', 'middle', 'ring', 'pinky'):
            eval_result['hand2d_wa_nme_' + name] = []
            eval_result['hand2d_wa_n_' + name] = []
        for name in ('j1', 'j2', 'j3', 'tip'):
            eval_result['hand2d_wa_nme_' + name] = []
            eval_result['hand2d_wa_n_' + name] = []
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]

            # perspective transformation of the joints
            mesh_gt = out['smplx_mesh_cam_pseudo_gt']
            cam_trans = out['cam_trans']
            joint_gt_body_wo_trans = np.dot(smpl_x.j14_regressor, mesh_gt - cam_trans)
            joint_gt_body_proj = self.perspective_transform(out['cam_trans'], out['bb2img_trans'],
                                                            joint_gt_body_wo_trans)  # origin image space
            joint_gt_lhand_wo_trans = np.dot(smpl_x.orig_hand_regressor['left'], mesh_gt - cam_trans)
            joint_gt_lhand_proj = self.perspective_transform(out['cam_trans'], out['bb2img_trans'],
                                                             joint_gt_lhand_wo_trans)  # origin image space
            joint_gt_rhand_wo_trans = np.dot(smpl_x.orig_hand_regressor['right'], mesh_gt - cam_trans)
            joint_gt_rhand_proj = self.perspective_transform(out['cam_trans'], out['bb2img_trans'],
                                                             joint_gt_rhand_wo_trans)  # origin image space
            mesh_gt_proj = self.perspective_transform(out['cam_trans'], out['bb2img_trans'], mesh_gt-cam_trans)

            img_path = annot['img_path']
            img = load_img(img_path)[:, :, ::-1]

            # We only calculate the error of the joints/vertices within the image plane
            joint_gt_body_valid = self.validate_within_img(img, joint_gt_body_proj)
            joint_gt_lhand_valid = self.validate_within_img(img, joint_gt_lhand_proj)
            joint_gt_rhand_valid = self.validate_within_img(img, joint_gt_rhand_proj)
            mesh_valid = self.validate_within_img(img, mesh_gt_proj)
            mesh_lhand_valid = mesh_valid[smpl_x.hand_vertex_idx['left_hand']]
            mesh_rhand_valid = mesh_valid[smpl_x.hand_vertex_idx['right_hand']]
            mesh_face_valid = mesh_valid[smpl_x.face_vertex_idx]

            # MPVPE from all vertices
            mesh_out = out['smplx_mesh_cam']
            mesh_out_align = rigid_align(mesh_out, mesh_gt)
            eval_result['pa_mpvpe_all'].append(np.sqrt(np.sum((mesh_out_align - mesh_gt) ** 2, 1))[mesh_valid].mean() * 1000)
            mesh_out_align = mesh_out - np.dot(smpl_x.J_regressor, mesh_out)[smpl_x.J_regressor_idx['pelvis'], None, :] + \
                             np.dot(smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['pelvis'], None, :]
            eval_result['mpvpe_all'].append(np.sqrt(np.sum((mesh_out_align - mesh_gt) ** 2, 1))[mesh_valid].mean() * 1000)

            # MPVPE from hand vertices
            mesh_gt_lhand = mesh_gt[smpl_x.hand_vertex_idx['left_hand'], :]
            mesh_out_lhand = mesh_out[smpl_x.hand_vertex_idx['left_hand'], :]
            mesh_out_lhand_align = rigid_align(mesh_out_lhand, mesh_gt_lhand)
            mesh_gt_rhand = mesh_gt[smpl_x.hand_vertex_idx['right_hand'], :]
            mesh_out_rhand = mesh_out[smpl_x.hand_vertex_idx['right_hand'], :]
            mesh_out_rhand_align = rigid_align(mesh_out_rhand, mesh_gt_rhand)
            pa_mpvpe_hand = []
            if sum(mesh_lhand_valid) != 0:
                pa_mpvpe_lhand = np.sqrt(np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1))[
                                     mesh_lhand_valid].mean() * 1000
                pa_mpvpe_hand.append(pa_mpvpe_lhand)
            if sum(mesh_rhand_valid) != 0:
                pa_mpvpe_rhand = np.sqrt(np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1))[
                                     mesh_rhand_valid].mean() * 1000
                pa_mpvpe_hand.append(pa_mpvpe_rhand)
            if len(pa_mpvpe_hand) > 0:
                eval_result['pa_mpvpe_hand'].append(np.mean(pa_mpvpe_hand))

            mesh_out_lhand_align = mesh_out_lhand - np.dot(smpl_x.J_regressor, mesh_out)[
                                                    smpl_x.J_regressor_idx['lwrist'], None, :] + np.dot(
                smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['lwrist'], None, :]
            mesh_out_rhand_align = mesh_out_rhand - np.dot(smpl_x.J_regressor, mesh_out)[
                                                    smpl_x.J_regressor_idx['rwrist'], None, :] + np.dot(
                smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['rwrist'], None, :]

            mpvpe_hand = []
            if sum(mesh_lhand_valid) != 0:
                mpvpe_lhand = np.sqrt(np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1))[mesh_lhand_valid].mean() * 1000
                mpvpe_hand.append(mpvpe_lhand)
            if sum(mesh_rhand_valid) != 0:
                mpvpe_rhand = np.sqrt(np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1))[mesh_rhand_valid].mean() * 1000
                mpvpe_hand.append(mpvpe_rhand)
            if len(mpvpe_hand) > 0:
                eval_result['mpvpe_hand'].append(np.mean(mpvpe_hand))

            # MPVPE from face vertices
            mesh_gt_face = mesh_gt[smpl_x.face_vertex_idx, :]
            mesh_out_face = mesh_out[smpl_x.face_vertex_idx, :]
            mesh_out_face_align = rigid_align(mesh_out_face, mesh_gt_face)
            if sum(mesh_face_valid) != 0:
                eval_result['pa_mpvpe_face'].append(
                    np.sqrt(np.sum((mesh_out_face_align - mesh_gt_face) ** 2, 1))[mesh_face_valid].mean() * 1000)
            mesh_out_face_align = mesh_out_face - np.dot(smpl_x.J_regressor, mesh_out)[smpl_x.J_regressor_idx['neck'],
                                                  None, :] + np.dot(smpl_x.J_regressor, mesh_gt)[
                                                             smpl_x.J_regressor_idx['neck'], None, :]
            if sum(mesh_face_valid) != 0:
                eval_result['mpvpe_face'].append(
                    np.sqrt(np.sum((mesh_out_face_align - mesh_gt_face) ** 2, 1))[mesh_face_valid].mean() * 1000)

            # MPJPE from body joints
            joint_gt_body = np.dot(smpl_x.j14_regressor, mesh_gt)
            joint_out_body = np.dot(smpl_x.j14_regressor, mesh_out)
            joint_out_body_align = rigid_align(joint_out_body, joint_gt_body)
            eval_result['pa_mpjpe_body'].append(
                np.sqrt(np.sum((joint_out_body_align - joint_gt_body) ** 2, 1))[joint_gt_body_valid].mean() * 1000)

            # MPJPE from hand joints
            joint_gt_lhand = np.dot(smpl_x.orig_hand_regressor['left'], mesh_gt)
            joint_out_lhand = np.dot(smpl_x.orig_hand_regressor['left'], mesh_out)
            joint_out_lhand_align = rigid_align(joint_out_lhand, joint_gt_lhand)
            joint_gt_rhand = np.dot(smpl_x.orig_hand_regressor['right'], mesh_gt)
            joint_out_rhand = np.dot(smpl_x.orig_hand_regressor['right'], mesh_out)
            joint_out_rhand_align = rigid_align(joint_out_rhand, joint_gt_rhand)

            pa_mpjpe_hand = []
            if sum(joint_gt_lhand_valid)!=0:
                pa_mpjpe_lhand = np.sqrt(np.sum((joint_out_lhand_align - joint_gt_lhand) ** 2, 1))[joint_gt_lhand_valid].mean() * 1000
                pa_mpjpe_hand.append(pa_mpjpe_lhand)
            if sum(joint_gt_rhand_valid)!=0:
                pa_mpjpe_rhand = np.sqrt(np.sum((joint_out_rhand_align - joint_gt_rhand) ** 2, 1))[joint_gt_rhand_valid].mean() * 1000
                pa_mpjpe_hand.append(pa_mpjpe_rhand)
            if len(pa_mpjpe_hand)>0:
                eval_result['pa_mpjpe_hand'].append(np.mean(pa_mpjpe_hand))

            # --- Natural-hand 2D keypoint accuracy vs REAL annotations ---
            # The pa_mpjpe_hand above compares the predicted mesh to the SMPL-X
            # pseudo-GT mesh; since OSX was fit to that pseudo-GT it cannot tell
            # "matches the fit label" from "correct on the actual image". Here we
            # instead project the predicted hand joints (regressed from the
            # predicted mesh) into the original image and compare to the real
            # COCO-WholeBody hand keypoints -> the in-the-wild hand-quality signal.
            if not hasattr(self, '_hand2d_idx'):
                jn = self.joint_set['joints_name']
                ls = jn.index('L_Wrist_Hand'); rs = jn.index('R_Wrist_Hand')
                expected = ('Wrist_Hand',
                            'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4',
                            'Index_1', 'Index_2', 'Index_3', 'Index_4',
                            'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4',
                            'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4',
                            'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4')
                self._hand2d_idx = {'l': list(range(ls, ls + 21)),
                                     'r': list(range(rs, rs + 21))}
                for side, idxs in self._hand2d_idx.items():
                    gt_order = tuple(jn[i][2:] for i in idxs)
                    assert gt_order == expected, 'Unexpected UBody %s hand order: %s' % (side, gt_order)
                self._hand2d_finger_idx = {
                    'thumb': np.array([1, 2, 3, 4], dtype=np.int64),
                    'index': np.array([5, 6, 7, 8], dtype=np.int64),
                    'middle': np.array([9, 10, 11, 12], dtype=np.int64),
                    'ring': np.array([13, 14, 15, 16], dtype=np.int64),
                    'pinky': np.array([17, 18, 19, 20], dtype=np.int64),
                }
                self._hand2d_level_idx = {
                    'j1': np.array([1, 5, 9, 13, 17], dtype=np.int64),
                    'j2': np.array([2, 6, 10, 14, 18], dtype=np.int64),
                    'j3': np.array([3, 7, 11, 15, 19], dtype=np.int64),
                    'tip': np.array([4, 8, 12, 16, 20], dtype=np.int64),
                }
            gt_kpt = annot['joint_img'][:, :2]          # (134,2), original image px
            gt_val = annot['joint_valid'][:, 0] > 0     # (134,)
            pred_hand_cam = {
                'l': np.dot(smpl_x.orig_hand_regressor['left'], mesh_out),
                'r': np.dot(smpl_x.orig_hand_regressor['right'], mesh_out),
            }
            for side in ('l', 'r'):
                idxs = self._hand2d_idx[side]
                gt = gt_kpt[idxs]                        # (21,2)
                v = gt_val[idxs]                         # (21,)
                if int(v.sum()) < 4:
                    continue                             # too few visible GT joints
                span = float(np.linalg.norm(gt[v].max(0) - gt[v].min(0)))
                if span < 1e-3:
                    continue                             # degenerate hand
                pred = self.perspective_transform(
                    cam_trans, out['bb2img_trans'], pred_hand_cam[side] - cam_trans)[:, :2]
                err_abs = np.linalg.norm(pred - gt, axis=1) / span
                e = err_abs[v]   # normalized err
                eval_result['hand2d_pck01_abs'].append(float((e < 0.1).mean()))
                eval_result['hand2d_pck02_abs'].append(float((e < 0.2).mean()))
                eval_result['hand2d_nme_abs'].append(float(e.mean()))
                eval_result['hand2d_pck02_abs_' + side].append(float((e < 0.2).mean()))
                eval_result['hand2d_n'].append(1)
                # wrist-aligned: shift pred so its wrist (joint 0) matches GT wrist,
                # isolating finger articulation from where the hand is placed.
                if v[0]:
                    pred_wa = pred + (gt[0] - pred[0])
                    err_wa = np.linalg.norm(pred_wa - gt, axis=1) / span
                    ew = err_wa[v]
                    eval_result['hand2d_pck01_wa'].append(float((ew < 0.1).mean()))
                    eval_result['hand2d_pck02_wa'].append(float((ew < 0.2).mean()))
                    eval_result['hand2d_nme_wa'].append(float(ew.mean()))
                    eval_result['hand2d_nme_wa_' + side].append(float(ew.mean()))
                    # Per-hand, per-joint dump for grouped attribution. err_wa is
                    # (21,) wrist-aligned normalized error over ALL 21 joints (incl.
                    # wrist itself at idx 0); err_abs is the abs (placement) error.
                    # span = GT hand-kpt bbox diagonal (hand-size proxy); v is the
                    # (21,) GT visibility mask; side 0=left 1=right. Appended only
                    # when wrist valid -> stays aligned with hand2d_nme_wa above.
                    eval_result['hand2d_wa_joints'].append(err_wa.astype(np.float32))
                    eval_result['hand2d_abs_joints'].append(err_abs.astype(np.float32))
                    eval_result['hand2d_joint_valid'].append(v.astype(np.float32))
                    eval_result['hand2d_side'].append(0.0 if side == 'l' else 1.0)
                    eval_result['hand2d_hand_size'].append(span)
                    eval_result['hand2d_n_visible'].append(float(int(v.sum())))
                    eval_result['hand2d_img_path'].append(annot['img_path'])
                    for name, finger_idxs in self._hand2d_finger_idx.items():
                        vf = v[finger_idxs]
                        if int(vf.sum()) >= 2:
                            eval_result['hand2d_wa_nme_' + name].append(float(err_wa[finger_idxs][vf].mean()))
                            eval_result['hand2d_wa_n_' + name].append(1)
                    for name, level_idxs in self._hand2d_level_idx.items():
                        vf = v[level_idxs]
                        if int(vf.sum()) >= 2:
                            eval_result['hand2d_wa_nme_' + name].append(float(err_wa[level_idxs][vf].mean()))
                            eval_result['hand2d_wa_n_' + name].append(1)

            # data_dict = {}
            # data_dict['mpvpe_all'] = eval_result['mpvpe_all'][-1]
            # data_dict['mpvpe_hand'] = eval_result['mpvpe_hand'][-1]
            # data_dict['mpvpe_face'] = eval_result['mpvpe_face'][-1]
            # data_dict['mesh'] = mesh_out
            # data_dict['mesh_gt'] = mesh_gt

            vis = cfg.vis
            save_folder = cfg.vis_dir
            data_folder = os.path.join(cfg.root_dir, 'dataset', 'UBody', 'images')
            if vis:
                from common.utils.vis import vis_keypoints, vis_mesh, save_obj, render_mesh
                img_path = annot['img_path']
                render_img_save_path = img_path.replace(data_folder, f'{save_folder}/render/')
                if os.path.exists(render_img_save_path):
                    img = load_img(render_img_save_path)[:, :, ::-1]
                else:
                    img = load_img(img_path)[:, :, ::-1]


                ''' for debug
                kpt_path = render_img_save_path.replace('/render/', '/keypoints/')
                kpt_img = img.copy()
                kpt_img = vis_keypoints(kpt_img, joint_proj)
                # kpt_img = vis_keypoints(kpt_img, mesh_gt_proj)
                os.makedirs(os.path.dirname(kpt_path), exist_ok=True)
                cv2.imwrite(kpt_path, kpt_img)    
                '''

                bbox = annot['bbox']
                focal = list(cfg.focal)
                princpt = list(cfg.princpt)
                focal[0] = focal[0] / cfg.input_body_shape[1] * bbox[2]
                focal[1] = focal[1] / cfg.input_body_shape[0] * bbox[3]
                princpt[0] = princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0]
                princpt[1] = princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]
                img = render_mesh(img, out['smplx_mesh_cam'], smpl_x.face, {'focal': focal, 'princpt': princpt})
                os.makedirs(os.path.dirname(render_img_save_path), exist_ok=True)
                cv2.imwrite(render_img_save_path, img)

        return eval_result

    def print_eval_result(self, eval_result):
        print('PA MPVPE (All): %.2f mm' % np.mean(eval_result['pa_mpvpe_all']))
        print('PA MPVPE (Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_hand']))
        print('PA MPVPE (Face): %.2f mm' % np.mean(eval_result['pa_mpvpe_face']))
        print()

        print('MPVPE (All): %.2f mm' % np.mean(eval_result['mpvpe_all']))
        print('MPVPE (Hands): %.2f mm' % np.mean(eval_result['mpvpe_hand']))
        print('MPVPE (Face): %.2f mm' % np.mean(eval_result['mpvpe_face']))
        print()

        print('PA MPJPE (Body): %.2f mm' % np.mean(eval_result['pa_mpjpe_body']))
        print('PA MPJPE (Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_hand']))

        # --- Natural-hand 2D keypoint metrics (vs REAL annotations) ---
        def _m(k):
            v = eval_result.get(k, [])
            return float(np.mean(v)) if len(v) else float('nan')
        def _n(k):
            return len(eval_result.get(k, []))
        n_hand = len(eval_result.get('hand2d_n', []))
        finger_names = ('thumb', 'index', 'middle', 'ring', 'pinky')
        level_names = ('j1', 'j2', 'j3', 'tip')
        print()
        print('--- Natural-hand 2D (vs real hand kpts; PCK/NME normalized by GT hand-kpt bbox diag) ---')
        print('Evaluated hands: %d' % n_hand)
        print('[abs]           PCK@0.1: %.3f  PCK@0.2: %.3f  NME: %.3f' % (
            _m('hand2d_pck01_abs'), _m('hand2d_pck02_abs'), _m('hand2d_nme_abs')))
        print('[wrist-aligned] PCK@0.1: %.3f  PCK@0.2: %.3f  NME: %.3f' % (
            _m('hand2d_pck01_wa'), _m('hand2d_pck02_wa'), _m('hand2d_nme_wa')))
        print('[abs] PCK@0.2  L: %.3f  R: %.3f    [wa] NME  L: %.3f  R: %.3f' % (
            _m('hand2d_pck02_abs_l'), _m('hand2d_pck02_abs_r'),
            _m('hand2d_nme_wa_l'), _m('hand2d_nme_wa_r')))
        print('[wa-finger] NME ' + '  '.join(
            '%s: %.3f' % (name, _m('hand2d_wa_nme_' + name)) for name in finger_names))
        print('[wa-finger] N   ' + '  '.join(
            '%s: %d' % (name, _n('hand2d_wa_n_' + name)) for name in finger_names))
        print('[wa-level]  NME ' + '  '.join(
            '%s: %.3f' % (name, _m('hand2d_wa_nme_' + name)) for name in level_names))
        print('[wa-level]  N   ' + '  '.join(
            '%s: %d' % (name, _n('hand2d_wa_n_' + name)) for name in level_names))

        f = open(os.path.join(cfg.result_dir, 'result.txt'), 'w')
        f.write(f'UBody dataset: \n')
        f.write('PA MPVPE (All): %.2f mm\n' % np.mean(eval_result['pa_mpvpe_all']))
        f.write('PA MPVPE (Hands): %.2f mm\n' % np.mean(eval_result['pa_mpvpe_hand']))
        f.write('PA MPVPE (Face): %.2f mm\n' % np.mean(eval_result['pa_mpvpe_face']))
        f.write('MPVPE (All): %.2f mm\n' % np.mean(eval_result['mpvpe_all']))
        f.write('MPVPE (Handsls): %.2f mm\n' % np.mean(eval_result['mpvpe_hand']))
        f.write('MPVPE (Face): %.2f mm\n' % np.mean(eval_result['mpvpe_face']))
        f.write('PA MPJPE (Body): %.2f mm\n' % np.mean(eval_result['pa_mpjpe_body']))
        f.write('PA MPJPE (Hands): %.2f mm\n' % np.mean(eval_result['pa_mpjpe_hand']))
        f.write('--- Natural-hand 2D (vs real hand kpts; normalized by GT hand-kpt bbox diag) ---\n')
        f.write('Evaluated hands: %d\n' % n_hand)
        f.write('[abs]           PCK@0.1: %.3f  PCK@0.2: %.3f  NME: %.3f\n' % (
            _m('hand2d_pck01_abs'), _m('hand2d_pck02_abs'), _m('hand2d_nme_abs')))
        f.write('[wrist-aligned] PCK@0.1: %.3f  PCK@0.2: %.3f  NME: %.3f\n' % (
            _m('hand2d_pck01_wa'), _m('hand2d_pck02_wa'), _m('hand2d_nme_wa')))
        f.write('[abs] PCK@0.2  L: %.3f  R: %.3f    [wa] NME  L: %.3f  R: %.3f\n' % (
            _m('hand2d_pck02_abs_l'), _m('hand2d_pck02_abs_r'),
            _m('hand2d_nme_wa_l'), _m('hand2d_nme_wa_r')))
        f.write('[wa-finger] NME ' + '  '.join(
            '%s: %.3f' % (name, _m('hand2d_wa_nme_' + name)) for name in finger_names) + '\n')
        f.write('[wa-finger] N   ' + '  '.join(
            '%s: %d' % (name, _n('hand2d_wa_n_' + name)) for name in finger_names) + '\n')
        f.write('[wa-level]  NME ' + '  '.join(
            '%s: %.3f' % (name, _m('hand2d_wa_nme_' + name)) for name in level_names) + '\n')
        f.write('[wa-level]  N   ' + '  '.join(
            '%s: %d' % (name, _n('hand2d_wa_n_' + name)) for name in level_names) + '\n')
        f.close()
