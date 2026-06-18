import copy
import json
import os
import os.path as osp

import numpy as np
import torch
from pycocotools.coco import COCO

from config import cfg
from common.utils.human_models import smpl_x
from common.utils.preprocessing import (
    augmentation,
    load_img,
    process_bbox,
    process_db_coord,
)
from common.utils.transforms import cam2pixel, transform_joint_to_other_db, world2cam


class InterHand26M(torch.utils.data.Dataset):
    """InterHand2.6M loader using official COCO-style annotations.

    The dataset can build its annotation index before images are downloaded by
    setting cfg.interhand_skip_missing_images=False. Calling __getitem__ still
    requires the corresponding image files.
    """

    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.img_path = getattr(
            cfg,
            "interhand_img_dir",
            osp.join(cfg.data_dir, "InterHand26M", "images"),
        )
        self.annot_path = getattr(
            cfg,
            "interhand_annot_path",
            osp.join(cfg.data_dir, "InterHand26M", "annotations"),
        )
        self.sample_interval = getattr(cfg, "interhand_sample_interval", 1)
        self.max_samples = getattr(cfg, "interhand_max_samples", None)
        self.dedup_by_group = getattr(cfg, "interhand_dedup_by_group", False)
        self.frames_per_group = getattr(cfg, "interhand_frames_per_group", 1)
        self.skip_missing_images = getattr(cfg, "interhand_skip_missing_images", False)
        self.use_human_annot = getattr(cfg, "interhand_use_human_annot", True)
        self.require_mano = getattr(cfg, "interhand_require_mano", True)
        # On an unreadable image, __getitem__ resamples other datalist entries
        # (bounded by _fallback_max_tries) instead of retaining the full raw
        # COCO / joint / MANO annotations in memory. _warned_bad_imgs rate-limits
        # the per-path warning so a few corrupt files don't flood the log.
        self._fallback_max_tries = 200
        self._warned_bad_imgs = set()

        self.joint_set = {
            "joint_num": 42,
            "joints_name": (
                "R_Thumb_4", "R_Thumb_3", "R_Thumb_2", "R_Thumb_1",
                "R_Index_4", "R_Index_3", "R_Index_2", "R_Index_1",
                "R_Middle_4", "R_Middle_3", "R_Middle_2", "R_Middle_1",
                "R_Ring_4", "R_Ring_3", "R_Ring_2", "R_Ring_1",
                "R_Pinky_4", "R_Pinky_3", "R_Pinky_2", "R_Pinky_1",
                "R_Wrist",
                "L_Thumb_4", "L_Thumb_3", "L_Thumb_2", "L_Thumb_1",
                "L_Index_4", "L_Index_3", "L_Index_2", "L_Index_1",
                "L_Middle_4", "L_Middle_3", "L_Middle_2", "L_Middle_1",
                "L_Ring_4", "L_Ring_3", "L_Ring_2", "L_Ring_1",
                "L_Pinky_4", "L_Pinky_3", "L_Pinky_2", "L_Pinky_1",
                "L_Wrist",
            ),
            "flip_pairs": tuple((i, i + 21) for i in range(21)),
        }
        self.joint_set["part_idx"] = {
            "right": np.arange(0, 21),
            "left": np.arange(21, 42),
        }
        self.joint_set["root_idx"] = {
            "right": self.joint_set["joints_name"].index("R_Wrist"),
            "left": self.joint_set["joints_name"].index("L_Wrist"),
        }

        self.datalist = self.load_data()

    @staticmethod
    def _as_key(value):
        return str(value)

    @staticmethod
    def _cam_param(cameras, capture_id, cam_id):
        capture = cameras[str(capture_id)]
        cam_id = str(cam_id)
        R = np.array(capture["camrot"][cam_id], dtype=np.float32).reshape(3, 3)
        campos = np.array(capture["campos"][cam_id], dtype=np.float32).reshape(3)
        t = -np.dot(R, campos.reshape(3, 1)).reshape(3)
        focal = np.array(capture["focal"][cam_id], dtype=np.float32).reshape(2)
        princpt = np.array(capture["princpt"][cam_id], dtype=np.float32).reshape(2)
        return {"R": R, "t": t, "focal": focal, "princpt": princpt}

    @staticmethod
    def _bbox_from_points(points, valid, img_width, img_height, extend_ratio=1.2):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        valid = np.asarray(valid, dtype=np.float32).reshape(-1) > 0
        valid = valid & (points[:, 0] >= 0) & (points[:, 0] < img_width)
        valid = valid & (points[:, 1] >= 0) & (points[:, 1] < img_height)
        if valid.sum() < 2:
            return None

        pts = points[valid]
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        w = xmax - xmin
        h = ymax - ymin
        if w <= 0 or h <= 0:
            return None

        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        w *= extend_ratio
        h *= extend_ratio
        return np.array([cx - w * 0.5, cy - h * 0.5, w, h], dtype=np.float32)

    @staticmethod
    def _xywh_to_xyxy(bbox):
        bbox = np.asarray(bbox, dtype=np.float32).reshape(4)
        bbox[2:] += bbox[:2]
        return bbox

    @staticmethod
    def _full_image_bbox(img_width, img_height):
        return np.array([0, 0, img_width, img_height], dtype=np.float32)

    def _annotation_dir(self):
        return osp.join(self.annot_path, self.data_split)

    def _aid_list(self, db):
        aid_path = osp.join(self._annotation_dir(), "aid_human_annot_%s.txt" % self.data_split)
        if self.use_human_annot and osp.isfile(aid_path):
            with open(aid_path, "r") as f:
                aids = [int(line.strip()) for line in f if line.strip()]
            return [aid for aid in aids if aid in db.anns]
        return sorted(db.anns.keys())

    def _mano_for_frame(self, mano_params, capture_id, frame_idx):
        capture = mano_params.get(str(capture_id), {})
        frame = capture.get(str(frame_idx), {})
        if self.require_mano and frame == {}:
            return None
        return frame

    @staticmethod
    def _mano_pose(mano_param):
        if mano_param is None:
            return None, None
        pose = np.asarray(mano_param["pose"], dtype=np.float32).reshape(-1)
        shape = np.asarray(mano_param["shape"], dtype=np.float32).reshape(-1)
        if pose.shape[0] < 48:
            return None, None
        return pose, shape

    def _dedup_aids_by_group(self, db, aids):
        """Temporal de-duplication of InterHand frames.

        Frames are grouped by (Capture, sequence, camera) = dirname(file_name).
        Within each group the 5fps frames are near-identical, so we keep only
        `self.frames_per_group` representatives spread evenly across the group's
        time axis (sorted by frame_idx). Every subject / gesture / camera view
        is preserved; only intra-view temporal redundancy is removed.
        """
        groups = {}
        for aid in aids:
            ann = db.anns[aid]
            img = db.loadImgs(ann["image_id"])[0]
            group_key = osp.dirname(img["file_name"])
            frame_idx = int(img.get("frame_idx", 0))
            groups.setdefault(group_key, []).append((frame_idx, aid))

        keep = []
        k = max(1, int(self.frames_per_group))
        for group_key in sorted(groups.keys()):
            members = sorted(groups[group_key])  # by (frame_idx, aid)
            n = len(members)
            if n <= k:
                picks = members
            else:
                # Evenly spaced indices across the sorted group (inclusive ends).
                idxs = [int(round(i * (n - 1) / (k - 1))) for i in range(k)] if k > 1 else [n // 2]
                idxs = sorted(set(idxs))
                picks = [members[i] for i in idxs]
            keep.extend(aid for _, aid in picks)

        print(
            "InterHand26M temporal dedup: %d groups, %d -> %d aids (frames_per_group=%d)"
            % (len(groups), len(aids), len(keep), k)
        )
        return keep

    def _make_data_dict(self, db, cameras, joints, mano_params, aid):
        ann = db.anns[aid]
        img = db.loadImgs(ann["image_id"])[0]
        img_width, img_height = img["width"], img["height"]
        img_path = osp.join(self.img_path, self.data_split, img["file_name"])
        if self.skip_missing_images and not osp.isfile(img_path):
            return None

        capture_id = self._as_key(img["capture"])
        frame_idx = self._as_key(img["frame_idx"])
        cam_id = self._as_key(img["camera"])

        if capture_id not in joints or frame_idx not in joints[capture_id]:
            return None
        joint_frame = joints[capture_id][frame_idx]
        joint_valid = np.array(joint_frame["joint_valid"], dtype=np.float32).reshape(-1, 1)
        joint_world = np.array(joint_frame["world_coord"], dtype=np.float32).reshape(-1, 3)

        cam_param = self._cam_param(cameras, capture_id, cam_id)
        joint_cam = world2cam(joint_world, cam_param["R"], cam_param["t"])
        joint_img = cam2pixel(joint_cam, cam_param["focal"], cam_param["princpt"])

        bbox_src = ann.get("bbox", None)
        if bbox_src is not None:
            bbox = process_bbox(np.array(bbox_src, dtype=np.float32), img_width, img_height)
        else:
            bbox_src = self._bbox_from_points(joint_img[:, :2], joint_valid, img_width, img_height)
            bbox = process_bbox(bbox_src, img_width, img_height) if bbox_src is not None else None
        if bbox is None:
            bbox = process_bbox(self._full_image_bbox(img_width, img_height), img_width, img_height)
        if bbox is None:
            return None

        hand_bboxes = {}
        for side in ("left", "right"):
            part_idx = self.joint_set["part_idx"][side]
            hand_bbox = self._bbox_from_points(
                joint_img[part_idx, :2], joint_valid[part_idx], img_width, img_height
            )
            hand_bboxes[side] = self._xywh_to_xyxy(hand_bbox) if hand_bbox is not None else None

        mano_frame = self._mano_for_frame(mano_params, capture_id, frame_idx)
        if mano_frame is None:
            return None

        return {
            "ann_id": aid,
            "img_path": img_path,
            "img_shape": (img_height, img_width),
            "bbox": bbox,
            "joint_img": joint_img.astype(np.float32),
            "joint_cam": joint_cam.astype(np.float32),
            "joint_valid": joint_valid.astype(np.float32),
            "cam_param": cam_param,
            "mano_param": copy.deepcopy(mano_frame),
            "lhand_bbox": hand_bboxes["left"],
            "rhand_bbox": hand_bboxes["right"],
            "hand_type": ann.get("hand_type", joint_frame.get("hand_type", "")),
            "hand_type_valid": ann.get("hand_type_valid", joint_frame.get("hand_type_valid", 1)),
        }

    def load_data(self):
        annot_dir = self._annotation_dir()
        data_json = osp.join(annot_dir, "InterHand2.6M_%s_data.json" % self.data_split)
        camera_json = osp.join(annot_dir, "InterHand2.6M_%s_camera.json" % self.data_split)
        joint_json = osp.join(annot_dir, "InterHand2.6M_%s_joint_3d.json" % self.data_split)
        mano_json = osp.join(annot_dir, "InterHand2.6M_%s_MANO_NeuralAnnot.json" % self.data_split)

        db = COCO(data_json)
        with open(camera_json, "r") as f:
            cameras = json.load(f)
        with open(joint_json, "r") as f:
            joints = json.load(f)
        with open(mano_json, "r") as f:
            mano_params = json.load(f)

        aid_list = self._aid_list(db)

        if self.dedup_by_group:
            aid_list = self._dedup_aids_by_group(db, aid_list)

        if self.sample_interval > 1:
            aid_list = aid_list[:: self.sample_interval]

        # Cap by even sub-sampling across the (already deduped) list so that the
        # cap preserves subject/gesture/view diversity instead of truncating
        # whole later Captures sequentially.
        if self.max_samples is not None and len(aid_list) > self.max_samples:
            n = len(aid_list)
            m = self.max_samples
            idxs = [int(i * n / m) for i in range(m)]
            aid_list = [aid_list[i] for i in idxs]

        datalist = []
        for aid in aid_list:
            data_dict = self._make_data_dict(db, cameras, joints, mano_params, aid)
            if data_dict is None:
                continue
            datalist.append(data_dict)
            if self.max_samples is not None and len(datalist) >= self.max_samples:
                break

        if len(datalist) == 0:
            raise RuntimeError(
                "No InterHand26M samples were loaded. Check cfg.interhand_annot_path, "
                "cfg.interhand_img_dir, and cfg.interhand_skip_missing_images."
            )
        print("Loaded InterHand26M %s samples: %d" % (self.data_split, len(datalist)))
        return datalist

    def process_hand_bbox(self, bbox, do_flip, img_shape, img2bb_trans):
        if bbox is None:
            bbox = np.array([0, 0, 1, 1], dtype=np.float32).reshape(2, 2)
            bbox_valid = float(False)
        else:
            bbox = bbox.reshape(2, 2)
            if do_flip:
                bbox[:, 0] = img_shape[1] - bbox[:, 0] - 1
                bbox[0, 0], bbox[1, 0] = bbox[1, 0].copy(), bbox[0, 0].copy()

            xmin, ymin, xmax, ymax = bbox.reshape(4).tolist()
            corners = np.array(
                [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
                dtype=np.float32,
            )
            corners_xy1 = np.concatenate((corners, np.ones_like(corners[:, :1])), 1)
            bbox = np.dot(img2bb_trans, corners_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            bbox[:, 0] = bbox[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            bbox[:, 1] = bbox[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

            xmin, ymin = bbox.min(axis=0)
            xmax, ymax = bbox.max(axis=0)
            bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32).reshape(2, 2)
            bbox_valid = float(True)
        return bbox, bbox_valid

    def _per_hand_relative_interhand(self, joint_cam, joint_valid):
        joint_cam = joint_cam.copy()
        coord_valid = joint_valid.copy()

        for side in ("right", "left"):
            part_idx = self.joint_set["part_idx"][side]
            root_idx = self.joint_set["root_idx"][side]
            if joint_valid[root_idx, 0] <= 0:
                joint_cam[part_idx] = 0
                coord_valid[part_idx] = 0
                continue

            root = joint_cam[root_idx, None, :]
            joint_cam[part_idx] = (joint_cam[part_idx] - root) / 1000.0
            coord_valid[root_idx, 0] = 0

        return joint_cam, coord_valid

    def _process_coord_valid(self, coord_valid, do_flip):
        coord_valid = coord_valid.copy()
        if do_flip:
            for pair in self.joint_set["flip_pairs"]:
                coord_valid[pair[0], :], coord_valid[pair[1], :] = (
                    coord_valid[pair[1], :].copy(),
                    coord_valid[pair[0], :].copy(),
                )
        return transform_joint_to_other_db(
            coord_valid,
            self.joint_set["joints_name"],
            smpl_x.joints_name,
        )

    def _make_smplx_pose_target(self, mano_param, do_flip):
        pose = np.zeros((smpl_x.orig_joint_num, 3), dtype=np.float32)
        pose_valid = np.zeros((smpl_x.orig_joint_num), dtype=np.float32)

        side_to_part = {"left": "lhand", "right": "rhand"}
        for side in ("left", "right"):
            mano_pose, _ = self._mano_pose(mano_param.get(side))
            if mano_pose is None:
                continue

            part_name = side_to_part[side]
            part_idx = list(smpl_x.orig_joint_part[part_name])
            pose[part_idx] = mano_pose[3:48].reshape(15, 3)
            pose_valid[part_idx] = 1

        if do_flip:
            for pair in smpl_x.orig_flip_pairs:
                pose[pair[0], :], pose[pair[1], :] = pose[pair[1], :].copy(), pose[pair[0], :].copy()
                pose_valid[pair[0]], pose_valid[pair[1]] = pose_valid[pair[1]].copy(), pose_valid[pair[0]].copy()
            pose[:, 1:3] *= -1

        return pose.reshape(-1), np.tile(pose_valid[:, None], (1, 3)).reshape(-1)

    def _load_img_with_fallback(self, data, idx):
        """Return (img, data) for the requested sample. If its image is unreadable
        (corrupt / missing), replace the WHOLE sample with another readable entry
        from the already-built datalist — no raw COCO / joint / MANO annotations
        are retained for this. Bounded by _fallback_max_tries so a wholesale
        outage fails loudly instead of looping forever.
        """
        img_path = data["img_path"]
        try:
            return load_img(img_path), data
        except (IOError, OSError) as exc:
            first_error = exc
            if img_path not in self._warned_bad_imgs:
                self._warned_bad_imgs.add(img_path)
                print("[InterHand26M] unreadable image, resampling another sample: %s" % img_path)

        n = len(self.datalist)
        for step in range(1, min(n, self._fallback_max_tries) + 1):
            candidate_data = self.datalist[(idx + step) % n]
            candidate_path = candidate_data["img_path"]
            if candidate_path == img_path:
                continue
            expected_h, expected_w = candidate_data["img_shape"]
            try:
                img = load_img(candidate_path)
            except (IOError, OSError):
                continue
            if img.shape[0] != expected_h or img.shape[1] != expected_w:
                continue
            return img, copy.deepcopy(candidate_data)

        raise IOError(
            "Fail to read %s and %d fallback samples were also unreadable"
            % (img_path, min(n, self._fallback_max_tries))
        ) from first_error

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])

        img, data = self._load_img_with_fallback(data, idx)
        img_path, img_shape, bbox = data["img_path"], data["img_shape"], data["bbox"]
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32)) / 255.0

        lhand_bbox, lhand_bbox_valid = self.process_hand_bbox(
            data["lhand_bbox"], do_flip, img_shape, img2bb_trans
        )
        rhand_bbox, rhand_bbox_valid = self.process_hand_bbox(
            data["rhand_bbox"], do_flip, img_shape, img2bb_trans
        )
        if do_flip:
            lhand_bbox, rhand_bbox = rhand_bbox, lhand_bbox
            lhand_bbox_valid, rhand_bbox_valid = rhand_bbox_valid, lhand_bbox_valid

        lhand_bbox_center = (lhand_bbox[0] + lhand_bbox[1]) * 0.5
        rhand_bbox_center = (rhand_bbox[0] + rhand_bbox[1]) * 0.5
        lhand_bbox_size = lhand_bbox[1] - lhand_bbox[0]
        rhand_bbox_size = rhand_bbox[1] - rhand_bbox[0]
        dummy_center = np.zeros((2), dtype=np.float32)
        dummy_size = np.zeros((2), dtype=np.float32)

        joint_cam, coord_valid = self._per_hand_relative_interhand(
            data["joint_cam"],
            data["joint_valid"],
        )
        joint_img = np.concatenate((data["joint_img"][:, :2], joint_cam[:, 2:]), 1)
        joint_img[:, 2] = (joint_img[:, 2] / (cfg.hand_3d_size / 2.0) + 1.0) / 2.0
        joint_img[:, 2] *= cfg.output_hm_shape[0]

        joint_img, joint_cam, _, joint_trunc = process_db_coord(
            joint_img,
            joint_cam,
            data["joint_valid"],
            do_flip,
            img_shape,
            self.joint_set["flip_pairs"],
            img2bb_trans,
            rot,
            self.joint_set["joints_name"],
            smpl_x.joints_name,
        )
        joint_valid = self._process_coord_valid(coord_valid, do_flip)

        smplx_joint_img = joint_img.copy()
        smplx_joint_cam = joint_cam.copy()
        smplx_joint_valid = joint_valid.copy()
        smplx_joint_trunc = joint_trunc.copy()

        smplx_pose, smplx_pose_valid = self._make_smplx_pose_target(data["mano_param"], do_flip)
        smplx_shape = np.zeros((smpl_x.shape_param_dim), dtype=np.float32)
        smplx_expr = np.zeros((smpl_x.expr_code_dim), dtype=np.float32)
        smplx_mesh_cam = np.zeros((smpl_x.vertex_num, 3), dtype=np.float32)

        inputs = {"img": img}
        targets = {
            "joint_img": joint_img,
            "joint_cam": joint_cam,
            "smplx_joint_img": smplx_joint_img,
            "smplx_joint_cam": smplx_joint_cam,
            "smplx_pose": smplx_pose,
            "smplx_shape": smplx_shape,
            "smplx_expr": smplx_expr,
            "smplx_mesh_cam": smplx_mesh_cam,
            "lhand_bbox_center": lhand_bbox_center,
            "lhand_bbox_size": lhand_bbox_size,
            "rhand_bbox_center": rhand_bbox_center,
            "rhand_bbox_size": rhand_bbox_size,
            "face_bbox_center": dummy_center,
            "face_bbox_size": dummy_size,
        }
        meta_info = {
            "bb2img_trans": bb2img_trans,
            "joint_valid": joint_valid,
            "joint_trunc": joint_trunc,
            "smplx_joint_valid": smplx_joint_valid,
            "smplx_joint_trunc": smplx_joint_trunc,
            "smplx_pose_valid": smplx_pose_valid,
            "smplx_shape_valid": float(False),
            "smplx_expr_valid": float(False),
            "is_3D": float(True),
            "lhand_bbox_valid": lhand_bbox_valid,
            "rhand_bbox_valid": rhand_bbox_valid,
            "face_bbox_valid": float(False),
            "dataset_id": float(1),
            "is_interhand": float(True),
            "is_bedlam": float(False),
            "is_hand_only": float(True),
        }
        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        return {}

    def print_eval_result(self, eval_result):
        print("InterHand26M evaluation is not implemented for this SMPL-X loader.")
