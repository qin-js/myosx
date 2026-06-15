import copy
import os
import os.path as osp
import pickle
from glob import glob

import cv2
import numpy as np
import torch

from config import cfg
from common.utils.human_models import smpl_x
from common.utils.preprocessing import (
    augmentation,
    load_img,
    process_bbox,
    process_human_model_output,
)


class BEDLAM(torch.utils.data.Dataset):
    """BEDLAM SMPL-X loader for processed official .npz annotations.

    Expected annotation files are produced by BEDLAM's data_processing scripts
    and contain fields such as imgname, pose_cam, shape, trans_cam, gtkps,
    cam_int, center, and scale. Raw motion_seq.npz files are not enough because
    they do not contain image paths, camera intrinsics, or projected keypoints.
    """

    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.data_path = getattr(cfg, "bedlam_dir", osp.join(cfg.data_dir, "BEDLAM"))
        self.img_path = getattr(cfg, "bedlam_img_dir", osp.join(self.data_path, "images"))
        self.annot_path = getattr(
            cfg,
            "bedlam_annot_path",
            osp.join(self.data_path, "processed_labels"),
        )
        self.sample_interval = getattr(cfg, "bedlam_sample_interval", 1)
        self.max_samples = getattr(cfg, "bedlam_max_samples", None)
        self.datalist = self.load_data()

    def _annotation_files(self):
        if osp.isfile(self.annot_path):
            return [self.annot_path]
        if osp.isdir(self.annot_path):
            pkl_files = sorted(
                path for path in glob(osp.join(self.annot_path, "*.pkl"))
                if not path.endswith(".checkpoint.pkl")
            )
            if pkl_files:
                return pkl_files
            return sorted(glob(osp.join(self.annot_path, "*.npz")))
        raise FileNotFoundError(
            "BEDLAM annotation path was not found: %s. Set cfg.bedlam_annot_path "
            "to BEDLAM processed_labels or to one processed .npz file." % self.annot_path
        )

    @staticmethod
    def _rel_parts(path):
        return [part for part in str(path).replace("\\", "/").split("/") if part and part != "."]

    def _resolve_img_path(self, img_path, scene_name=None):
        img_path = str(img_path)
        if osp.isabs(img_path):
            return img_path

        rel_parts = self._rel_parts(img_path)
        direct_path = osp.join(self.img_path, *rel_parts)
        candidates = []

        if scene_name is not None:
            if len(rel_parts) >= 2 and rel_parts[0] == scene_name:
                candidates.append(direct_path)
            elif rel_parts and rel_parts[0] == "png":
                candidates.append(osp.join(self.img_path, scene_name, *rel_parts))
            else:
                candidates.append(osp.join(self.img_path, scene_name, "png", *rel_parts))

        candidates.append(direct_path)
        for candidate in candidates:
            if osp.isfile(candidate):
                return candidate
        return candidates[0]

    @staticmethod
    def _normalize_gender(gender):
        if isinstance(gender, bytes):
            gender = gender.decode("utf-8")
        gender = str(gender).lower()
        if gender in ("m", "male"):
            return "male"
        if gender in ("f", "female"):
            return "female"
        return "neutral"

    @staticmethod
    def _to_bbox_from_center_scale(center, scale):
        center = np.asarray(center, dtype=np.float32).reshape(2)
        side = float(scale) * 200.0
        return np.array(
            [center[0] - side / 2.0, center[1] - side / 2.0, side, side],
            dtype=np.float32,
        )

    @staticmethod
    def _bbox_from_keypoints(joint_img, img_width, img_height):
        if joint_img is None:
            return None
        joint_img = np.asarray(joint_img, dtype=np.float32)
        if joint_img.ndim != 2 or joint_img.shape[1] < 2:
            return None
        if joint_img.shape[1] >= 3:
            valid = joint_img[:, 2] > 0
        else:
            valid = np.ones((joint_img.shape[0],), dtype=bool)
        valid = valid & (joint_img[:, 0] >= 0) & (joint_img[:, 0] < img_width)
        valid = valid & (joint_img[:, 1] >= 0) & (joint_img[:, 1] < img_height)
        if valid.sum() < 1:
            return None
        pts = joint_img[valid, :2]
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        return np.array([xmin, ymin, xmax - xmin, ymax - ymin], dtype=np.float32)

    @staticmethod
    def _add_cam_translation(cam_param, cam_ext):
        if cam_ext is None:
            return
        cam_ext = np.asarray(cam_ext, dtype=np.float32).reshape(4, 4)
        cam_param["R"] = np.eye(3, dtype=np.float32)
        cam_param["t"] = cam_ext[:3, 3].astype(np.float32)

    @staticmethod
    def _patch_record_camera(record, companion_npz=None, idx=None):
        cam_param = record.setdefault("cam_param", {})
        cam_ext = record.get("cam_ext")
        if cam_ext is None and companion_npz is not None and "cam_ext" in companion_npz.files and idx is not None:
            if idx < len(companion_npz["cam_ext"]):
                cam_ext = companion_npz["cam_ext"][idx]
                record["cam_ext"] = np.asarray(cam_ext, dtype=np.float32).reshape(4, 4)

        if cam_ext is not None:
            BEDLAM._add_cam_translation(cam_param, cam_ext)
        elif "t" in cam_param and "R" not in cam_param:
            cam_param["R"] = np.eye(3, dtype=np.float32)

    def _load_npz_records(self, npz_path):
        ann = np.load(npz_path, allow_pickle=True)
        scene_name = osp.splitext(osp.basename(npz_path))[0]
        required = {"imgname", "pose_cam", "shape", "trans_cam", "cam_int"}
        missing = sorted(required - set(ann.files))
        if missing:
            raise ValueError(
                "%s is not a processed BEDLAM annotation file. Missing keys: %s. "
                "Run BEDLAM/data_processing/df_full_body.py or the compatible "
                "BEDLAM2 script after images and camera GT are downloaded."
                % (npz_path, ", ".join(missing))
            )

        img_names = ann["imgname"]
        poses = ann["pose_cam"]
        shapes = ann["shape"]
        trans = ann["trans_cam"]
        cam_ints = ann["cam_int"]
        genders = ann["gender"] if "gender" in ann.files else None
        centers = ann["center"] if "center" in ann.files else None
        scales = ann["scale"] if "scale" in ann.files else None
        gtkps = ann["gtkps"] if "gtkps" in ann.files else None
        exprs = ann["expr"] if "expr" in ann.files else None
        cam_exts = ann["cam_ext"] if "cam_ext" in ann.files else None

        records = []
        for idx in range(len(img_names)):
            if self.sample_interval > 1 and idx % self.sample_interval != 0:
                continue

            img_rel = str(img_names[idx])
            img_path = self._resolve_img_path(img_rel, scene_name)
            img_shape = None
            if osp.isfile(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                if img is None:
                    continue
                img_shape = img.shape[:2]
            elif getattr(cfg, "bedlam_skip_missing_images", True):
                continue

            joint_img = None
            if gtkps is not None:
                joint_img = np.asarray(gtkps[idx], dtype=np.float32)

            if img_shape is not None:
                bbox_src = None
                if centers is not None and scales is not None:
                    bbox_src = self._to_bbox_from_center_scale(centers[idx], scales[idx])
                if bbox_src is None:
                    bbox_src = self._bbox_from_keypoints(joint_img, img_shape[1], img_shape[0])
                bbox = process_bbox(bbox_src, img_shape[1], img_shape[0]) if bbox_src is not None else None
                if bbox is None:
                    continue
            else:
                bbox = None

            pose = np.asarray(poses[idx], dtype=np.float32).reshape(-1)
            shape = np.asarray(shapes[idx], dtype=np.float32).reshape(-1)[: smpl_x.shape_param_dim]
            if shape.shape[0] < smpl_x.shape_param_dim:
                shape = np.pad(shape, (0, smpl_x.shape_param_dim - shape.shape[0]))

            smplx_param = {
                "root_pose": pose[:3],
                "body_pose": pose[3:66],
                "jaw_pose": pose[66:69] if pose.shape[0] >= 69 else np.zeros(3, dtype=np.float32),
                "lhand_pose": pose[75:120] if pose.shape[0] >= 120 else np.zeros(45, dtype=np.float32),
                "rhand_pose": pose[120:165] if pose.shape[0] >= 165 else np.zeros(45, dtype=np.float32),
                "shape": shape,
                "expr": np.asarray(exprs[idx], dtype=np.float32).reshape(-1)[: smpl_x.expr_code_dim]
                if exprs is not None else np.zeros((smpl_x.expr_code_dim), dtype=np.float32),
                "trans": np.asarray(trans[idx], dtype=np.float32).reshape(3),
                "lhand_valid": True,
                "rhand_valid": True,
                "face_valid": exprs is not None and pose.shape[0] >= 69,
                "gender": self._normalize_gender(genders[idx]) if genders is not None else "neutral",
            }

            cam_int = np.asarray(cam_ints[idx], dtype=np.float32).reshape(3, 3)
            cam_param = {
                "focal": [cam_int[0, 0], cam_int[1, 1]],
                "princpt": [cam_int[0, 2], cam_int[1, 2]],
            }
            cam_ext = np.asarray(cam_exts[idx], dtype=np.float32).reshape(4, 4) if cam_exts is not None else None
            self._add_cam_translation(cam_param, cam_ext)

            record = {
                "ann_id": "%s:%d" % (osp.basename(npz_path), idx),
                "img_path": img_path,
                "img_shape": img_shape,
                "bbox": bbox,
                "joint_img": joint_img,
                "smplx_param": smplx_param,
                "cam_param": cam_param,
            }
            if cam_ext is not None:
                record["cam_ext"] = cam_ext
            records.append(record)
            if self.max_samples is not None and len(records) >= self.max_samples:
                break
        return records

    def _load_pkl_records(self, pkl_path):
        with open(pkl_path, "rb") as f:
            records = pickle.load(f)

        companion_npz = None
        companion_npz_path = osp.splitext(pkl_path)[0] + ".npz"
        if osp.isfile(companion_npz_path):
            companion_npz = np.load(companion_npz_path, allow_pickle=True)

        out = []
        try:
            for idx, record in enumerate(records):
                if self.sample_interval > 1 and idx % self.sample_interval != 0:
                    continue
                record = copy.deepcopy(record)
                scene_name = None
                if "ann_id" in record:
                    scene_name = str(record["ann_id"]).split(":", 1)[0]
                    scene_name = osp.splitext(osp.basename(scene_name))[0]
                record["img_path"] = self._resolve_img_path(record["img_path"], scene_name)
                self._patch_record_camera(record, companion_npz, idx)
                if getattr(cfg, "bedlam_skip_missing_images", True) and not osp.isfile(record["img_path"]):
                    continue
                out.append(record)
                if self.max_samples is not None and len(out) >= self.max_samples:
                    break
        finally:
            if companion_npz is not None:
                companion_npz.close()
        return out

    def load_data(self):
        datalist = []
        for annot_path in self._annotation_files():
            ext = osp.splitext(annot_path)[1].lower()
            if ext == ".pkl":
                datalist.extend(self._load_pkl_records(annot_path))
            else:
                datalist.extend(self._load_npz_records(annot_path))
            if self.max_samples is not None and len(datalist) >= self.max_samples:
                datalist = datalist[: self.max_samples]
                break

        if len(datalist) == 0:
            raise RuntimeError(
                "No BEDLAM samples were loaded. If images are still downloading, "
                "set cfg.bedlam_skip_missing_images=False only for annotation "
                "schema checks; real training requires the image files."
            )
        print("Loaded BEDLAM %s samples: %d" % (self.data_split, len(datalist)))
        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox = data["img_path"], data["img_shape"], data["bbox"]

        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32)) / 255.0

        smplx_joint_img, smplx_joint_cam, smplx_joint_trunc, smplx_pose, smplx_shape, smplx_expr, \
            smplx_pose_valid, smplx_joint_valid, smplx_expr_valid, smplx_mesh_cam_orig = \
            process_human_model_output(
                data["smplx_param"],
                data["cam_param"],
                do_flip,
                img_shape,
                img2bb_trans,
                rot,
                "smplx",
            )

        smplx_pose_valid = np.tile(smplx_pose_valid[:, None], (1, 3)).reshape(-1)
        smplx_joint_valid = smplx_joint_valid[:, None]
        smplx_joint_trunc = smplx_joint_valid * smplx_joint_trunc

        dummy_center = np.zeros((2), dtype=np.float32)
        dummy_size = np.zeros((2), dtype=np.float32)
        joint_img = smplx_joint_img.copy()
        joint_cam = smplx_joint_cam.copy()

        inputs = {"img": img}
        targets = {
            "joint_img": joint_img,
            "joint_cam": joint_cam,
            "smplx_joint_img": smplx_joint_img,
            "smplx_joint_cam": smplx_joint_cam,
            "smplx_pose": smplx_pose,
            "smplx_shape": smplx_shape,
            "smplx_expr": smplx_expr,
            "smplx_mesh_cam": smplx_mesh_cam_orig,
            "lhand_bbox_center": dummy_center,
            "lhand_bbox_size": dummy_size,
            "rhand_bbox_center": dummy_center,
            "rhand_bbox_size": dummy_size,
            "face_bbox_center": dummy_center,
            "face_bbox_size": dummy_size,
        }
        meta_info = {
            "bb2img_trans": bb2img_trans,
            "joint_valid": smplx_joint_valid,
            "joint_trunc": smplx_joint_trunc,
            "smplx_joint_valid": smplx_joint_valid,
            "smplx_joint_trunc": smplx_joint_trunc,
            "smplx_pose_valid": smplx_pose_valid,
            "smplx_shape_valid": float(True),
            "smplx_expr_valid": float(smplx_expr_valid),
            "is_3D": float(True),
            "lhand_bbox_valid": float(False),
            "rhand_bbox_valid": float(False),
            "face_bbox_valid": float(False),
        }
        return inputs, targets, meta_info
