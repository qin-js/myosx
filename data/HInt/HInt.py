import copy
import json
import os
import os.path as osp
from glob import glob

import cv2
import numpy as np
import torch

from config import cfg
from common.utils.human_models import smpl_x
from common.utils.preprocessing import augmentation, load_img, process_bbox


class HInt(torch.utils.data.Dataset):
    """HInt hand-interaction test loader.

    HInt provides 2D keypoints and occlusion labels for individual hands. It
    has no SMPL-X mesh GT, so this loader evaluates projected SMPL-X hand
    joints against the real 2D hand annotations, using the same abs /
    wrist-aligned metrics as the UBody natural-hand diagnostic.
    """

    _FINGER_IDXS = {
        "thumb": np.array([1, 2, 3, 4], dtype=np.int64),
        "index": np.array([5, 6, 7, 8], dtype=np.int64),
        "middle": np.array([9, 10, 11, 12], dtype=np.int64),
        "ring": np.array([13, 14, 15, 16], dtype=np.int64),
        "pinky": np.array([17, 18, 19, 20], dtype=np.int64),
    }
    _LEVEL_IDXS = {
        "j1": np.array([1, 5, 9, 13, 17], dtype=np.int64),
        "j2": np.array([2, 6, 10, 14, 18], dtype=np.int64),
        "j3": np.array([3, 7, 11, 15, 19], dtype=np.int64),
        "tip": np.array([4, 8, 12, 16, 20], dtype=np.int64),
    }

    def __init__(self, transform, data_split):
        if data_split != "test":
            raise ValueError("HInt is test-only in this project")
        self.transform = transform
        self.data_split = data_split
        self.root = self._resolve_root()
        self.split = str(getattr(cfg, "hint_split", "test")).upper()
        self.sources = self._parse_sources(getattr(cfg, "hint_sources", ""))
        self.max_samples = getattr(cfg, "hint_max_samples", None)
        self.sample_interval = max(1, int(getattr(cfg, "hint_sample_interval", 1)))
        self.skip_missing_images = getattr(cfg, "hint_skip_missing_images", True)
        self.bbox_scale = float(getattr(cfg, "hint_bbox_scale", 1.25))

        self.datalist = self.load_data()
        print("Loaded HInt %s samples: %d" % (self.split, len(self.datalist)))

    def _resolve_root(self):
        roots = []
        cfg_root = getattr(cfg, "hint_root", "")
        if cfg_root:
            roots.append(cfg_root)
        roots.extend([
            osp.join(cfg.data_dir, "HInt", "HInt_annotation"),
            osp.join(cfg.data_dir, "HInt", "HInt_annotation_partial"),
            osp.join(cfg.data_dir, "HInt"),
            osp.join(cfg.data_dir, "HInt_annotation"),
            osp.join(cfg.data_dir, "HInt_annotation_partial"),
        ])

        for root in roots:
            if root and not osp.isabs(root):
                root = osp.join(cfg.root_dir, root)
            if root and osp.isdir(root):
                return root

        raise RuntimeError(
            "HInt root was not found. Set HINT_ROOT or pass --hint_root. "
            "Tried: %s" % ", ".join(roots)
        )

    @staticmethod
    def _parse_sources(sources):
        if not sources:
            return None
        if isinstance(sources, (list, tuple)):
            values = sources
        else:
            values = str(sources).split(",")
        values = [v.strip().lower() for v in values if v.strip()]
        return set(values) if values else None

    def _source_ok(self, dirname):
        if self.sources is None:
            return True
        lower = dirname.lower()
        return any(source in lower for source in self.sources)

    def _split_dirs(self):
        prefix = self.split + "_"
        dirs = []
        for name in sorted(os.listdir(self.root)):
            path = osp.join(self.root, name)
            if not osp.isdir(path):
                continue
            if not name.upper().startswith(prefix):
                continue
            if not self._source_ok(name):
                continue
            dirs.append(path)
        if not dirs:
            raise RuntimeError(
                "No HInt %s folders found under %s for sources=%s"
                % (self.split, self.root, sorted(self.sources) if self.sources else "all")
            )
        return dirs

    @staticmethod
    def _image_for_json(json_path):
        stem = osp.splitext(json_path)[0]
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            img_path = stem + ext
            if osp.isfile(img_path):
                return img_path
        return stem + ".jpg"

    @staticmethod
    def _parse_keypoints(raw):
        arr = np.asarray(raw, dtype=np.float32)
        if arr.size == 42:
            arr = arr.reshape(21, 2)
        elif arr.size == 63:
            arr = arr.reshape(21, 3)[:, :2]
        elif arr.shape == (21, 3):
            arr = arr[:, :2]
        elif arr.shape != (21, 2):
            return None
        return arr.astype(np.float32)

    @staticmethod
    def _parse_occlusion(raw):
        if raw is None:
            return np.zeros((21), dtype=np.float32)
        arr = np.asarray(raw, dtype=np.float32).reshape(-1)
        if arr.size != 21:
            return np.zeros((21), dtype=np.float32)
        return arr

    @staticmethod
    def _hand_side(ann):
        for key in ("side", "hand_type", "handedness", "hand_side", "label"):
            if key not in ann:
                continue
            value = ann[key]
            if isinstance(value, (list, tuple)) and len(value):
                value = value[0]
            value = str(value).lower()
            if "left" in value or value == "l":
                return "left"
            if "right" in value or value == "r":
                return "right"
        for key, side in (("is_left", "left"), ("is_right", "right")):
            if key in ann and bool(ann[key]):
                return side
        return "unknown"

    @staticmethod
    def _hand_side_from_path(json_path):
        stem = osp.splitext(osp.basename(json_path))[0].lower()
        if stem.endswith("_l"):
            return "left"
        if stem.endswith("_r"):
            return "right"
        return "unknown"

    @staticmethod
    def _points_valid(kpts, img_width, img_height):
        valid = np.isfinite(kpts).all(axis=1)
        valid &= kpts[:, 0] >= 0
        valid &= kpts[:, 0] < img_width
        valid &= kpts[:, 1] >= 0
        valid &= kpts[:, 1] < img_height
        return valid

    @staticmethod
    def _xyxy_from_points(kpts, valid, scale=1.2):
        if int(valid.sum()) < 2:
            return None
        pts = kpts[valid]
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        w = xmax - xmin
        h = ymax - ymin
        if w <= 1 or h <= 1:
            return None
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        w *= scale
        h *= scale
        return np.array([cx - w * 0.5, cy - h * 0.5,
                         cx + w * 0.5, cy + h * 0.5], dtype=np.float32)

    @staticmethod
    def _clip_xyxy(bbox, img_width, img_height):
        bbox = np.asarray(bbox, dtype=np.float32).reshape(4).copy()
        bbox[0] = np.clip(bbox[0], 0, img_width - 1)
        bbox[2] = np.clip(bbox[2], 0, img_width - 1)
        bbox[1] = np.clip(bbox[1], 0, img_height - 1)
        bbox[3] = np.clip(bbox[3], 0, img_height - 1)
        x1, x2 = sorted((bbox[0], bbox[2]))
        y1, y2 = sorted((bbox[1], bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return None
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def _bbox_candidates(self, raw_bbox):
        if raw_bbox is None:
            return []
        arr = np.asarray(raw_bbox, dtype=np.float32).reshape(-1)
        if arr.size < 4:
            return []
        a, b, c, d = arr[:4].tolist()
        return [
            np.array([a, b, c, d], dtype=np.float32),          # xyxy
            np.array([a, b, a + c, b + d], dtype=np.float32),  # xywh
            np.array([b, a, d, c], dtype=np.float32),          # yxyx
            np.array([b, a, b + d, a + c], dtype=np.float32),  # yxhw
        ]

    def _bbox_from_annotation(self, raw_bbox, kpts, valid, img_width, img_height):
        best = None
        best_score = (-1.0, float("inf"))
        for candidate in self._bbox_candidates(raw_bbox):
            xyxy = self._clip_xyxy(candidate, img_width, img_height)
            if xyxy is None:
                continue
            if int(valid.sum()) > 0:
                pts = kpts[valid]
                inside = (
                    (pts[:, 0] >= xyxy[0] - 2) & (pts[:, 0] <= xyxy[2] + 2) &
                    (pts[:, 1] >= xyxy[1] - 2) & (pts[:, 1] <= xyxy[3] + 2)
                )
                contain = float(inside.mean())
            else:
                contain = 0.0
            area = float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
            score = (contain, -area)
            if score > best_score:
                best_score = score
                best = xyxy

        if best is not None and best_score[0] >= 0.5:
            return best
        return self._xyxy_from_points(kpts, valid, scale=self.bbox_scale)

    @staticmethod
    def _xyxy_to_xywh(xyxy):
        x1, y1, x2, y2 = np.asarray(xyxy, dtype=np.float32).reshape(4)
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

    def load_data(self):
        datalist = []
        json_paths = []
        for folder in self._split_dirs():
            json_paths.extend(sorted(glob(osp.join(folder, "*.json"))))

        if self.sample_interval > 1:
            json_paths = json_paths[:: self.sample_interval]

        for json_path in json_paths:
            img_path = self._image_for_json(json_path)
            if self.skip_missing_images and not osp.isfile(img_path):
                continue
            try:
                img_shape = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION).shape[:2]
            except AttributeError:
                if self.skip_missing_images:
                    continue
                raise IOError("Fail to read %s" % img_path)
            img_height, img_width = img_shape

            with open(json_path, "r") as f:
                anns = json.load(f)
            if isinstance(anns, dict):
                anns = anns.get("annotations", anns.get("hands", anns.get("instances", [])))
            if not isinstance(anns, list):
                continue

            for instance_idx, ann in enumerate(anns):
                if not isinstance(ann, dict) or "keypoints" not in ann:
                    continue
                kpts = self._parse_keypoints(ann["keypoints"])
                if kpts is None:
                    continue
                valid = self._points_valid(kpts, img_width, img_height)
                if int(valid.sum()) < 4:
                    continue

                hand_bbox = self._bbox_from_annotation(
                    ann.get("bbox", None), kpts, valid, img_width, img_height
                )
                if hand_bbox is None:
                    continue

                body_bbox = process_bbox(self._xyxy_to_xywh(hand_bbox), img_width, img_height)
                if body_bbox is None:
                    continue

                hand_side = self._hand_side(ann)
                if hand_side == "unknown":
                    hand_side = self._hand_side_from_path(json_path)

                data_dict = {
                    "img_path": img_path,
                    "json_path": json_path,
                    "instance_idx": instance_idx,
                    "img_shape": (img_height, img_width),
                    "bbox": body_bbox,
                    "hand_bbox": hand_bbox,
                    "keypoints": kpts,
                    "keypoint_valid": valid.astype(np.float32),
                    "occlusion": self._parse_occlusion(ann.get("occlusion", None)),
                    "hand_side": hand_side,
                }
                datalist.append(data_dict)
                if self.max_samples is not None and len(datalist) >= self.max_samples:
                    return datalist

        if len(datalist) == 0:
            raise RuntimeError(
                "No HInt samples were loaded from %s split=%s sources=%s. "
                "If using HInt_annotation_partial, Ego4D folders have annotations "
                "but no images and will be skipped."
                % (self.root, self.split, sorted(self.sources) if self.sources else "all")
            )
        return datalist

    def process_hand_bbox(self, bbox, do_flip, img_shape, img2bb_trans):
        if bbox is None:
            bbox = np.array([0, 0, 1, 1], dtype=np.float32).reshape(2, 2)
            bbox_valid = float(False)
        else:
            bbox = np.asarray(bbox, dtype=np.float32).reshape(2, 2).copy()
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

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img = load_img(data["img_path"])
        img_shape = data["img_shape"]

        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(
            img, data["bbox"], self.data_split
        )
        img = self.transform(img.astype(np.float32)) / 255.0

        hand_bbox = data["hand_bbox"].copy()
        lhand_bbox, lhand_bbox_valid = self.process_hand_bbox(
            hand_bbox, do_flip, img_shape, img2bb_trans
        )
        rhand_bbox, rhand_bbox_valid = self.process_hand_bbox(
            hand_bbox, do_flip, img_shape, img2bb_trans
        )

        lhand_bbox_center = (lhand_bbox[0] + lhand_bbox[1]) * 0.5
        rhand_bbox_center = (rhand_bbox[0] + rhand_bbox[1]) * 0.5
        lhand_bbox_size = lhand_bbox[1] - lhand_bbox[0]
        rhand_bbox_size = rhand_bbox[1] - rhand_bbox[0]
        dummy_center = np.zeros((2), dtype=np.float32)
        dummy_size = np.zeros((2), dtype=np.float32)

        inputs = {"img": img}
        targets = {
            "lhand_bbox_center": lhand_bbox_center,
            "lhand_bbox_size": lhand_bbox_size,
            "rhand_bbox_center": rhand_bbox_center,
            "rhand_bbox_size": rhand_bbox_size,
            "face_bbox_center": dummy_center,
            "face_bbox_size": dummy_size,
        }
        meta_info = {
            "bb2img_trans": bb2img_trans,
            "lhand_bbox_valid": float(lhand_bbox_valid),
            "rhand_bbox_valid": float(rhand_bbox_valid),
            "face_bbox_valid": float(False),
            "is_hand_only": float(True),
        }
        return inputs, targets, meta_info

    def perspective_transform(self, cam_trans, bb2img_trans, points):
        x = (points[:, 0] + cam_trans[None, 0]) / (
            points[:, 2] + cam_trans[None, 2] + 1e-4
        ) * cfg.focal[0] + cfg.princpt[0]
        y = (points[:, 1] + cam_trans[None, 1]) / (
            points[:, 2] + cam_trans[None, 2] + 1e-4
        ) * cfg.focal[1] + cfg.princpt[1]
        x = x / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        y = y / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        out = np.stack((x, y, np.ones_like(x)), 1)
        return np.dot(bb2img_trans, out.transpose(1, 0)).transpose(1, 0)[:, :2]

    @staticmethod
    def _mean(values):
        return float(np.mean(values)) if len(values) else float("nan")

    def _append_subset_metrics(self, eval_result, prefix, err_abs, err_wa, valid, span):
        if int(valid.sum()) < 1:
            return
        e = err_abs[valid]
        ew = err_wa[valid]
        eval_result[prefix + "_pck01_abs"].append(float((e < 0.1).mean()))
        eval_result[prefix + "_pck02_abs"].append(float((e < 0.2).mean()))
        eval_result[prefix + "_nme_abs"].append(float(e.mean()))
        eval_result[prefix + "_pck01_wa"].append(float((ew < 0.1).mean()))
        eval_result[prefix + "_pck02_wa"].append(float((ew < 0.2).mean()))
        eval_result[prefix + "_nme_wa"].append(float(ew.mean()))
        eval_result[prefix + "_n"].append(1)

        if prefix == "all":
            for name, idxs in self._FINGER_IDXS.items():
                vf = valid[idxs]
                if int(vf.sum()) >= 2:
                    eval_result["wa_nme_" + name].append(float(err_wa[idxs][vf].mean()))
                    eval_result["wa_n_" + name].append(1)
            for name, idxs in self._LEVEL_IDXS.items():
                vf = valid[idxs]
                if int(vf.sum()) >= 2:
                    eval_result["wa_nme_" + name].append(float(err_wa[idxs][vf].mean()))
                    eval_result["wa_n_" + name].append(1)

    @staticmethod
    def _known_side_candidates(hand_side):
        if hand_side == "left":
            return ("left",)
        if hand_side == "right":
            return ("right",)
        return ("left", "right")

    def _best_prediction_for_gt(self, pred_by_side, gt, valid, hand_side):
        best = None
        best_side = None
        best_score = float("inf")
        for side in self._known_side_candidates(hand_side):
            pred = pred_by_side[side]
            err_abs = np.linalg.norm(pred - gt, axis=1)
            score = float(err_abs[valid].mean())
            if score < best_score:
                best_score = score
                best = pred
                best_side = side
        return best, best_side

    def evaluate(self, outs, cur_sample_idx):
        eval_result = {}
        for prefix in ("all", "vis", "occ"):
            for metric in (
                "pck01_abs", "pck02_abs", "nme_abs",
                "pck01_wa", "pck02_wa", "nme_wa", "n",
            ):
                eval_result[prefix + "_" + metric] = []
        for name in self._FINGER_IDXS:
            eval_result["wa_nme_" + name] = []
            eval_result["wa_n_" + name] = []
        for name in self._LEVEL_IDXS:
            eval_result["wa_nme_" + name] = []
            eval_result["wa_n_" + name] = []
        eval_result["chosen_left"] = []
        eval_result["chosen_right"] = []
        eval_result["known_side"] = []

        for n, out in enumerate(outs):
            annot = self.datalist[cur_sample_idx + n]
            mesh_out = out["smplx_mesh_cam"]
            cam_trans = out["cam_trans"]
            gt = annot["keypoints"]
            valid = annot["keypoint_valid"].astype(bool)
            if int(valid.sum()) < 4:
                continue
            span = float(np.linalg.norm(gt[valid].max(0) - gt[valid].min(0)))
            if span < 1e-3:
                continue

            pred_by_side = {}
            for side in ("left", "right"):
                pred_cam = np.dot(smpl_x.orig_hand_regressor[side], mesh_out)
                pred_by_side[side] = self.perspective_transform(
                    cam_trans, out["bb2img_trans"], pred_cam - cam_trans
                )

            pred, chosen_side = self._best_prediction_for_gt(
                pred_by_side, gt, valid, annot["hand_side"]
            )
            eval_result["chosen_" + chosen_side].append(1)
            if annot["hand_side"] != "unknown":
                eval_result["known_side"].append(1)

            err_abs = np.linalg.norm(pred - gt, axis=1) / span
            pred_wa = pred + (gt[0] - pred[0])
            err_wa = np.linalg.norm(pred_wa - gt, axis=1) / span

            occ = annot["occlusion"].reshape(-1) > 0.5
            self._append_subset_metrics(eval_result, "all", err_abs, err_wa, valid, span)
            self._append_subset_metrics(eval_result, "vis", err_abs, err_wa, valid & (~occ), span)
            self._append_subset_metrics(eval_result, "occ", err_abs, err_wa, valid & occ, span)

        return eval_result

    def print_eval_result(self, eval_result):
        def m(key):
            return self._mean(eval_result.get(key, []))

        def n(key):
            return len(eval_result.get(key, []))

        n_all = n("all_n")
        chosen_l = n("chosen_left")
        chosen_r = n("chosen_right")
        known_side = n("known_side")

        print("HInt dataset:")
        print("Evaluated hands: %d" % n_all)
        print("Known side labels used: %d / %d" % (known_side, n_all))
        print("Unknown-side assignment: left=%d right=%d" % (chosen_l, chosen_r))
        for prefix, title in (("all", "all"), ("vis", "visible"), ("occ", "occluded")):
            print("[%s abs] PCK@0.1: %.3f  PCK@0.2: %.3f  NME: %.3f" % (
                title, m(prefix + "_pck01_abs"), m(prefix + "_pck02_abs"), m(prefix + "_nme_abs")))
            print("[%s wa ] PCK@0.1: %.3f  PCK@0.2: %.3f  NME: %.3f" % (
                title, m(prefix + "_pck01_wa"), m(prefix + "_pck02_wa"), m(prefix + "_nme_wa")))

        finger_names = ("thumb", "index", "middle", "ring", "pinky")
        level_names = ("j1", "j2", "j3", "tip")
        print("[wa-finger] NME " + "  ".join(
            "%s: %.3f" % (name, m("wa_nme_" + name)) for name in finger_names))
        print("[wa-finger] N   " + "  ".join(
            "%s: %d" % (name, n("wa_n_" + name)) for name in finger_names))
        print("[wa-level]  NME " + "  ".join(
            "%s: %.3f" % (name, m("wa_nme_" + name)) for name in level_names))
        print("[wa-level]  N   " + "  ".join(
            "%s: %d" % (name, n("wa_n_" + name)) for name in level_names))

        result_path = osp.join(cfg.result_dir, "result.txt")
        with open(result_path, "w") as f:
            f.write("HInt dataset:\n")
            f.write("Evaluated hands: %d\n" % n_all)
            f.write("Known side labels used: %d / %d\n" % (known_side, n_all))
            f.write("Unknown-side assignment: left=%d right=%d\n" % (chosen_l, chosen_r))
            for prefix, title in (("all", "all"), ("vis", "visible"), ("occ", "occluded")):
                f.write("[%s abs] PCK@0.1: %.3f  PCK@0.2: %.3f  NME: %.3f\n" % (
                    title, m(prefix + "_pck01_abs"), m(prefix + "_pck02_abs"), m(prefix + "_nme_abs")))
                f.write("[%s wa ] PCK@0.1: %.3f  PCK@0.2: %.3f  NME: %.3f\n" % (
                    title, m(prefix + "_pck01_wa"), m(prefix + "_pck02_wa"), m(prefix + "_nme_wa")))
            f.write("[wa-finger] NME " + "  ".join(
                "%s: %.3f" % (name, m("wa_nme_" + name)) for name in finger_names) + "\n")
            f.write("[wa-finger] N   " + "  ".join(
                "%s: %d" % (name, n("wa_n_" + name)) for name in finger_names) + "\n")
            f.write("[wa-level]  NME " + "  ".join(
                "%s: %.3f" % (name, m("wa_nme_" + name)) for name in level_names) + "\n")
            f.write("[wa-level]  N   " + "  ".join(
                "%s: %d" % (name, n("wa_n_" + name)) for name in level_names) + "\n")
