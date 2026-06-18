"""Stage-0 batch / mask checker for InterHand26M + BEDLAM mixed finetuning.

This implements the pre-training checks described in
``docs/interhand_bedlam_training_plan.md`` section 13.7. It runs purely on the
data side by default (no GPU, no model weights) and verifies that:

  * the weighted mixed loader produces the configured BEDLAM:InterHand ratio;
  * InterHand ``L_Wrist`` / ``R_Wrist`` *pose* valid flags are 0;
  * InterHand ``L_Wrist`` / ``R_Wrist`` *3D cam* valid flags are 0;
  * InterHand left-hand fingers are relative to ``L_Wrist`` and right-hand
    fingers are relative to ``R_Wrist`` (per-hand, NOT a shared root);
  * InterHand shape / expression / face-bbox valid flags are 0;
  * BEDLAM shape valid is 1;
  * the per-key tensor shapes match between the two loaders (so mixed-batch
    default collate will not fail).

Optionally (``--forward``) it builds the real ``pytorch`` model, loads the
frozen backbone + a trained snapshot, runs one InterHand batch in test mode and
saves the input crops with the *model-predicted* hand boxes drawn, which is the
Stage-0 gate for "are the predicted InterHand hand boxes plausible?".

Run from anywhere; paths are resolved relative to the repo root::

    python tool/InterHand26M/check_interhand_bedlam_batch.py \
        --interhand_max_samples 256 --bedlam_max_samples 256 \
        --num_batches 8 --batch_size 8

Add ``--forward --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
        --continue_train_path <snapshot.pth.tar>`` to also dump predicted boxes.
"""

import argparse
import os
import os.path as osp
import random
import sys

import numpy as np

# --------------------------------------------------------------------------
# Path setup: make ``config`` and the per-dataset modules importable exactly
# the way ``main/train.py`` does (config.py itself adds the data/<name> dirs).
# --------------------------------------------------------------------------
THIS_DIR = osp.dirname(osp.abspath(__file__))
ROOT_DIR = osp.abspath(osp.join(THIS_DIR, "..", ".."))
sys.path.insert(0, osp.join(ROOT_DIR, "main"))
sys.path.insert(0, ROOT_DIR)

import torch  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402

from config import cfg  # noqa: E402  (also registers data/<name> on sys.path)
from common.utils.human_models import smpl_x  # noqa: E402
from dataset import MultipleDatasets  # noqa: E402
from InterHand26M import InterHand26M  # noqa: E402
from BEDLAM import BEDLAM  # noqa: E402


GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


class Report(object):
    """Tiny pass/fail collector with pretty printing."""

    def __init__(self):
        self.items = []

    def check(self, name, ok, detail=""):
        self.items.append((name, bool(ok), detail))
        tag = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        line = f"  [{tag}] {name}"
        if detail:
            line += f"  ->  {detail}"
        print(line)
        return ok

    def info(self, name, detail):
        print(f"  [{YELLOW}INFO{RESET}] {name}  ->  {detail}")

    @property
    def failed(self):
        return [n for n, ok, _ in self.items if not ok]


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------
# Joint-index helpers (resolved once from the smpl_x singleton).
# --------------------------------------------------------------------------
def _indices():
    return {
        "lwrist_137": smpl_x.lwrist_idx,
        "rwrist_137": smpl_x.rwrist_idx,
        "lhand_137": list(smpl_x.joint_part["lhand"]),
        "rhand_137": list(smpl_x.joint_part["rhand"]),
        "lwrist_orig": smpl_x.orig_joints_name.index("L_Wrist"),
        "rwrist_orig": smpl_x.orig_joints_name.index("R_Wrist"),
        "lhand_orig": list(smpl_x.orig_joint_part["lhand"]),
        "rhand_orig": list(smpl_x.orig_joint_part["rhand"]),
    }


def _as_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# --------------------------------------------------------------------------
# Per-sample InterHand checks.
# --------------------------------------------------------------------------
def check_interhand_sample(targets, meta_info, idx, rep):
    ix = _indices()
    hand_3d_size = float(cfg.hand_3d_size)

    shape_valid = float(_as_np(meta_info["smplx_shape_valid"]))
    expr_valid = float(_as_np(meta_info["smplx_expr_valid"]))
    face_valid = float(_as_np(meta_info["face_bbox_valid"]))
    rep.check(f"[IH#{idx}] smplx_shape_valid == 0", shape_valid == 0, f"={shape_valid}")
    rep.check(f"[IH#{idx}] smplx_expr_valid  == 0", expr_valid == 0, f"={expr_valid}")
    rep.check(f"[IH#{idx}] face_bbox_valid   == 0", face_valid == 0, f"={face_valid}")

    rep.check(f"[IH#{idx}] is_interhand flag", float(_as_np(meta_info["is_interhand"])) == 1.0)
    rep.check(f"[IH#{idx}] is_bedlam flag", float(_as_np(meta_info["is_bedlam"])) == 0.0)
    rep.check(f"[IH#{idx}] is_hand_only flag", float(_as_np(meta_info["is_hand_only"])) == 1.0)

    # --- pose valid: wrists must be 0 ---
    pose_valid = _as_np(meta_info["smplx_pose_valid"]).reshape(smpl_x.orig_joint_num, 3)
    lwrist_pv = pose_valid[ix["lwrist_orig"]].sum()
    rwrist_pv = pose_valid[ix["rwrist_orig"]].sum()
    rep.check(f"[IH#{idx}] L_Wrist pose_valid == 0", lwrist_pv == 0, f"sum={lwrist_pv}")
    rep.check(f"[IH#{idx}] R_Wrist pose_valid == 0", rwrist_pv == 0, f"sum={rwrist_pv}")

    # only the 15+15 finger joints may be pose-valid
    finger_orig = ix["lhand_orig"] + ix["rhand_orig"]
    nonfinger_mask = np.ones(smpl_x.orig_joint_num, dtype=bool)
    nonfinger_mask[finger_orig] = False
    nonfinger_pv = pose_valid[nonfinger_mask].sum()
    rep.check(
        f"[IH#{idx}] only finger joints pose-valid",
        nonfinger_pv == 0,
        f"non-finger pose_valid sum={nonfinger_pv}",
    )

    # --- 3D cam valid: wrists must be 0 ---
    joint_valid = _as_np(meta_info["joint_valid"]).reshape(smpl_x.joint_num, -1)
    smplx_joint_valid = _as_np(meta_info["smplx_joint_valid"]).reshape(smpl_x.joint_num, -1)
    rep.check(
        f"[IH#{idx}] L_Wrist joint_valid(3D) == 0",
        joint_valid[ix["lwrist_137"]].sum() == 0,
        f"={joint_valid[ix['lwrist_137']].sum()}",
    )
    rep.check(
        f"[IH#{idx}] R_Wrist joint_valid(3D) == 0",
        joint_valid[ix["rwrist_137"]].sum() == 0,
        f"={joint_valid[ix['rwrist_137']].sum()}",
    )
    rep.check(
        f"[IH#{idx}] L/R_Wrist smplx_joint_valid(3D) == 0",
        smplx_joint_valid[ix["lwrist_137"]].sum() == 0
        and smplx_joint_valid[ix["rwrist_137"]].sum() == 0,
    )

    # --- per-hand wrist-relative coordinates ---
    joint_cam = _as_np(targets["joint_cam"]).reshape(smpl_x.joint_num, 3)
    lwrist_mag = np.abs(joint_cam[ix["lwrist_137"]]).max()
    rwrist_mag = np.abs(joint_cam[ix["rwrist_137"]]).max()
    # Both wrists must sit at the origin of their own hand frame. Under a single
    # shared root (the old bug) only one wrist could be ~0; the other would be
    # offset by the inter-wrist vector.
    rep.check(
        f"[IH#{idx}] L_Wrist joint_cam ~= 0 (own-frame origin)",
        lwrist_mag < 1e-4,
        f"max|coord|={lwrist_mag:.2e} m",
    )
    rep.check(
        f"[IH#{idx}] R_Wrist joint_cam ~= 0 (own-frame origin)",
        rwrist_mag < 1e-4,
        f"max|coord|={rwrist_mag:.2e} m",
    )

    # finger magnitudes must be within the hand depth window for valid hands
    for side, hand_part, wrist_orig in (
        ("L", ix["lhand_137"], ix["lwrist_orig"]),
        ("R", ix["rhand_137"], ix["rwrist_orig"]),
    ):
        valid_rows = joint_valid[hand_part].reshape(len(hand_part), -1).sum(axis=1) > 0
        if valid_rows.sum() == 0:
            rep.info(f"[IH#{idx}] {side}-hand", "no valid finger joints in this sample")
            continue
        mag = np.abs(joint_cam[hand_part][valid_rows]).max()
        rep.check(
            f"[IH#{idx}] {side}-hand fingers within hand_3d_size",
            mag <= hand_3d_size,
            f"max|coord|={mag:.3f} m (limit {hand_3d_size} m)",
        )


def check_bedlam_sample(targets, meta_info, idx, rep):
    shape_valid = float(_as_np(meta_info["smplx_shape_valid"]))
    rep.check(f"[BL#{idx}] smplx_shape_valid == 1", shape_valid == 1, f"={shape_valid}")
    rep.check(f"[BL#{idx}] is_bedlam flag", float(_as_np(meta_info["is_bedlam"])) == 1.0)
    rep.check(f"[BL#{idx}] is_interhand flag", float(_as_np(meta_info["is_interhand"])) == 0.0)
    rep.check(f"[BL#{idx}] is_hand_only flag", float(_as_np(meta_info["is_hand_only"])) == 0.0)


# --------------------------------------------------------------------------
# Shape report (one InterHand + one BEDLAM sample side by side).
# --------------------------------------------------------------------------
def report_shapes(ih_sample, bl_sample, rep):
    keys_targets = ["smplx_pose", "smplx_joint_cam", "joint_cam", "smplx_shape", "smplx_expr",
                    "lhand_bbox_center", "lhand_bbox_size", "rhand_bbox_center", "rhand_bbox_size"]
    keys_meta = ["smplx_pose_valid", "joint_valid", "smplx_joint_valid", "joint_trunc",
                 "smplx_joint_trunc"]
    print("\n--- tensor shapes (InterHand | BEDLAM) ---")
    (_, ih_t, ih_m) = ih_sample
    (_, bl_t, bl_m) = bl_sample
    ok_all = True
    for k in keys_targets:
        a, b = np.asarray(ih_t[k]).shape, np.asarray(bl_t[k]).shape
        same = a == b
        ok_all &= same
        col = GREEN if same else RED
        print(f"  targets[{k:>18}]: {col}{a}{RESET} | {col}{b}{RESET}")
    for k in keys_meta:
        a, b = np.asarray(ih_m[k]).shape, np.asarray(bl_m[k]).shape
        same = a == b
        ok_all &= same
        col = GREEN if same else RED
        print(f"  meta  [{k:>18}]: {col}{a}{RESET} | {col}{b}{RESET}")
    rep.check("InterHand/BEDLAM tensor shapes match (collate-safe)", ok_all)


# --------------------------------------------------------------------------
# Mixed weighted loader ratio.
# --------------------------------------------------------------------------
def check_mixed_ratio(ih_db, bl_db, args, rep):
    names = list(cfg.trainset_3d)
    dbs_by_name = {"InterHand26M": ih_db, "BEDLAM": bl_db}
    dbs = [dbs_by_name[n] for n in names]

    prob_cfg = getattr(cfg, "trainset_3d_sample_prob", None)
    if not getattr(cfg, "use_weighted_dataset_sampling", False) or not prob_cfg:
        rep.info("weighted sampling", "disabled in cfg; skipping ratio check")
        return
    sample_prob = np.asarray([prob_cfg[n] for n in names], dtype=np.float32)
    sample_prob = sample_prob / sample_prob.sum()
    print("\n--- mixed weighted-sampling ratio ---")
    print("  configured: " + ", ".join("%s=%.3f" % (n, p) for n, p in zip(names, sample_prob)))

    mixed = MultipleDatasets(dbs, make_same_len=False, sample_prob=sample_prob)
    loader = torch.utils.data.DataLoader(
        mixed,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_thread,
        drop_last=True,
        worker_init_fn=_seed_worker if args.num_thread > 0 else None,
    )

    counts = {"is_interhand": 0.0, "is_bedlam": 0.0}
    total = 0
    it = iter(loader)
    for b in range(args.num_batches):
        try:
            _, _, meta_info = next(it)
        except StopIteration:
            break
        counts["is_interhand"] += float(meta_info["is_interhand"].sum())
        counts["is_bedlam"] += float(meta_info["is_bedlam"].sum())
        total += int(meta_info["is_interhand"].numel())
    if total == 0:
        rep.check("mixed loader produced batches", False, "no batches drawn")
        return
    ih_ratio = counts["is_interhand"] / total
    bl_ratio = counts["is_bedlam"] / total
    rep.info("realized over %d samples" % total,
             "BEDLAM=%.3f, InterHand26M=%.3f" % (bl_ratio, ih_ratio))
    target_ih = float(sample_prob[names.index("InterHand26M")])
    # The ratio only converges with enough draws; with a tiny sample this is
    # pure finite-sample noise, so only hard-assert above a minimum count.
    MIN_SAMPLES_FOR_RATIO = 64
    if total >= MIN_SAMPLES_FOR_RATIO:
        rep.check(
            "realized InterHand ratio within 0.15 of target",
            abs(ih_ratio - target_ih) < 0.15,
            "target=%.3f realized=%.3f" % (target_ih, ih_ratio),
        )
    else:
        rep.info(
            "ratio assertion skipped",
            "only %d samples (< %d); increase --num_batches/--batch_size to assert"
            % (total, MIN_SAMPLES_FOR_RATIO),
        )
    rep.check(
        "both datasets appear in mixed batches",
        counts["is_interhand"] > 0 and counts["is_bedlam"] > 0,
    )


# --------------------------------------------------------------------------
# Optional: real forward pass, dump predicted InterHand hand boxes.
# --------------------------------------------------------------------------
def run_forward_dump(args, rep):
    try:
        import cv2
        from torch.nn.parallel.data_parallel import DataParallel  # noqa: F401
    except Exception as exc:  # pragma: no cover
        rep.check("forward: import deps", False, str(exc))
        return

    out_dir = args.out_dir or osp.join(ROOT_DIR, "output", "check_interhand_bedlam", "pred_boxes")
    os.makedirs(out_dir, exist_ok=True)

    # cfg needs valid output dirs for the logger inside Tester.
    cfg.set_args(args.gpu_ids if args.gpu_ids else "0", lr=cfg.lr if hasattr(cfg, "lr") else 1e-4)
    # Relative paths in the docs assume CWD=main/. This script runs from the
    # repo root, so resolve the pretrained path against main/ and the root too.
    pre = args.pretrained_model_path
    if not osp.isabs(pre) and not osp.isfile(pre):
        for base in (osp.join(ROOT_DIR, "main"), ROOT_DIR):
            cand = osp.normpath(osp.join(base, pre))
            if osp.isfile(cand):
                pre = cand
                break
    cfg.pretrained_model_path = pre
    if args.continue_train_path:
        cfg.continue_train_path = args.continue_train_path
    # set_additional_args (which we deliberately skip) is what normally maps
    # encoder_setting -> feat_dim / encoder config. Replicate the feat_dim part
    # so the heads are built for the right encoder width (osx_l -> 1024).
    cfg.feat_dim = 1024 if cfg.encoder_setting == "osx_l" else 768
    cfg.prepare_dirs(osp.join("output", "check_interhand_bedlam"))

    from common.base import Tester  # lazy: pulls in model_core / DCNv4 ops
    tester = Tester()
    tester._make_model()
    model = tester.model
    model.eval()

    ih_db = InterHand26M(transforms.ToTensor(), "train")
    n = min(args.num_forward_samples, len(ih_db))

    print("\n--- forward: predicted InterHand hand boxes ---")
    print("  (boxes are in input_img_shape space: W=%d, H=%d)"
          % (cfg.input_img_shape[1], cfg.input_img_shape[0]))
    for i in range(n):
        inputs, targets, meta_info = ih_db[i]
        batch = {"img": torch.from_numpy(np.asarray(inputs["img"]))[None].cuda()}
        t = {k: torch.from_numpy(np.asarray(v))[None].cuda() for k, v in targets.items()}
        m = {}
        for k, v in meta_info.items():
            arr = np.asarray(v, dtype=np.float32)
            m[k] = torch.from_numpy(arr)[None].cuda()
        with torch.no_grad():
            out = model(batch, t, m, "test")

        # dataset only does img/255.0 (ToTensor does not rescale float32 input),
        # so undo exactly that -- no ImageNet mean/std was ever applied.
        img = out["img"][0].detach().cpu().numpy().transpose(1, 2, 0)
        img = (img * 255.0).clip(0, 255).astype(np.uint8)[:, :, ::-1].copy()
        h, w = img.shape[:2]
        # overlay GT hand keypoints (output_hm space -> input_img_shape) so we
        # can confirm the GT-injected box actually tracks the hand.
        ji = np.asarray(targets["joint_img"]).reshape(smpl_x.joint_num, 3)
        tr = np.asarray(meta_info["joint_trunc"]).reshape(smpl_x.joint_num, -1)
        for part, kp_color in (("lhand", (0, 0, 255)), ("rhand", (0, 255, 0))):
            for j in smpl_x.joint_part[part]:
                if tr[j].sum() <= 0:
                    continue
                px = int(ji[j, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1])
                py = int(ji[j, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0])
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(img, (px, py), 3, kp_color, -1)
        coords = {}
        for name, color in (("lhand_bbox", (0, 0, 255)), ("rhand_bbox", (0, 255, 0))):
            b = out[name][0].detach().cpu().numpy().astype(np.float32)
            coords[name] = b
            x0 = int(np.clip(b[0], 0, w - 1)); y0 = int(np.clip(b[1], 0, h - 1))
            x1 = int(np.clip(b[2], 0, w - 1)); y1 = int(np.clip(b[3], 0, h - 1))
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 3)
        dst = osp.join(out_dir, "interhand_pred_box_%03d.jpg" % i)
        cv2.imwrite(dst, img)
        lb, rb = coords["lhand_bbox"], coords["rhand_bbox"]
        print("  #%03d saved | lhand(red) xyxy=[%.0f,%.0f,%.0f,%.0f] "
              "rhand(green) xyxy=[%.0f,%.0f,%.0f,%.0f]"
              % (i, lb[0], lb[1], lb[2], lb[3], rb[0], rb[1], rb[2], rb[3]))
    rep.info("forward dump", "wrote %d images to %s" % (n, out_dir))
    rep.info("forward gate", "inspect the boxes: they must cover the hands")


def build_datasets(args):
    if args.interhand_max_samples is not None:
        cfg.interhand_max_samples = args.interhand_max_samples
    if args.bedlam_max_samples is not None:
        cfg.bedlam_max_samples = args.bedlam_max_samples
    print("Building InterHand26M (max_samples=%s) ..." % cfg.interhand_max_samples)
    ih_db = InterHand26M(transforms.ToTensor(), "train")
    print("Building BEDLAM (max_samples=%s) ..." % cfg.bedlam_max_samples)
    bl_db = BEDLAM(transforms.ToTensor(), "train")
    return ih_db, bl_db


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num_batches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_thread", type=int, default=0,
                        help="DataLoader workers for the ratio check (0 = main process)")
    parser.add_argument("--num_samples", type=int, default=8,
                        help="per-dataset samples for the mask checks")
    parser.add_argument("--interhand_max_samples", type=int, default=256)
    parser.add_argument("--bedlam_max_samples", type=int, default=256)
    # optional forward
    parser.add_argument("--forward", action="store_true",
                        help="also build the model and dump predicted hand boxes")
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--pretrained_model_path", type=str,
                        default="../pretrained_models/osx_l.pth.tar")
    parser.add_argument("--continue_train_path", type=str, default="")
    parser.add_argument("--num_forward_samples", type=int, default=8)
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()

    rep = Report()
    print("=" * 70)
    print("InterHand26M + BEDLAM Stage-0 batch / mask check")
    print("=" * 70)

    ih_db, bl_db = build_datasets(args)

    # ---- per-sample mask checks ----
    print("\n--- InterHand per-sample mask checks ---")
    n_ih = min(args.num_samples, len(ih_db))
    ih_sample = None
    for i in range(n_ih):
        try:
            inputs, targets, meta_info = ih_db[i]
        except Exception as exc:
            rep.check(f"[IH#{i}] __getitem__", False, repr(exc))
            continue
        if ih_sample is None:
            ih_sample = (inputs, targets, meta_info)
        check_interhand_sample(targets, meta_info, i, rep)

    print("\n--- BEDLAM per-sample mask checks ---")
    n_bl = min(args.num_samples, len(bl_db))
    bl_sample = None
    for i in range(n_bl):
        try:
            inputs, targets, meta_info = bl_db[i]
        except Exception as exc:
            rep.check(f"[BL#{i}] __getitem__", False, repr(exc))
            continue
        if bl_sample is None:
            bl_sample = (inputs, targets, meta_info)
        check_bedlam_sample(targets, meta_info, i, rep)

    # ---- shapes + ratio ----
    if ih_sample is not None and bl_sample is not None:
        report_shapes(ih_sample, bl_sample, rep)
    else:
        rep.check("loaded at least one sample from each dataset", False)

    check_mixed_ratio(ih_db, bl_db, args, rep)

    # ---- optional forward ----
    if args.forward:
        try:
            run_forward_dump(args, rep)
        except Exception as exc:
            rep.check("forward dump", False, repr(exc))

    # ---- summary ----
    print("\n" + "=" * 70)
    if rep.failed:
        print(f"{RED}FAILED CHECKS ({len(rep.failed)}):{RESET}")
        for n in rep.failed:
            print("  - " + n)
        print("=" * 70)
        sys.exit(1)
    print(f"{GREEN}All checks passed.{RESET}")
    print("=" * 70)


if __name__ == "__main__":
    main()
