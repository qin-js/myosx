"""Offline BEDLAM hand-ROI quality statistics.

Background (see docs/bedlam_problems.txt):
  The pytorch decoder path freezes encoder/body_position_net/box_net/hand_roi_net.
  BEDLAM does NOT inject a GT hand bbox, so its hand crop relies entirely on the
  frozen top-down box_net prediction. On BEDLAM's multi-person / occluded scenes
  the box_net may frame the wrong person, producing an ROI that does not contain
  the target person's GT hand joints. The resulting hand-space 2D losses
  (joint_proj / joint_img / smplx_joint_img) become silent noise supervision.

This script forwards every BEDLAM sample through ONLY
  encoder -> body_position_net -> box_net
(the exact frozen path the trainer uses), restores the predicted left/right hand
bboxes with restore_bbox(), and measures how well each predicted bbox covers the
sample's valid GT hand joints. It writes a per-sample quality .npz that the
trainer can later use to mask BEDLAM hand-space 2D loss, and prints an aggregate
report.

Coordinate spaces
  restore_bbox() returns xyxy in input_body_shape (H=256, W=192) space.
  target['smplx_joint_img'] x/y live in output_hm_shape (W=12, H=16) space.
  Both normalize to the same body crop, so we map joints into input_body space:
    x_body = x_hm / output_hm_shape[2] * input_body_shape[1]
    y_body = y_hm / output_hm_shape[1] * input_body_shape[0]

Per-sample, per-hand metrics
  valid_gt_joints   = hand joints with smplx_joint_trunc == 1
  inside            = valid_gt_joints whose (x,y) fall inside the predicted bbox
  coverage          = inside / valid_gt_joint_count
  bbox_valid        = bbox finite AND width >= min_wh AND height >= min_wh
  roi_valid (ok)    = bbox_valid AND valid_gt_joint_count >= min_joints
                                 AND coverage >= coverage_thr

Example
  cd /workspace/myosx/main
  python ../tool/BEDLAM/compute_hand_roi_quality.py \
      --gpu 0 --split train --max_samples 5000 --batch_size 32 \
      --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
      --output ../dataset/BEDLAM/processed_labels/hand_roi_quality_train.npz
"""

import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch

# --- repo path setup (mirror main/*.py: run from main/, common/ is one level up) ---
CUR_DIR = osp.dirname(os.path.abspath(__file__))
ROOT_DIR = osp.join(CUR_DIR, "..", "..")
MAIN_DIR = osp.join(ROOT_DIR, "main")
for p in (ROOT_DIR, MAIN_DIR):
    p = osp.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)


def parse_args():
    parser = argparse.ArgumentParser(description="BEDLAM hand-ROI quality statistics")
    parser.add_argument("--gpu", type=str, default="0", help="GPU id, e.g. '0'")
    parser.add_argument("--split", type=str, default="train",
                        help="dataset split label (used for output naming / dataset construction)")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="cap on number of BEDLAM samples to load/check (None=all)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_thread", type=int, default=8)
    parser.add_argument("--pretrained_model_path", type=str,
                        default="../pretrained_models/osx_l.pth.tar",
                        help="original OSX checkpoint that fills the frozen backbone")
    parser.add_argument("--output", type=str,
                        default="../dataset/BEDLAM/processed_labels/hand_roi_quality_train.npz")
    parser.add_argument("--encoder_setting", type=str, default="osx_l", choices=["osx_b", "osx_l"])
    # quality thresholds (see docs/bedlam_problems.txt)
    parser.add_argument("--coverage_thr", type=float, default=0.6)
    parser.add_argument("--min_joints", type=int, default=8)
    parser.add_argument("--min_wh", type=float, default=8.0,
                        help="min bbox width/height in input_body_shape pixels")
    parser.add_argument("--augment", action="store_true",
                        help="evaluate with train-time augmentation (default: deterministic, no aug)")
    return parser.parse_args()


def build_frozen_path_model(ckpt_path, feat_dim):
    """Build & load ONLY encoder + body_position_net + box_net from the OSX ckpt.

    We deliberately avoid get_model('test1') because the full pytorch model imports
    DCNv4 / my_decoder which require separately-compiled deformable-attention ops.
    Only the frozen top-down path is needed to reproduce the box_net hand bbox.
    """
    from common.nets.vit import StandardViT
    from common.nets.module import PositionNet, BoxNet

    embed_dim = 1024 if feat_dim == 1024 else 768
    encoder = StandardViT(img_size=(256, 192), embed_dim=embed_dim, depth=24, num_heads=16)
    body_position_net = PositionNet("body", feat_dim=feat_dim)
    box_net = BoxNet(feat_dim=feat_dim)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "network" in ckpt:
        src = ckpt["network"]
    elif "state_dict" in ckpt:
        src = ckpt["state_dict"]
    elif "model" in ckpt:
        src = ckpt["model"]
    else:
        src = ckpt
    # strip DataParallel 'module.' prefix
    src = {k.replace("module.", "", 1): v for k, v in src.items()}

    def load_submodule(module, prefix):
        sub = {}
        for k, v in src.items():
            if not k.startswith(prefix + "."):
                continue
            sub_key = k[len(prefix) + 1:]
            # encoder.last_norm.* -> encoder.norm.* (StandardViT names it 'norm')
            if prefix == "encoder" and sub_key.startswith("last_norm."):
                sub_key = sub_key.replace("last_norm.", "norm.", 1)
            sub[sub_key] = v

        # pos_embed expansion: OSX ckpt has 193 (=1 cls + 192 patch),
        # StandardViT expects 192 patch + 31 task = 223. Repeat cls token 31x.
        if prefix == "encoder" and "pos_embed" in sub:
            tgt_shape = module.pos_embed.shape
            v = sub["pos_embed"]
            if v.shape != tgt_shape and v.shape[1] == 193 and tgt_shape[1] == 223:
                cls_pos = v[:, :1, :]
                patch_pos = v[:, 1:, :]
                task_pos = cls_pos.repeat(1, 31, 1)
                sub["pos_embed"] = torch.cat([task_pos, patch_pos], dim=1)

        # drop shape-mismatched tensors so load_state_dict(strict=False) is clean
        msd = module.state_dict()
        filtered, mism = {}, []
        for k, v in sub.items():
            if k in msd and msd[k].shape == v.shape:
                filtered[k] = v
            elif k in msd:
                mism.append("%s: ckpt %s vs model %s" % (k, tuple(v.shape), tuple(msd[k].shape)))
        result = module.load_state_dict(filtered, strict=False)
        loaded = len(filtered)
        total = len(msd)
        print("  [%s] loaded %d/%d params (missing=%d, mismatch=%d)"
              % (prefix, loaded, total, len(result.missing_keys), len(mism)))
        for m in mism[:5]:
            print("     mismatch %s" % m)
        if loaded < total:
            missing_real = [k for k in result.missing_keys]
            for k in missing_real[:5]:
                print("     missing %s.%s" % (prefix, k))
        return loaded, total

    print("Loading frozen path from %s" % ckpt_path)
    enc_loaded, enc_total = load_submodule(encoder, "encoder")
    bpn_loaded, bpn_total = load_submodule(body_position_net, "body_position_net")
    box_loaded, box_total = load_submodule(box_net, "box_net")
    if min(enc_loaded, bpn_loaded, box_loaded) == 0:
        raise RuntimeError("Failed to load one of the frozen modules; check checkpoint keys.")

    encoder = encoder.cuda().eval()
    body_position_net = body_position_net.cuda().eval()
    box_net = box_net.cuda().eval()
    for m in (encoder, body_position_net, box_net):
        for p in m.parameters():
            p.requires_grad_(False)
    return encoder, body_position_net, box_net


def percentile(arr, q):
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def main():
    args = parse_args()

    from config import cfg
    cfg.set_args(args.gpu)
    # set only what we need; avoid set_additional_args -> prepare_dirs side effects
    cfg.encoder_setting = args.encoder_setting
    cfg.feat_dim = 1024 if args.encoder_setting == "osx_l" else 768
    cfg.decoder_setting = "pytorch"
    cfg.pretrained_model_path = args.pretrained_model_path
    if args.max_samples is not None and args.max_samples > 0:
        cfg.bedlam_max_samples = args.max_samples

    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    from common.utils.human_models import smpl_x
    from common.utils.transforms import restore_bbox
    import torchvision.transforms as transforms
    from data.BEDLAM.BEDLAM import BEDLAM

    # ---- dataset (deterministic eval unless --augment) ----
    ds_split = args.split if args.augment else "test"
    if not args.augment:
        print("Evaluating WITHOUT augmentation (deterministic) for a stable quality file. "
              "Use --augment to match train-time random crop distribution.")
    dataset = BEDLAM(transforms.ToTensor(), ds_split)
    n_total = len(dataset)
    print("BEDLAM samples to check: %d" % n_total)

    # stable per-sample keys from the datalist (same order as the loader, no shuffle)
    sample_keys = []
    img_paths = []
    for rec in dataset.datalist:
        sample_keys.append(str(rec.get("ann_id", "")))
        img_paths.append(str(rec.get("img_path", "")))

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_thread, pin_memory=True, drop_last=False,
    )

    # ---- model (frozen path only) ----
    encoder, body_position_net, box_net = build_frozen_path_model(
        cfg.pretrained_model_path, cfg.feat_dim)

    lhand_idx = np.array(list(smpl_x.joint_part["lhand"]), dtype=np.int64)
    rhand_idx = np.array(list(smpl_x.joint_part["rhand"]), dtype=np.int64)

    # joint(hm) -> input_body_shape scaling factors
    sx = cfg.input_body_shape[1] / cfg.output_hm_shape[2]   # 192/12
    sy = cfg.input_body_shape[0] / cfg.output_hm_shape[1]   # 256/16
    aspect = cfg.input_hand_shape[1] / cfg.input_hand_shape[0]

    # per-sample accumulators
    lhand_coverage = []
    rhand_coverage = []
    lhand_valid_cnt = []
    rhand_valid_cnt = []
    lhand_bbox_valid = []
    rhand_bbox_valid = []
    lhand_roi_valid = []
    rhand_roi_valid = []

    def per_hand(bbox_np, joint_img_np, trunc_np, idx):
        """Compute (coverage, valid_cnt, bbox_valid, roi_valid) for one hand of one sample."""
        x0, y0, x1, y1 = bbox_np
        w = x1 - x0
        h = y1 - y0
        finite = np.all(np.isfinite(bbox_np))
        bbox_ok = bool(finite and w >= args.min_wh and h >= args.min_wh)

        jt = joint_img_np[idx]          # (J,3)
        tr = trunc_np[idx].reshape(-1)  # (J,)
        valid = tr > 0.5
        valid_cnt = int(valid.sum())

        if valid_cnt == 0:
            coverage = float("nan")
        else:
            jx = jt[valid, 0] * sx
            jy = jt[valid, 1] * sy
            inside = (jx >= x0) & (jx <= x1) & (jy >= y0) & (jy <= y1)
            coverage = float(inside.sum()) / float(valid_cnt)

        roi_ok = bool(bbox_ok and valid_cnt >= args.min_joints
                      and (not np.isnan(coverage)) and coverage >= args.coverage_thr)
        cov_store = coverage if not np.isnan(coverage) else -1.0
        return cov_store, valid_cnt, bbox_ok, roi_ok

    n_checked = 0
    with torch.no_grad():
        for bi, (inputs, targets, meta_info) in enumerate(loader):
            import torch.nn.functional as F
            body_img = F.interpolate(inputs["img"].cuda(), cfg.input_body_shape)

            img_feat, _ = encoder(body_img)
            body_joint_hm, _ = body_position_net(img_feat)
            lc, ls, rc, rs, _, _ = box_net(img_feat, body_joint_hm.detach())

            lhand_bbox = restore_bbox(lc, ls, aspect, 2.0).detach().cpu().numpy()  # (B,4) xyxy
            rhand_bbox = restore_bbox(rc, rs, aspect, 2.0).detach().cpu().numpy()

            joint_img = targets["smplx_joint_img"].cpu().numpy()        # (B,137,3)
            trunc = meta_info["smplx_joint_trunc"].cpu().numpy()        # (B,137,1)

            B = lhand_bbox.shape[0]
            for s in range(B):
                lc_cov, lc_cnt, lc_bb, lc_roi = per_hand(lhand_bbox[s], joint_img[s], trunc[s], lhand_idx)
                rc_cov, rc_cnt, rc_bb, rc_roi = per_hand(rhand_bbox[s], joint_img[s], trunc[s], rhand_idx)
                lhand_coverage.append(lc_cov); lhand_valid_cnt.append(lc_cnt)
                lhand_bbox_valid.append(lc_bb); lhand_roi_valid.append(lc_roi)
                rhand_coverage.append(rc_cov); rhand_valid_cnt.append(rc_cnt)
                rhand_bbox_valid.append(rc_bb); rhand_roi_valid.append(rc_roi)
            n_checked += B
            if (bi + 1) % 20 == 0 or n_checked >= n_total:
                print("  checked %d / %d" % (n_checked, n_total))

    # ---- to arrays ----
    lhand_coverage = np.asarray(lhand_coverage, dtype=np.float32)
    rhand_coverage = np.asarray(rhand_coverage, dtype=np.float32)
    lhand_valid_cnt = np.asarray(lhand_valid_cnt, dtype=np.int32)
    rhand_valid_cnt = np.asarray(rhand_valid_cnt, dtype=np.int32)
    lhand_bbox_valid = np.asarray(lhand_bbox_valid, dtype=bool)
    rhand_bbox_valid = np.asarray(rhand_bbox_valid, dtype=bool)
    lhand_roi_valid = np.asarray(lhand_roi_valid, dtype=bool)
    rhand_roi_valid = np.asarray(rhand_roi_valid, dtype=bool)
    sample_keys = np.asarray(sample_keys[:n_checked])
    img_paths = np.asarray(img_paths[:n_checked])

    # ---- save .npz ----
    out_path = osp.abspath(args.output)
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        sample_keys=sample_keys,
        img_paths=img_paths,
        lhand_roi_valid=lhand_roi_valid.astype(np.float32),
        rhand_roi_valid=rhand_roi_valid.astype(np.float32),
        lhand_coverage=lhand_coverage,
        rhand_coverage=rhand_coverage,
        lhand_valid_joint_count=lhand_valid_cnt,
        rhand_valid_joint_count=rhand_valid_cnt,
        lhand_bbox_valid=lhand_bbox_valid.astype(np.float32),
        rhand_bbox_valid=rhand_bbox_valid.astype(np.float32),
        coverage_thr=np.float32(args.coverage_thr),
        min_joints=np.int32(args.min_joints),
        min_wh=np.float32(args.min_wh),
        augmented=np.bool_(args.augment),
    )
    print("\nSaved quality file -> %s" % out_path)

    # ---- aggregate report ----
    def ratio(mask):
        return float(mask.mean()) if mask.size else float("nan")

    either_ok = lhand_roi_valid | rhand_roi_valid
    both_ok = lhand_roi_valid & rhand_roi_valid
    invalid_bbox = (~lhand_bbox_valid) & (~rhand_bbox_valid)
    no_valid_joints = (lhand_valid_cnt == 0) & (rhand_valid_cnt == 0)

    # coverage stats only over samples that HAVE valid joints (coverage >= 0)
    l_cov_valid = lhand_coverage[lhand_coverage >= 0.0]
    r_cov_valid = rhand_coverage[rhand_coverage >= 0.0]
    both_cov = np.concatenate([l_cov_valid, r_cov_valid]) if (l_cov_valid.size or r_cov_valid.size) else np.array([])

    print("\n" + "=" * 64)
    print("BEDLAM hand-ROI quality report")
    print("=" * 64)
    print("checked samples            : %d" % n_checked)
    print("coverage_thr=%.2f  min_joints=%d  min_wh=%.1f  augment=%s"
          % (args.coverage_thr, args.min_joints, args.min_wh, args.augment))
    print("-" * 64)
    print("left  hand ok ratio        : %.4f" % ratio(lhand_roi_valid))
    print("right hand ok ratio        : %.4f" % ratio(rhand_roi_valid))
    print("either hand ok ratio       : %.4f" % ratio(either_ok))
    print("both hands ok ratio        : %.4f" % ratio(both_ok))
    print("invalid bbox ratio (both)  : %.4f" % ratio(invalid_bbox))
    print("no valid GT hand joints    : %.4f" % ratio(no_valid_joints))
    print("-" * 64)
    print("coverage (over hands with valid GT joints, n=%d)" % both_cov.size)
    print("  mean   : %.4f" % (float(both_cov.mean()) if both_cov.size else float("nan")))
    print("  median : %.4f" % percentile(both_cov, 50))
    print("  p10    : %.4f" % percentile(both_cov, 10))
    print("  p90    : %.4f" % percentile(both_cov, 90))
    print("-" * 64)
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]
    print("coverage bins (fraction of valid-joint hands):")
    for lo, hi in bins:
        if both_cov.size:
            m = (both_cov >= lo) & (both_cov < hi)
            frac = float(m.mean())
        else:
            frac = float("nan")
        hi_disp = min(hi, 1.0)
        print("  [%.1f, %.1f): %.4f" % (lo, hi_disp, frac))
    print("=" * 64)

    # ---- verdict (docs/bedlam_problems.txt) ----
    either = ratio(either_ok)
    print("\nVerdict (based on either-hand-ok ratio = %.4f):" % either)
    if either < 0.5:
        print("  < 50%: BEDLAM is NOT a good strong hand supervision under the frozen ROI.")
        print("         Use masked hand loss only, or lower BEDLAM sampling ratio.")
    elif either < 0.7:
        print("  50-70%: Keep BEDLAM but REQUIRE the ROI-quality mask and stay InterHand-heavy.")
    else:
        print("  > 70%: BEDLAM is a useful auxiliary hand source; still keep the mask")
        print("         to prevent tail bad samples from polluting training.")


if __name__ == "__main__":
    main()
