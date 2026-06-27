#!/usr/bin/env python3
"""Quantify how much the FROZEN body path diverges between the original OSX
(``--decoder_setting normal``, MMCV ViT) and the pytorch port
(``--decoder_setting pytorch``), using the per-sample params dumped by
``test.py --dump_analysis`` (``crosspath_<testset>.npz``).

Why this exists: the UBody hand comparison is OSX-normal (MMCV path) vs the
fork (ported StandardViT encoder + remapped frozen body). The frozen body is
loaded from the SAME osx_l.pth.tar, so root/body pose + shape + cam_trans
SHOULD be ~identical; any residual is encoder-port error. If that residual is
comparable to the UBody [wa] 0.219->0.229 gap, then part of the "hand" gap is
actually the port, and the comparison must be reported as confounded.

``bb2img_trans`` is a model-INDEPENDENT per-sample fingerprint (test-time
augmentation is identity, so it depends only on the GT bbox + image size). We
use it to assert the two runs are row-aligned before comparing.

Run test.py twice with --dump_analysis (same testset, same test_batch_size):
  A = OSX normal      (--decoder_setting normal, no --continue_train_path)
  B = pytorch snap    (--decoder_setting pytorch, --continue_train_path ...)
then:
  python tool/analysis/crosspath_compare.py \
    --a output/eval_ubody/result/pretrained/crosspath_UBody.npz \
    --b output/eval_joint_polish_f/result/snapshot_2/crosspath_UBody.npz
"""
import argparse

import numpy as np

R2D = 180.0 / np.pi


def _stats(x):
    x = np.abs(np.asarray(x, dtype=np.float64)).reshape(-1)
    if x.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(x.mean()), float(np.median(x)), float(np.percentile(x, 95)), float(x.max())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--a', required=True, help='crosspath npz for OSX normal')
    p.add_argument('--b', required=True, help='crosspath npz for the pytorch path')
    p.add_argument('--align-tol', type=float, default=1e-3,
                   help='max bb2img_trans abs diff allowed to consider rows aligned')
    args = p.parse_args()

    za = np.load(args.a)
    zb = np.load(args.b)

    # ---- alignment check via the model-independent bb2img_trans fingerprint ----
    if 'bb2img_trans' in za.files and 'bb2img_trans' in zb.files:
        fa = np.asarray(za['bb2img_trans'], dtype=np.float64)
        fb = np.asarray(zb['bb2img_trans'], dtype=np.float64)
        if fa.shape != fb.shape:
            raise SystemExit("row/shape mismatch A=%s B=%s -> runs not aligned "
                             "(same testset + test_batch_size required)" % (fa.shape, fb.shape))
        max_d = float(np.abs(fa - fb).max()) if fa.size else 0.0
        if max_d > args.align_tol:
            raise SystemExit("bb2img_trans mismatch (max abs diff %.3e > tol %.1e): the two "
                             "runs are NOT over the same samples in the same order."
                             % (max_d, args.align_tol))
        print("[align] OK: %d samples, bb2img_trans max abs diff %.2e" % (fa.shape[0], max_d))
    else:
        print("[align] WARNING: no bb2img_trans anchor present; assuming index alignment.")

    print("=" * 74)
    print("Frozen-body cross-path divergence   A = OSX normal   vs   B = pytorch port")
    print("units: root/body/wrist in DEGREES, shape in beta units, cam_trans in model units")
    print("-" * 74)
    print("%-26s %10s %10s %10s %10s" % ('param (|A-B|)', 'mean', 'median', 'p95', 'max'))

    def report(name, diff, scale=1.0):
        m, md, p95, mx = _stats(diff * scale)
        print("%-26s %10.4f %10.4f %10.4f %10.4f" % (name, m, md, p95, mx))

    for key, scale, label in (('smplx_root_pose', R2D, 'root_pose (deg)'),
                              ('smplx_body_pose', R2D, 'body_pose all (deg)'),
                              ('smplx_shape', 1.0, 'shape betas'),
                              ('cam_trans', 1.0, 'cam_trans')):
        if key in za.files and key in zb.files:
            d = np.asarray(za[key], dtype=np.float64) - np.asarray(zb[key], dtype=np.float64)
            report(label, d, scale)

    # wrist-specific body_pose -- the DOF that leaks into UBody [wa] after wrist
    # alignment. body_pose excludes pelvis, so SMPL-X L/R wrist (joints 20/21)
    # land at body_pose cols 57:60 and 60:63.
    if 'smplx_body_pose' in za.files and 'smplx_body_pose' in zb.files:
        d = np.asarray(za['smplx_body_pose'], dtype=np.float64) - np.asarray(zb['smplx_body_pose'], dtype=np.float64)
        if d.ndim == 2 and d.shape[1] >= 63:
            report('  L_wrist (deg)', d[:, 57:60], R2D)
            report('  R_wrist (deg)', d[:, 60:63], R2D)

    print("=" * 74)
    print("Read: if wrist/body mean << 1 deg and cam_trans mean is tiny, the frozen")
    print("path is faithful -> the UBody [wa] gap is the hand head, not the port.")
    print("If wrist/cam diffs are on the order of the 0.219->0.229 gap, the UBody")
    print("comparison is partly confounded and must be reported as such.")


if __name__ == '__main__':
    main()
