#!/usr/bin/env python3
"""Localize where the StandardViT port (pytorch path) diverges from the MMCV
ViT (normal path), using the encoder I/O dumped by ``test.py --dump_encoder_n N``
(``encoder_<testset>.npz``).

We earlier showed (crosspath_compare) that the FROZEN body output diverges
(~1.4 deg median at the wrist) even though body_regressor / body_position_net
are the SAME code+weights in both paths. That isolates the cause to the
encoder. This script confirms it and characterizes it:

  1. encoder INPUT (body_img) must match -> proves the divergence is the encoder
     itself, not the data pipeline / preprocessing.
  2. img_feat (the B,C,16,12 patch feature map) divergence.
  3. task_tokens divergence PER TOKEN, grouped by role
     (shape / cam / expr / jaw / hand / body), so we see WHICH outputs drift.
     The earlier finding (shape betas ~identical, pose/cam drift) predicts the
     shape token should be clean and the body/cam/pose tokens should not.

Run test.py twice with the SAME testset + test_batch_size:
  A: --decoder_setting normal  --dump_encoder_n 32   (MMCV ViT)
  B: --decoder_setting pytorch --dump_encoder_n 32 --continue_train_path ...  (StandardViT)
then:
  python tool/analysis/encoder_compare.py \
    --a <A_result_dir>/encoder_UBody.npz \
    --b <B_result_dir>/encoder_UBody.npz
"""
import argparse

import numpy as np

EPS = 1e-12

# task_tokens layout (see model forward): 0=shape, 1=cam, 2=expr, 3=jaw,
# 4-5=hand, 6-30=body_pose.
TOKEN_GROUPS = (
    ('shape', [0]),
    ('cam', [1]),
    ('expr', [2]),
    ('jaw', [3]),
    ('hand', [4, 5]),
    ('body', list(range(6, 31))),
)


def _rel_l2_per_sample(a, b):
    """||a-b|| / ||a|| per sample, flattening all non-sample dims."""
    n = a.shape[0]
    af = a.reshape(n, -1).astype(np.float64)
    bf = b.reshape(n, -1).astype(np.float64)
    num = np.linalg.norm(af - bf, axis=1)
    den = np.linalg.norm(af, axis=1) + EPS
    return num / den


def _fmt_stats(x):
    return ("mean=%.3e  median=%.3e  p95=%.3e  max=%.3e"
            % (float(np.mean(x)), float(np.median(x)),
               float(np.percentile(x, 95)), float(np.max(x))))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--a', required=True, help='encoder npz for MMCV ViT (normal)')
    p.add_argument('--b', required=True, help='encoder npz for StandardViT (pytorch)')
    p.add_argument('--align-tol', type=float, default=1e-3,
                   help='max bb2img_trans abs diff allowed to consider rows aligned')
    p.add_argument('--noise-floor', type=float, default=1e-4,
                   help='relative-L2 below this is treated as float noise (faithful)')
    args = p.parse_args()

    za = np.load(args.a)
    zb = np.load(args.b)

    # ---- alignment via the model-independent bb2img_trans fingerprint ----
    if 'bb2img_trans' in za.files and 'bb2img_trans' in zb.files:
        fa = np.asarray(za['bb2img_trans'], dtype=np.float64)
        fb = np.asarray(zb['bb2img_trans'], dtype=np.float64)
        n = min(fa.shape[0], fb.shape[0])
        fa, fb = fa[:n], fb[:n]
        max_d = float(np.abs(fa - fb).max()) if fa.size else 0.0
        if max_d > args.align_tol:
            raise SystemExit("bb2img_trans mismatch (max abs diff %.3e > tol %.1e): the two "
                             "runs are not over the same samples in the same order." % (max_d, args.align_tol))
        print("[align] OK: %d samples, bb2img_trans max abs diff %.2e" % (n, max_d))
    else:
        n = None
        print("[align] WARNING: no bb2img_trans anchor; assuming index alignment.")

    def take(z, key):
        x = np.asarray(z[key], dtype=np.float64)
        return x if n is None else x[:n]

    print("=" * 74)
    print("Encoder fidelity   A = MMCV ViT (normal)   vs   B = StandardViT (pytorch)")
    print("=" * 74)

    # ---- 1. input must match ----
    if '_enc_input' in za.files and '_enc_input' in zb.files:
        ia, ib = take(za, '_enc_input'), take(zb, '_enc_input')
        in_max = float(np.abs(ia - ib).max())
        verdict = "OK (same input -> any output drift is the encoder)" if in_max <= args.align_tol \
            else "MISMATCH -> drift is upstream of the encoder (preprocessing), not the ViT!"
        print("[input ] body_img  max|A-B| = %.3e   %s" % (in_max, verdict))
    else:
        print("[input ] (no _enc_input dumped; cannot confirm identical encoder input)")

    # ---- 2. img_feat ----
    if '_enc_img_feat' in za.files and '_enc_img_feat' in zb.files:
        a, b = take(za, '_enc_img_feat'), take(zb, '_enc_img_feat')
        rel = _rel_l2_per_sample(a, b)
        print("-" * 74)
        print("[img_feat] relative L2 per sample:  %s" % _fmt_stats(rel))
        print("           |A| mean=%.3e   |A-B| mean=%.3e"
              % (float(np.linalg.norm(a.reshape(a.shape[0], -1), axis=1).mean()),
                 float(np.linalg.norm((a - b).reshape(a.shape[0], -1), axis=1).mean())))

    # ---- 3. task_tokens per token ----
    if '_enc_task_tokens' in za.files and '_enc_task_tokens' in zb.files:
        a, b = take(za, '_enc_task_tokens'), take(zb, '_enc_task_tokens')  # (N, 31, C)
        ntok = a.shape[1]
        # per-token relative L2 averaged over samples
        per_tok = np.zeros(ntok, dtype=np.float64)
        for t in range(ntok):
            per_tok[t] = float(np.mean(_rel_l2_per_sample(a[:, t], b[:, t])))
        print("-" * 74)
        print("[task_tokens] mean relative L2 by role (token indices):")
        for name, idxs in TOKEN_GROUPS:
            idxs = [i for i in idxs if i < ntok]
            if idxs:
                print("   %-6s tok%-7s  %.3e" % (name, str(idxs[0]) if len(idxs) == 1
                                                 else '%d-%d' % (idxs[0], idxs[-1]),
                                                 float(per_tok[idxs].mean())))
        order = np.argsort(per_tok)[::-1][:5]
        print("   top-5 divergent tokens:  " +
              "  ".join("tok%d=%.2e" % (int(t), per_tok[int(t)]) for t in order))

    print("=" * 74)
    print("Read: input max|A-B| ~ 0 confirms identical encoder input. If img_feat /")
    print("task-token relative L2 >> %.0e (noise floor), the StandardViT port is NOT" % args.noise_floor)
    print("faithful. A clean shape token but drifting body/cam/pose tokens points at")
    print("positional / spatial handling (pos_embed, attention numerics, norm).")


if __name__ == '__main__':
    main()
