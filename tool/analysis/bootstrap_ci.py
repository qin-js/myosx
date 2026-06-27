#!/usr/bin/env python3
"""Paired / unpaired bootstrap confidence intervals for the hand metrics dumped
by ``test.py --dump_analysis`` (``bootstrap_<testset>.npz``).

Each .npz holds one 1D float array per metric key (one entry per hand/sample).
Two eval runs over the SAME testset (test loader is shuffle-off, and every
visibility filter in the evaluator is GT-only / model-independent) produce
INDEX-ALIGNED arrays. We therefore default to a PAIRED bootstrap on the
per-hand difference, which is far tighter than comparing two independent means
and directly answers "is OSX - ours significantly different from zero?".

Why this exists: the result.txt files only report point estimates. Gaps like
UBody [wa] 0.219 vs 0.229 or HInt wa 0.487 vs 0.480 need a CI before they can
be called a real win/loss rather than noise.

Examples
--------
UBody wrist-aligned NME, OSX baseline (A) vs ours (B), error metric:
  python tool/analysis/bootstrap_ci.py \
    --a output/eval_ubody/result/pretrained/bootstrap_UBody.npz \
    --b output/eval_joint_polish_f/result/snapshot_2/bootstrap_UBody.npz \
    --key hand2d_nme_wa --lower-is-better

InterHand PA-MPJPE per hand:
  python tool/analysis/bootstrap_ci.py --a A.npz --b B.npz \
    --key pa_mpjpe --lower-is-better
"""
import argparse

import numpy as np


def _ci(samples, alpha):
    lo = float(np.percentile(samples, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def _boot_mean(rng, x, n_boot):
    """Bootstrap distribution of the mean of x (resample rows with replacement)."""
    n = x.shape[0]
    if n == 0:
        return np.zeros(n_boot, dtype=np.float64)
    idx = rng.integers(0, n, size=(n_boot, n))
    return x[idx].mean(axis=1)


def _load_metric(path, key):
    z = np.load(path)
    if key not in z.files:
        raise SystemExit("key '%s' not in %s\n  available: %s"
                         % (key, path, ', '.join(sorted(z.files))))
    a = np.asarray(z[key], dtype=np.float64)
    return a[np.isfinite(a)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--a', required=True, help='npz for model A (e.g. OSX baseline)')
    p.add_argument('--b', required=True, help='npz for model B (e.g. ours)')
    p.add_argument('--key', required=True, help='metric key, e.g. hand2d_nme_wa')
    p.add_argument('--key-b', default=None,
                   help='metric key in B if it differs from --key (rare)')
    p.add_argument('--n-boot', type=int, default=10000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--alpha', type=float, default=0.05, help='1-alpha = CI level (default 95%%)')
    p.add_argument('--lower-is-better', action='store_true',
                   help='set for error metrics (NME / MPJPE); flips the "better" verdict')
    p.add_argument('--no-paired', action='store_true',
                   help='force unpaired bootstrap even when the two arrays align')
    args = p.parse_args()

    a = _load_metric(args.a, args.key)
    b = _load_metric(args.b, args.key_b or args.key)

    rng = np.random.default_rng(args.seed)
    ba = _boot_mean(rng, a, args.n_boot)
    bb = _boot_mean(rng, b, args.n_boot)
    la, ha = _ci(ba, args.alpha)
    lb, hb = _ci(bb, args.alpha)
    conf = 100.0 * (1.0 - args.alpha)

    print("=" * 70)
    print("metric: %s   (n_boot=%d, %.0f%% CI, seed=%d)"
          % (args.key, args.n_boot, conf, args.seed))
    print("-" * 70)
    print("A  mean=%.4f  CI=[%.4f, %.4f]  n=%d  | %s"
          % (a.mean(), la, ha, a.size, args.a))
    print("B  mean=%.4f  CI=[%.4f, %.4f]  n=%d  | %s"
          % (b.mean(), lb, hb, b.size, args.b))

    paired = (not args.no_paired) and (a.size == b.size) and a.size > 0
    print("-" * 70)
    if paired:
        d = a - b  # A minus B, per hand
        bd = _boot_mean(rng, d, args.n_boot)
        ld, hd = _ci(bd, args.alpha)
        print("PAIRED delta = A - B   mean=%.4f  CI=[%.4f, %.4f]" % (d.mean(), ld, hd))
        print("  P(delta>0 under bootstrap) = %.3f" % float((bd > 0).mean()))
        sig = (ld > 0) or (hd < 0)
        if sig:
            better = 'B' if ((d.mean() > 0) == args.lower_is_better) else 'A'
            print("  => SIGNIFICANT at alpha=%.2f: %s is better (%.0f%% CI excludes 0)"
                  % (args.alpha, better, conf))
        else:
            print("  => NOT significant at alpha=%.2f: CI includes 0 (gap within noise)"
                  % args.alpha)
    else:
        if not args.no_paired and a.size != b.size:
            print("NOTE: lengths differ (A=%d, B=%d) -> falling back to UNPAIRED."
                  % (a.size, b.size))
        bd = ba - bb
        ld, hd = _ci(bd, args.alpha)
        print("UNPAIRED delta of means = A - B   mean=%.4f  CI=[%.4f, %.4f]"
              % (a.mean() - b.mean(), ld, hd))
        sig = (ld > 0) or (hd < 0)
        print("  => %s at alpha=%.2f"
              % ('SIGNIFICANT' if sig else 'NOT significant', args.alpha))
    print("=" * 70)


if __name__ == '__main__':
    main()
