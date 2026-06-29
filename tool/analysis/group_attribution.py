#!/usr/bin/env python3
"""Grouped attribution of the UBody wrist-aligned hand-NME gap.

Consumes the per-hand, per-joint arrays dumped by ``test.py --dump_analysis``
(``perhand_<testset>.npz``; see ``docs/wa2d_group_attribution.md``). Each run
produces index-aligned arrays because the UBody test loader is shuffle-off and
every visibility filter is GT-only / model-independent (same rationale as
``bootstrap_ci.py``). We compare a reference (A, e.g. OSX normal) against a
candidate (B, e.g. our snapshot) on the SAME hands, by computing the paired
per-joint delta B - A and slicing it by hand attributes (size / occlusion /
left-right / finger x level).

Goal: the UBody [wa] gap (OSX 0.219 vs ours 0.229) is cleanly the hand head
after the encoder fix. But "hand head" splits into sub-causes whose remedy
differs:

    gap concentrated on      -> root cause            -> remedy
    small hands / low-res    ROI crop quality         Route C / WA2D ROI part
    heavy occlusion          supervision/visibility   more wild-hand sup, soft gate
    clear big hands, j2/j3   decoder hand prior       WA2D loss / teacher distill
    tip/j3 (end)             end articulation         tip/j3 weighting (tried)

This script localizes the gap so the next route is chosen on evidence, not
guess. It also prints the worst-N failure-case image paths for visual diffing.

Examples
--------
python tool/analysis/group_attribution.py \
  --a output/eval_ubody/result/perhand_UBody.npz \
  --b output/eval_joint_polish_f/result/snapshot_2/perhand_UBody.npz
"""
import argparse

import numpy as np

# SMPL-X hand joint order within one 21-pt hand (wrist + thumb/index/middle/
# ring/pinky, 4 pts each). Matches data/UBody/UBody.py _hand2d_finger_idx /
# _hand2d_level_idx and the coco_hand_joint ordering in the WA2D loss.
WRIST = 0
FINGER = {
    'thumb':  np.array([1, 2, 3, 4]),
    'index':  np.array([5, 6, 7, 8]),
    'middle': np.array([9, 10, 11, 12]),
    'ring':   np.array([13, 14, 15, 16]),
    'pinky':  np.array([17, 18, 19, 20]),
}
LEVEL = {
    'j1':  np.array([1, 5, 9, 13, 17]),
    'j2':  np.array([2, 6, 10, 14, 18]),
    'j3':  np.array([3, 7, 11, 15, 19]),
    'tip': np.array([4, 8, 12, 16, 20]),
}


def _stack(npz, key):
    return np.asarray(npz[key], dtype=np.float64)


def _maybe_str_array(npz, key):
    if key not in npz.files:
        return None
    return np.asarray(npz[key])


def _as_str_array(arr):
    if arr is None:
        return None
    out = []
    for x in arr:
        if isinstance(x, (bytes, np.bytes_)):
            out.append(x.decode())
        else:
            out.append(str(x))
    return np.asarray(out)


def _per_joint_mask(joint_valid, idxs):
    """(N,) bool: hands where at least one of the given joints is visible on
    BOTH models (so the delta is defined). joint_valid is (N,21)."""
    return (joint_valid[:, idxs].sum(axis=1) > 0)


def _group_block(name, hand_mask, wa_a, wa_b, joint_valid, idxs):
    """Print mean error A vs B and paired delta over the given joint set, on
    hands in hand_mask. delta>0 means B (ours) is worse than A (ref)."""
    jm = hand_mask & _per_joint_mask(joint_valid, idxs)
    n = int(jm.sum())
    if n == 0:
        print(f'  {name:22s} N_hands={n:5d}  (no samples)')
        return
    # restrict to the joint set, only where GT-visible
    cols = joint_valid[jm][:, idxs] > 0
    a = wa_a[jm][:, idxs][cols]
    b = wa_b[jm][:, idxs][cols]
    delta = b - a
    print(f'  {name:22s} N_hands={n:5d}  '
          f'A(ref)={a.mean():.4f}  B(ours)={b.mean():.4f}  '
          f'delta={delta.mean():+.4f}  gain%={100.0*delta.mean()/max(a.mean(),1e-9):+.1f}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--a', required=True, help='reference perhand npz (e.g. OSX normal)')
    p.add_argument('--b', required=True, help='candidate perhand npz (e.g. our snapshot)')
    p.add_argument('--worst', type=int, default=30, help='print worst-N failure cases')
    args = p.parse_args()

    A = np.load(args.a, allow_pickle=True)
    B = np.load(args.b, allow_pickle=True)

    # --- alignment check -------------------------------------------------
    wa_a = _stack(A, 'hand2d_wa_joints')        # (N,21)
    wa_b = _stack(B, 'hand2d_wa_joints')
    va = _stack(A, 'hand2d_joint_valid')        # (N,21)
    vb = _stack(B, 'hand2d_joint_valid')
    size = _stack(A, 'hand2d_hand_size')
    size_b = _stack(B, 'hand2d_hand_size')
    nvis = _stack(A, 'hand2d_n_visible')
    nvis_b = _stack(B, 'hand2d_n_visible')
    side = _stack(A, 'hand2d_side')
    side_b = _stack(B, 'hand2d_side')
    img_path_a = _as_str_array(_maybe_str_array(A, 'hand2d_img_path'))
    img_path_b = _as_str_array(_maybe_str_array(B, 'hand2d_img_path'))
    img_path = img_path_b if img_path_b is not None else img_path_a

    if wa_a.shape != wa_b.shape:
        raise SystemExit(f'ALIGNMENT FAILED: A wa_joints {wa_a.shape} vs B {wa_b.shape}. '
                         f'Both runs must use the same UBody test split / '
                         f'test_sample_interval / test_batch_size.')
    if not np.array_equal(va, vb):
        # GT visibility should be identical across models; if not, the GT
        # differs (different annot cache) and paired deltas are not trustworthy.
        mismatch = int((va != vb).any(axis=1).sum())
        raise SystemExit(f'ALIGNMENT FAILED: joint_valid differs on {mismatch} hands. '
                         f'GT visibility must match between runs (same UBODY_ANNOTATION_DIR).')
    if not np.array_equal(side, side_b):
        mismatch = int((side != side_b).sum())
        raise SystemExit(f'ALIGNMENT FAILED: hand side differs on {mismatch} hands.')
    if not np.allclose(size, size_b, rtol=1e-7, atol=1e-7):
        mismatch = int((~np.isclose(size, size_b, rtol=1e-7, atol=1e-7)).sum())
        raise SystemExit(f'ALIGNMENT FAILED: hand_size differs on {mismatch} hands.')
    if not np.allclose(nvis, nvis_b, rtol=0.0, atol=0.0):
        mismatch = int((nvis != nvis_b).sum())
        raise SystemExit(f'ALIGNMENT FAILED: n_visible differs on {mismatch} hands.')
    if img_path_a is not None and img_path_b is not None and not np.array_equal(img_path_a, img_path_b):
        mismatch = int((img_path_a != img_path_b).sum())
        raise SystemExit(f'ALIGNMENT FAILED: img_path differs on {mismatch} hands.')

    n = wa_a.shape[0]
    # valid mask: a hand counted only when wrist valid (matches hand2d_nme_wa).
    wrist_valid = (va[:, WRIST] > 0)
    print('=' * 76)
    print(f'Grouped attribution   A(ref)={args.a}')
    print(f'                       B(ours)={args.b}')
    print(f'paired hands (wrist valid): {n}   delta = B(ours) - A(ref)  (>0 = ours worse)')
    print('=' * 76)

    # --- overall ---------------------------------------------------------
    # Reproduce the headline [wa] NME on the paired subset, as a sanity check
    # against the result.txt number.
    ov_mask = wrist_valid
    allj = np.arange(21)
    _group_block('overall [wa]', ov_mask, wa_a, wa_b, va, allj)

    # --- A. by hand size (GT hand-kpt bbox diagonal) ---------------------
    p33, p66 = np.percentile(size[wrist_valid], [33, 66])
    print('-' * 76)
    print(f'by hand size  (percentiles p33={p33:.0f}px p66={p66:.0f}px):')
    _group_block(f'small (<{p33:.0f})', wrist_valid & (size < p33), wa_a, wa_b, va, allj)
    _group_block(f'mid', wrist_valid & (size >= p33) & (size < p66), wa_a, wa_b, va, allj)
    _group_block(f'large (>={p66:.0f})', wrist_valid & (size >= p66), wa_a, wa_b, va, allj)

    # --- B. by occlusion (visible-joint count) ---------------------------
    print('-' * 76)
    print('by occlusion  (visible joints out of 21):')
    _group_block('heavy (<8)', wrist_valid & (nvis < 8), wa_a, wa_b, va, allj)
    _group_block('mid (8-15)', wrist_valid & (nvis >= 8) & (nvis < 16), wa_a, wa_b, va, allj)
    _group_block('clear (>=16)', wrist_valid & (nvis >= 16), wa_a, wa_b, va, allj)

    # --- C. by left / right ---------------------------------------------
    print('-' * 76)
    print('by side:')
    _group_block('left', wrist_valid & (side == 0), wa_a, wa_b, va, allj)
    _group_block('right', wrist_valid & (side == 1), wa_a, wa_b, va, allj)

    # --- D. by finger (same hands, per-finger joint set) ----------------
    print('-' * 76)
    print('by finger  (paired delta on each finger joint set):')
    for name, idxs in FINGER.items():
        _group_block(name, wrist_valid, wa_a, wa_b, va, idxs)

    # --- E. by level (same hands, per-level joint set) ------------------
    print('-' * 76)
    print('by level  (paired delta on each phalanx-depth joint set):')
    for name, idxs in LEVEL.items():
        _group_block(name, wrist_valid, wa_a, wa_b, va, idxs)

    # --- F. failure cases (worst-N by our per-hand wa NME) --------------
    print('-' * 76)
    per_hand_b = (wa_b * (va > 0)).sum(axis=1) / (va > 0).sum(axis=1).clip(min=1)
    per_hand_a = (wa_a * (va > 0)).sum(axis=1) / (va > 0).sum(axis=1).clip(min=1)
    order = np.argsort(per_hand_b)[::-1]  # worst (highest our-error) first
    print(f'worst-{args.worst} hands by our [wa] NME  (size=px diag, nvis=visible joints):')
    for i in order[:args.worst]:
        ip = img_path[i] if img_path is not None else '?'
        if isinstance(ip, (bytes, np.bytes_)):
            ip = ip.decode()
        print(f'  wa_ours={per_hand_b[i]:.3f}  wa_ref={per_hand_a[i]:.3f}  '
              f'delta={per_hand_b[i]-per_hand_a[i]:+.3f}  size={size[i]:.0f}  '
              f'nvis={nvis[i]:.0f}  side={"L" if side[i]==0 else "R"}  {ip}')
    print('-' * 76)
    print('Read: delta>0 on a group = ours worse than ref there. The group(s)')
    print('carrying the gap point at the root cause (see docstring / handoff doc).')
    print('Visually diff the worst-N img_paths between the two models.')


if __name__ == '__main__':
    main()
