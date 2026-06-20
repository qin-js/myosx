#!/usr/bin/env python3
"""离线检查"应当冻结的分支"在两个 checkpoint 之间是否真的没变（不需要 GPU）。

用途：Stage 2「冻手训脸」训完后，确认手部权重和暖启用的 ckpt 一模一样（Δ=0）。
权重相等 ⇒ 前向输出必然相等 ⇒ InterHand 手部指标必然不变，无需再跑评估去验证。

典型用法
--------
  # 手部应当冻结（Δ=0），脸部应当被训练（Δ>0）
  python tool/check_hand_frozen.py \
    --ref  output/interhand_bedlam_c/model_dump/snapshot_8.pth.tar \
    --ckpt output/face_ubody_e/model_dump/snapshot_11.pth.tar \
    --also_face

  # 一次比多个 snapshot
  python tool/check_hand_frozen.py --ref .../snapshot_8.pth.tar \
    --ckpt .../snapshot_8.pth.tar .../snapshot_9.pth.tar .../snapshot_11.pth.tar

判读
----
  HAND  全部 Δ=0      → ✅ 手确实冻住了
  HAND  某模块 Δ>0    → ❌ 冻结/优化器有 bug（手被更新了）
  HAND  MISSING       → ❌ 保存漏了手（trainable_module_names 没保住）→ eval 会回退到错误权重
  FACE  Δ>0           → ✅ 脸确实训到了（--also_face 时报告）
"""
import argparse
import os

import torch

HAND_PREFIXES = ("hand_position_net.", "hand_decoder.", "hand_regressor.")
FACE_PREFIXES = ("face_position_net.", "face_decoder.", "face_regressor.")


def load_network(path):
    """读取 ckpt 的 network dict，剥掉 DataParallel 的 'module.' 前缀，只留张量。"""
    ckpt = torch.load(path, map_location="cpu")
    net = ckpt.get("network", ckpt) if isinstance(ckpt, dict) else ckpt
    out = {}
    for k, v in net.items():
        if not torch.is_tensor(v):
            continue
        out[k[7:] if k.startswith("module.") else k] = v
    return out


def group_of(key, prefixes):
    for p in prefixes:
        if key.startswith(p):
            return p.rstrip(".")
    return None


def compare(ref, ckpt, prefixes):
    """按模块前缀分组，统计 ref→ckpt 的最大绝对差 / 变化数 / 缺失数。"""
    stats = {}
    for k, v in ref.items():
        g = group_of(k, prefixes)
        if g is None:
            continue
        s = stats.setdefault(g, {"max": 0.0, "n": 0, "changed": 0, "missing": 0, "worst": None})
        s["n"] += 1
        if k not in ckpt:
            s["missing"] += 1
            continue
        if ckpt[k].shape != v.shape:
            s["changed"] += 1
            s["max"] = float("inf")
            s["worst"] = k + " (shape mismatch)"
            continue
        d = (v.float() - ckpt[k].float()).abs().max().item()
        if d > s["max"]:
            s["max"], s["worst"] = d, k
        if d > 0:
            s["changed"] += 1
    return stats


def report_frozen(stats):
    """期望冻结：无缺失且全 Δ=0。返回 True/False。"""
    if not stats:
        print("  (没有匹配到 hand_* 张量 —— ref 里就没有手部权重？)")
        return False
    ok = True
    for g in sorted(stats):
        s = stats[g]
        if s["missing"]:
            print(f"  {g:22s} ❌ MISSING {s['missing']}/{s['n']}（保存漏了！eval 会用错权重）")
            ok = False
        elif s["max"] > 0:
            print(f"  {g:22s} ❌ 变了 {s['changed']}/{s['n']}  max|Δ|={s['max']:.3e}  worst={s['worst']}")
            ok = False
        else:
            print(f"  {g:22s} ✅ Δ=0  ({s['n']} tensors)")
    return ok


def report_trained(stats):
    """期望已训练：Δ>0 才正常（仅信息性，不影响通过判定）。"""
    if not stats:
        print("  (没有匹配到 face_* 张量)")
        return
    for g in sorted(stats):
        s = stats[g]
        if s["missing"]:
            print(f"  {g:22s} ⚠️ MISSING {s['missing']}/{s['n']}")
        elif s["max"] > 0:
            print(f"  {g:22s} ✅ 已更新 {s['changed']}/{s['n']}  max|Δ|={s['max']:.3e}")
        else:
            print(f"  {g:22s} ⚠️ Δ=0（这一阶段没训到脸？）")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ref", required=True, help="参考 ckpt（暖启用的，如 snapshot_8）")
    ap.add_argument("--ckpt", required=True, nargs="+", help="待检查 ckpt（可多个）")
    ap.add_argument("--also_face", action="store_true",
                    help="额外报告脸部分支（Stage 2 应当 Δ>0，确认脸确实训了）")
    args = ap.parse_args()

    if not os.path.isfile(args.ref):
        raise SystemExit(f"找不到 ref: {args.ref}")
    ref = load_network(args.ref)
    print(f"ref = {args.ref}  ({len(ref)} tensors)")

    all_ok = True
    for cp in args.ckpt:
        print(f"\n################ {cp} ################")
        if not os.path.isfile(cp):
            print("  !!! 文件不存在")
            all_ok = False
            continue
        ck = load_network(cp)
        print(f"  ({len(ck)} tensors)")
        print(" HAND（期望冻结 Δ=0）:")
        ok = report_frozen(compare(ref, ck, HAND_PREFIXES))
        print("  → ✅ 手部已冻结" if ok else "  → ❌ 手部变化/缺失 —— 冻结或保存有问题")
        all_ok = all_ok and ok
        if args.also_face:
            print(" FACE（期望已训练 Δ>0）:")
            report_trained(compare(ref, ck, FACE_PREFIXES))

    print("\n" + ("✅ 所有 ckpt 手部均冻结" if all_ok else "❌ 存在手部未冻结/缺失的 ckpt"))
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
