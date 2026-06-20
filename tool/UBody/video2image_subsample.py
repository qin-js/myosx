#!/usr/bin/env python3
"""UBody 视频抽帧（可设 fps / 间隔），磁盘友好且与标注严格对齐。

背景
----
原版 ``tool/UBody/video2image.py`` 用 ``ffmpeg -r 30 ... %06d.png`` 把每个视频
重采样到 **30fps** 后逐帧导出，文件名是 **1-based 帧号** ``000001.png`` …。
UBody 的标注就是按这个 30fps 帧号索引图片的（已核实：每个 30fps 帧都有标注，
帧号连续）。loader ``data/UBody/UBody.py`` 取图时若文件不存在会自动 ``continue``
跳过，所以——**只要每张图仍用它"真实的 30fps 帧号"命名，抽多抽少都不会错位**。

本脚本就是这么做的：只导出 30fps 网格上的一个等间隔子集（每 stride 帧取 1 帧），
但把它重命名回原始 30fps 帧号。因此：
  * 不会出现"低 fps 重新顺序编号 → 图和标注错位/张冠李戴"的灾难；
  * 没被导出的帧，loader 自动跳过，安全降级。

⚠️ 重要：用本脚本抽帧后，请把 ``cfg.train_sample_interval`` 设为 1。
   抽帧本身已经做了稀疏化；loader 默认还按 10 在标注顺序上再抽一次，两个间隔
   相乘/相撞可能把可用样本数砍到接近 0（例如 stride=6 × interval=10 对某些视频
   交集为空）。抽帧负责稀疏，loader 就别再抽了。

磁盘
----
UBody 共 ~1.07M 个 30fps 帧。PNG 无损很大；本脚本支持 ``--format jpg``：把 JPEG
数据写进 **.png 文件名**（loader 用 cv2.imread 按内容解码，扩展名无所谓），体积约
为 PNG 的 1/5~1/10。磁盘紧张时强烈建议 jpg。

用法
----
  # 推荐：3fps（≈复刻 train_sample_interval=10 的有效训练量）、jpg 省盘
  python tool/UBody/video2image_subsample.py --fps 3 --format jpg

  # 先估算体积，不实际抽
  python tool/UBody/video2image_subsample.py --fps 3 --format jpg --dry_run

  # 只处理部分场景，盯着磁盘一批批来
  python tool/UBody/video2image_subsample.py --fps 2 --format jpg --scenes Movie,TVShow

抽完后把 images 软链给 loader 用：
  ln -s /workspace/UBody  /workspace/myosx/dataset/UBody
(只要 dataset/UBody/{images,annotations,splits} 能被 cfg.data_dir 找到即可)
"""
import os
import sys
import json
import glob
import shutil
import argparse
import subprocess
from functools import partial
from multiprocessing import Pool

SRC_FPS = 30  # 必须与原 video2image.py 的 `-r 30` 一致，否则帧号对不上标注


def find_videos(video_root):
    vids = []
    for root, _, files in os.walk(video_root):
        for f in files:
            if f.lower().endswith(".mp4"):
                vids.append(os.path.join(root, f))
    return sorted(vids)


def scene_of(video_path, video_root):
    return os.path.relpath(video_path, video_root).split(os.sep)[0]


def video_to_imgdir(video_path, video_root, image_root):
    """videos/<scene>/<sub>/<name>.mp4 -> images/<scene>/<sub>/<name>/

    与原 video2image.py 的路径映射一致，从而和标注里的 file_name 完全对上。
    """
    rel = os.path.relpath(video_path, video_root)      # scene/sub/name.mp4
    rel_dir = os.path.splitext(rel)[0]                 # scene/sub/name
    return os.path.join(image_root, rel_dir)


def extract_one(video_path, cfg):
    """抽一个视频。返回 (status, video_path, num_frames_written)。"""
    out_dir = video_to_imgdir(video_path, cfg["video_root"], cfg["image_root"])
    done_marker = os.path.join(out_dir, ".done")

    # 断点续跑：已完成的视频直接跳过
    if (not cfg["overwrite"]) and os.path.isfile(done_marker):
        return ("skip_done", video_path, 0)

    # 软性磁盘保护：空间不足就跳过（多进程下非严格，但够用）
    os.makedirs(cfg["image_root"], exist_ok=True)
    free_gb = shutil.disk_usage(cfg["image_root"]).free / 2 ** 30
    if free_gb < cfg["min_free_gb"]:
        return ("skip_disk", video_path, 0)

    stride = cfg["stride"]
    ext = "jpg" if cfg["format"] == "jpg" else "png"
    tmp_dir = out_dir + ".extracting"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # 先重采样到 30fps（复刻原脚本），再每 stride 帧取 1 帧。
    # select 里的逗号要转义成 \,（否则被当成 filtergraph 分隔符）。
    if stride == 1:
        vf = f"fps={SRC_FPS}"
    else:
        vf = f"fps={SRC_FPS},select=not(mod(n\\,{stride}))"

    args = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", video_path,
        "-vf", vf,
        "-fps_mode", "vfr",        # 只输出被 select 选中的帧，不补帧（保证计数/映射正确）
        "-start_number", "1",
    ]
    if ext == "jpg":
        args += ["-q:v", str(cfg["jpeg_qscale"])]   # 2(最好)~31(最差)
    args += [os.path.join(tmp_dir, f"%06d.{ext}")]

    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return ("error", video_path, 0)

    # 把 ffmpeg 的顺序编号(j=1,2,3,…) 重命名回"原始 30fps 帧号"：
    #   select 保留的是 n=0, stride, 2*stride, …（n 为 0-based）
    #   原 1-based 帧号 = n + 1 = (j-1)*stride + 1
    # 无论内容是 png 还是 jpg，文件名一律用 .png（标注要求 .png）。
    temp_files = sorted(
        glob.glob(os.path.join(tmp_dir, f"*.{ext}")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for p in temp_files:
        j = int(os.path.splitext(os.path.basename(p))[0])
        orig_idx = (j - 1) * stride + 1
        dst = os.path.join(out_dir, f"{orig_idx:06d}.png")
        shutil.move(p, dst)
        count += 1
    shutil.rmtree(tmp_dir, ignore_errors=True)

    with open(done_marker, "w") as f:
        json.dump({"stride": stride, "fps": SRC_FPS / stride,
                   "format": ext, "count": count}, f)
    return ("ok", video_path, count)


def dry_run_estimate(videos, cfg):
    """用标注帧数 / stride 估算抽帧后的张数与体积。"""
    ann_root = cfg["annot_root"]
    per_frame_mb = 0.12 if cfg["format"] == "jpg" else 1.0  # 粗略：jpg~0.12MB, png~1MB
    scenes = sorted({scene_of(v, cfg["video_root"]) for v in videos})
    total_kept = 0
    print(f"{'scene':18s} {'ann_frames':>11s} {'kept(~)':>9s}")
    for sc in scenes:
        p = os.path.join(ann_root, sc, "keypoint_annotation.json")
        if not os.path.isfile(p):
            print(f"{sc:18s} {'(no ann)':>11s}")
            continue
        n = len(json.load(open(p))["images"])
        kept = n // cfg["stride"]
        total_kept += kept
        print(f"{sc:18s} {n:>11d} {kept:>9d}")
    print("-" * 42)
    print(f"stride={cfg['stride']}  effective_fps={SRC_FPS / cfg['stride']:.2f}  "
          f"format={cfg['format']}")
    print(f"≈ {total_kept} 张, 估算体积 ≈ {total_kept * per_frame_mb / 1024:.1f} GB "
          f"(每张按 {per_frame_mb}MB 估)")
    free_gb = shutil.disk_usage(cfg["image_root"] if os.path.isdir(cfg["image_root"])
                                else os.path.dirname(cfg["video_root"])).free / 2 ** 30
    print(f"当前可用磁盘 ≈ {free_gb:.0f} GB")


def main():
    ap = argparse.ArgumentParser(
        description="UBody 视频按 fps/间隔抽帧（磁盘友好、与标注严格对齐）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--video_root", default="/workspace/UBody/videos",
                    help="存放 .mp4 的根目录")
    ap.add_argument("--image_root", default=None,
                    help="图片输出根目录；默认把 video_root 里的 /videos 换成 /images")
    ap.add_argument("--annot_root", default=None,
                    help="标注根目录（仅 --dry_run 估算用）；默认 video_root 同级的 annotations")
    ap.add_argument("--fps", type=float, default=3.0,
                    help="目标 fps；stride=round(30/fps)。30fps 即全量")
    ap.add_argument("--stride", type=int, default=0,
                    help="直接指定抽帧步长（每 stride 帧取 1 帧）；>0 时覆盖 --fps")
    ap.add_argument("--format", choices=["png", "jpg"], default="png",
                    help="png=无损(大)；jpg=JPEG 内容写进 .png 名(小, 推荐省盘)")
    ap.add_argument("--jpeg_qscale", type=int, default=3,
                    help="jpg 质量(ffmpeg -q:v)，2 最好~31 最差")
    ap.add_argument("--scenes", default="",
                    help="只处理这些场景，逗号分隔（如 Movie,TVShow）；空=全部")
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--min_free_gb", type=float, default=5.0,
                    help="可用磁盘低于此值(GB)时跳过后续视频")
    ap.add_argument("--limit", type=int, default=0,
                    help="最多处理多少个视频（调试用）；0=不限")
    ap.add_argument("--overwrite", action="store_true",
                    help="重抽已完成(.done)的视频")
    ap.add_argument("--dry_run", action="store_true",
                    help="只估算张数/体积，不实际抽帧")
    args = ap.parse_args()

    video_root = os.path.abspath(args.video_root)
    if not os.path.isdir(video_root):
        sys.exit(f"video_root 不存在: {video_root}")

    if args.image_root:
        image_root = os.path.abspath(args.image_root)
    elif os.sep + "videos" in video_root:
        image_root = video_root.replace(os.sep + "videos", os.sep + "images")
    else:
        image_root = video_root + "_images"

    annot_root = os.path.abspath(args.annot_root) if args.annot_root \
        else os.path.join(os.path.dirname(video_root), "annotations")

    stride = args.stride if args.stride > 0 else max(1, round(SRC_FPS / args.fps))

    cfg = {
        "video_root": video_root,
        "image_root": image_root,
        "annot_root": annot_root,
        "stride": stride,
        "format": args.format,
        "jpeg_qscale": args.jpeg_qscale,
        "min_free_gb": args.min_free_gb,
        "overwrite": args.overwrite,
    }

    videos = find_videos(video_root)
    if args.scenes:
        want = {s.strip() for s in args.scenes.split(",") if s.strip()}
        videos = [v for v in videos if scene_of(v, video_root) in want]
    if args.limit > 0:
        videos = videos[:args.limit]

    print(f"video_root = {video_root}")
    print(f"image_root = {image_root}")
    print(f"videos     = {len(videos)}  | stride={stride} "
          f"(effective_fps={SRC_FPS / stride:.2f})  format={args.format}")
    if not videos:
        sys.exit("没找到 .mp4")

    if args.dry_run:
        dry_run_estimate(videos, cfg)
        return

    os.makedirs(image_root, exist_ok=True)
    worker = partial(extract_one, cfg=cfg)
    stats = {"ok": 0, "skip_done": 0, "skip_disk": 0, "error": 0}
    total_frames = 0
    errors = []

    with Pool(processes=args.processes) as pool:
        for i, (status, vpath, n) in enumerate(pool.imap_unordered(worker, videos), 1):
            stats[status] = stats.get(status, 0) + 1
            total_frames += n
            if status == "error":
                errors.append(vpath)
            if i % 50 == 0 or i == len(videos):
                free_gb = shutil.disk_usage(image_root).free / 2 ** 30
                print(f"[{i}/{len(videos)}] ok={stats['ok']} "
                      f"skip_done={stats['skip_done']} skip_disk={stats['skip_disk']} "
                      f"err={stats['error']} | frames={total_frames} "
                      f"| free={free_gb:.0f}GB", flush=True)

    print("=" * 50)
    print(f"完成。视频 ok={stats['ok']} skip_done={stats['skip_done']} "
          f"skip_disk={stats['skip_disk']} error={stats['error']}")
    print(f"共写出帧数 = {total_frames}")
    if stats["skip_disk"]:
        print(f"⚠️ 有 {stats['skip_disk']} 个视频因磁盘不足被跳过"
              f"（可清理后用相同命令断点续跑）")
    if errors:
        print(f"⚠️ {len(errors)} 个视频抽帧失败，前几个：")
        for e in errors[:5]:
            print("   ", e)


if __name__ == "__main__":
    main()
