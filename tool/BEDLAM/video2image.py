import argparse
import concurrent.futures
import os
import shutil
import subprocess
from pathlib import Path

import cv2


def find_videos(root, pattern):
    root = Path(root)
    videos = []
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        mp4_dir = scene_dir / "mp4"
        if not mp4_dir.is_dir():
            continue
        videos.extend(sorted(mp4_dir.glob(pattern)))
    return videos


def output_dir_for(video_path, input_root, output_root):
    video_path = Path(video_path)
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    scene_name = video_path.parent.parent.name
    return output_root / scene_name / "png" / video_path.stem


def already_done(out_dir, ext):
    return out_dir.is_dir() and any(out_dir.glob("*." + ext))


def extract_with_ffmpeg(video_path, out_dir, overwrite, ext, jpg_quality):
    out_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(out_dir / ("%06d." + ext))
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(video_path),
        "-start_number",
        "0",
    ]
    if ext in ("jpg", "jpeg"):
        # ffmpeg quality is 2(best)-31(worst). Keep JPEG visually close to PNG.
        cmd.extend(["-q:v", "2"])
    cmd.append(output_pattern)
    subprocess.run(cmd, check=True)


def extract_with_opencv(video_path, out_dir, overwrite, frame_step, ext, jpg_quality):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("failed to open video: %s" % video_path)

    frame_idx = 0
    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_step == 0:
                out_file = out_dir / ("%06d.%s" % (frame_idx, ext))
                if overwrite or not out_file.exists():
                    params = []
                    if ext in ("jpg", "jpeg"):
                        params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]
                    if not cv2.imwrite(str(out_file), frame, params):
                        raise RuntimeError("failed to write image: %s" % out_file)
                    written += 1
            frame_idx += 1
    finally:
        cap.release()
    return written


def extract_one(args):
    video_path, input_root, output_root, backend, overwrite, skip_existing, frame_step, ext, jpg_quality = args
    out_dir = output_dir_for(video_path, input_root, output_root)
    if skip_existing and already_done(out_dir, ext):
        return "skip", str(video_path), str(out_dir)

    if backend == "ffmpeg":
        if frame_step != 1:
            raise ValueError("ffmpeg backend is only supported for full-frame extraction; use opencv for sampled fps")
        extract_with_ffmpeg(video_path, out_dir, overwrite, ext, jpg_quality)
    elif backend == "opencv":
        extract_with_opencv(video_path, out_dir, overwrite, frame_step, ext, jpg_quality)
    else:
        raise ValueError("unknown backend: %s" % backend)

    return "ok", str(video_path), str(out_dir)


def resolve_backend(backend, frame_step):
    if backend != "auto":
        return backend
    if frame_step != 1:
        return "opencv"
    return "ffmpeg" if shutil.which("ffmpeg") else "opencv"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract BEDLAM videos from <root>/<scene>/mp4/seq*.mp4 to "
            "<output_root>/<scene>/png/<seq_name>/*.<ext>."
        )
    )
    parser.add_argument("--root", type=str, default="/workspace/BEDLAM_Dataset")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--pattern", type=str, default="seq*.mp4")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--source-fps", type=int, default=30)
    parser.add_argument("--ext", choices=["jpg", "jpeg", "png"], default="jpg")
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--backend", choices=["auto", "ffmpeg", "opencv"], default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root or args.root
    if args.fps <= 0 or args.source_fps % args.fps != 0:
        raise ValueError("fps must be a positive divisor of source-fps")
    if not 1 <= args.jpg_quality <= 100:
        raise ValueError("jpg-quality must be in [1, 100]")
    frame_step = args.source_fps // args.fps
    backend = resolve_backend(args.backend, frame_step)
    videos = find_videos(args.root, args.pattern)
    if not videos:
        raise FileNotFoundError(
            "no videos found under %s with pattern */mp4/%s" % (args.root, args.pattern)
        )

    jobs = [
        (
            video,
            args.root,
            output_root,
            backend,
            args.overwrite,
            not args.no_skip_existing,
            frame_step,
            args.ext,
            args.jpg_quality,
        )
        for video in videos
    ]

    print(
        "Found %d videos. backend=%s fps=%d frame_step=%d ext=%s output_root=%s"
        % (len(videos), backend, args.fps, frame_step, args.ext, output_root)
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(extract_one, job) for job in jobs]
        for idx, future in enumerate(concurrent.futures.as_completed(futures), 1):
            status, video_path, out_dir = future.result()
            print("[%d/%d] %s %s -> %s" % (idx, len(futures), status, video_path, out_dir))


if __name__ == "__main__":
    main()
