#!/usr/bin/env python3
"""Build pickle caches for UBody annotation JSON files.

The UBody loader reads keypoint_annotation.json and smplx_annotation.json for
every scene. Pickle caches avoid repeatedly parsing those large JSON files.
By default caches are written next to the JSON files:

    annotations/<scene>/keypoint_annotation.pkl
    annotations/<scene>/smplx_annotation.pkl

If the dataset tree is read-only, pass --cache_dir and set
UBODY_ANNOTATION_CACHE_DIR to the same path when training/testing.
"""

import argparse
import json
import os
import pickle
from pathlib import Path


UBODY_ANNOTATION_CACHE_VERSION = 1
ANNOTATION_FILES = ("keypoint_annotation.json", "smplx_annotation.json")


def _default_data_root():
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "dataset" / "UBody"


def _parse_scenes(raw):
    if not raw:
        return None
    return [scene.strip() for scene in raw.split(",") if scene.strip()]


def _cache_path(json_path, scene, cache_dir):
    if cache_dir:
        return Path(cache_dir) / scene / (json_path.stem + ".pkl")
    return json_path.with_suffix(".pkl")


def _payload(data, json_path):
    stat = json_path.stat()
    return {
        "_ubody_annotation_cache_version": UBODY_ANNOTATION_CACHE_VERSION,
        "json": {
            "path": str(json_path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "data": data,
    }


def _save_pickle_atomic(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(".%s.tmp.%d" % (path.name, os.getpid()))
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _is_newer_than_source(cache_path, json_path):
    return cache_path.is_file() and cache_path.stat().st_mtime_ns >= json_path.stat().st_mtime_ns


def build_cache(json_path, scene, cache_dir, overwrite=False, dry_run=False):
    out_path = _cache_path(json_path, scene, cache_dir)
    if not overwrite and _is_newer_than_source(out_path, json_path):
        print("skip fresh cache: %s" % out_path)
        return "skipped"

    if dry_run:
        print("would read json: %s" % json_path)
        print("would write: %s" % out_path)
        return "dry_run"

    print("read json: %s" % json_path)
    with open(json_path, "r") as f:
        data = json.load(f)
    _save_pickle_atomic(_payload(data, json_path), out_path)
    print("wrote cache: %s" % out_path)
    return "written"


def main():
    parser = argparse.ArgumentParser(description="Cache UBody annotation JSON files as pickle files.")
    parser.add_argument("--data_root", type=Path, default=_default_data_root(),
                        help="UBody root containing annotations/, images/, splits/.")
    parser.add_argument("--cache_dir", type=Path, default=None,
                        help="Optional external cache root. Loader needs UBODY_ANNOTATION_CACHE_DIR set to this.")
    parser.add_argument("--scenes", type=str, default="",
                        help="Comma-separated scene names. Default: all annotation scene folders.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild caches even when the pkl is newer than the JSON.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show planned outputs without reading or writing JSON/pkl files.")
    args = parser.parse_args()

    annotation_root = args.data_root / "annotations"
    if not annotation_root.is_dir():
        raise FileNotFoundError("UBody annotation root not found: %s" % annotation_root)

    scenes = _parse_scenes(args.scenes)
    if scenes is None:
        scenes = sorted(path.name for path in annotation_root.iterdir() if path.is_dir())

    stats = {"written": 0, "skipped": 0, "dry_run": 0, "missing": 0}
    for scene in scenes:
        scene_dir = annotation_root / scene
        if not scene_dir.is_dir():
            print("missing scene annotation dir: %s" % scene_dir)
            stats["missing"] += len(ANNOTATION_FILES)
            continue
        for file_name in ANNOTATION_FILES:
            json_path = scene_dir / file_name
            if not json_path.is_file():
                print("missing json: %s" % json_path)
                stats["missing"] += 1
                continue
            status = build_cache(json_path, scene, args.cache_dir, args.overwrite, args.dry_run)
            stats[status] += 1

    print("done: written=%d skipped=%d dry_run=%d missing=%d" % (
        stats["written"], stats["skipped"], stats["dry_run"], stats["missing"]))


if __name__ == "__main__":
    main()
