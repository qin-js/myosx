#!/usr/bin/env python3
"""Filter UBody annotations to the frames that actually exist on disk.

This is meant for the subsampled-image workflow. The original UBody annotation
JSON files still describe the full 30fps videos, while images/ may contain only
every Nth frame. Filtering once offline shrinks the annotation files and avoids
scanning annotations for frames that will be skipped anyway.

Default output layout:

    <output_root>/<scene>/keypoint_annotation.json
    <output_root>/<scene>/smplx_annotation.json

Optionally add --write_pkl to also write pickle caches next to those JSON files,
which the UBody loader will pick up automatically.
Use --pkl_only to write only keypoint_annotation.pkl / smplx_annotation.pkl.
"""

import argparse
import json
import os
import pickle
from pathlib import Path


ANNOTATION_CACHE_VERSION = 1
KEYPOINT_FILE = "keypoint_annotation.json"
SMPLX_FILE = "smplx_annotation.json"


def _default_data_root():
    return Path(__file__).resolve().parents[2] / "dataset" / "UBody"


def _parse_scenes(raw):
    if not raw:
        return None
    return [scene.strip() for scene in raw.split(",") if scene.strip()]


def _relative_existing_images(scene_image_root):
    paths = []
    if not scene_image_root.is_dir():
        return set()
    for path in scene_image_root.rglob("*.png"):
        if path.name.startswith("."):
            continue
        paths.append(path.relative_to(scene_image_root).as_posix())
    return set(paths)


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _save_json_atomic(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(".%s.tmp.%d" % (path.name, os.getpid()))
    try:
        with open(tmp_path, "w") as f:
            json.dump(obj, f, separators=(",", ":"), ensure_ascii=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


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


def _cache_payload(data, json_path, allow_missing_json=False):
    meta = {"path": str(json_path.resolve())}
    if json_path.is_file():
        stat = json_path.stat()
        meta.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {
        "_ubody_annotation_cache_version": ANNOTATION_CACHE_VERSION,
        "json": meta,
        "allow_missing_json": bool(allow_missing_json),
        "data": data,
    }


def _normal_file_name(file_name):
    return file_name[1:] if file_name.startswith("/") else file_name


def _filter_scene(scene, data_root, output_root, write_pkl=False, pkl_only=False, overwrite=False):
    scene_image_root = data_root / "images" / scene
    scene_annot_root = data_root / "annotations" / scene
    scene_out_root = output_root / scene

    keypoint_path = scene_annot_root / KEYPOINT_FILE
    smplx_path = scene_annot_root / SMPLX_FILE
    out_keypoint_path = scene_out_root / KEYPOINT_FILE
    out_smplx_path = scene_out_root / SMPLX_FILE

    if not keypoint_path.is_file():
        raise FileNotFoundError("missing keypoint annotation: %s" % keypoint_path)
    if not smplx_path.is_file():
        raise FileNotFoundError("missing smplx annotation: %s" % smplx_path)
    expected_outputs = [out_keypoint_path.with_suffix(".pkl"), out_smplx_path.with_suffix(".pkl")] if pkl_only \
        else [out_keypoint_path, out_smplx_path]
    if all(path.exists() for path in expected_outputs) and not overwrite:
        print("skip existing scene: %s" % scene)
        return {"scene": scene, "status": "skipped"}

    existing_rel = _relative_existing_images(scene_image_root)
    if not existing_rel:
        print("scene has no extracted images: %s" % scene_image_root)

    print("read keypoint json: %s" % keypoint_path)
    coco = _load_json(keypoint_path)

    old_images = coco.get("images", [])
    kept_images = []
    kept_image_ids = set()
    for img in old_images:
        file_name = _normal_file_name(img["file_name"])
        if file_name in existing_rel:
            kept_images.append(img)
            kept_image_ids.add(img["id"])

    old_annotations = coco.get("annotations", [])
    kept_annotations = []
    for orig_order, ann in enumerate(old_annotations, 1):
        if ann.get("image_id") not in kept_image_ids:
            continue
        ann_out = dict(ann)
        ann_out["_ubody_orig_ann_order"] = orig_order
        kept_annotations.append(ann_out)
    kept_ann_ids = {str(ann["id"]) for ann in kept_annotations}

    filtered_coco = dict(coco)
    filtered_coco["images"] = kept_images
    filtered_coco["annotations"] = kept_annotations

    print("read smplx json: %s" % smplx_path)
    smplx = _load_json(smplx_path)
    filtered_smplx = {aid: value for aid, value in smplx.items() if aid in kept_ann_ids}

    if not pkl_only:
        _save_json_atomic(filtered_coco, out_keypoint_path)
        _save_json_atomic(filtered_smplx, out_smplx_path)

    if write_pkl or pkl_only:
        _save_pickle_atomic(_cache_payload(filtered_coco, out_keypoint_path,
                                          allow_missing_json=pkl_only),
                            out_keypoint_path.with_suffix(".pkl"))
        _save_pickle_atomic(_cache_payload(filtered_smplx, out_smplx_path,
                                          allow_missing_json=pkl_only),
                            out_smplx_path.with_suffix(".pkl"))

    stats = {
        "scene": scene,
        "status": "written",
        "existing_images": len(existing_rel),
        "images_in": len(old_images),
        "images_out": len(kept_images),
        "annotations_in": len(old_annotations),
        "annotations_out": len(kept_annotations),
        "smplx_in": len(smplx),
        "smplx_out": len(filtered_smplx),
        "output": str(scene_out_root),
    }
    print(
        "%s: images %d/%d, annotations %d/%d, smplx %d/%d" % (
            scene,
            stats["images_out"], stats["images_in"],
            stats["annotations_out"], stats["annotations_in"],
            stats["smplx_out"], stats["smplx_in"],
        )
    )
    if stats["annotations_out"] != stats["smplx_out"]:
        print("warning: annotation/smplx count differs after filtering for scene %s" % scene)
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Filter UBody annotations to images that exist on disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root", type=Path, default=_default_data_root(),
                        help="UBody root containing images/ and annotations/.")
    parser.add_argument("--output_root", type=Path, required=True,
                        help="Where filtered scene annotation folders are written.")
    parser.add_argument("--scenes", type=str, default="",
                        help="Comma-separated scene names. Default: all annotation scene folders.")
    parser.add_argument("--write_pkl", action="store_true",
                        help="Also write keypoint_annotation.pkl and smplx_annotation.pkl caches.")
    parser.add_argument("--pkl_only", action="store_true",
                        help="Write only keypoint_annotation.pkl and smplx_annotation.pkl, no filtered JSON.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing filtered scene outputs.")
    args = parser.parse_args()

    data_root = args.data_root
    annotation_root = data_root / "annotations"
    if not annotation_root.is_dir():
        raise FileNotFoundError("UBody annotation root not found: %s" % annotation_root)

    scenes = _parse_scenes(args.scenes)
    if scenes is None:
        scenes = sorted(path.name for path in annotation_root.iterdir() if path.is_dir())

    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for scene in scenes:
        results.append(_filter_scene(scene, data_root, args.output_root,
                                     write_pkl=args.write_pkl,
                                     pkl_only=args.pkl_only,
                                     overwrite=args.overwrite))

    written = [r for r in results if r.get("status") == "written"]
    if written:
        total_images_in = sum(r["images_in"] for r in written)
        total_images_out = sum(r["images_out"] for r in written)
        total_ann_in = sum(r["annotations_in"] for r in written)
        total_ann_out = sum(r["annotations_out"] for r in written)
        total_smplx_in = sum(r["smplx_in"] for r in written)
        total_smplx_out = sum(r["smplx_out"] for r in written)
        print("=" * 60)
        print("written scenes: %d" % len(written))
        print("images:      %d -> %d" % (total_images_in, total_images_out))
        print("annotations: %d -> %d" % (total_ann_in, total_ann_out))
        print("smplx:       %d -> %d" % (total_smplx_in, total_smplx_out))
        print("use with: export UBODY_ANNOTATION_DIR=%s" % args.output_root)


if __name__ == "__main__":
    main()
