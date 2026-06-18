import argparse
import bisect
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def setup_project(project_root):
    project_root = Path(project_root).resolve()
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "main"))

    from config import cfg
    from common.utils.human_models import smpl_x
    from common.utils.smplx import smplx
    from common.utils.vis import render_mesh

    return cfg, smpl_x, smplx, render_mesh


def normalize_gender(gender):
    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")
    gender = str(gender).lower()
    if gender in ("m", "male"):
        return "male"
    if gender in ("f", "female"):
        return "female"
    return "neutral"


def is_checkpoint_file(path):
    return Path(path).name.endswith(".checkpoint.pkl")


def expand_label_files(label_dir, pattern):
    label_dir = Path(label_dir)
    if pattern == "auto":
        pkl_files = sorted(path for path in label_dir.glob("*.pkl") if not is_checkpoint_file(path))
        if pkl_files:
            return pkl_files
        return sorted(label_dir.glob("*.npz"))

    files = []
    for item in pattern.split(","):
        item = item.strip()
        if item:
            files.extend(label_dir.glob(item))
    return sorted(path for path in set(files) if not is_checkpoint_file(path))


def count_label_file(path):
    path = Path(path)
    if path.suffix.lower() == ".pkl":
        with open(path, "rb") as f:
            records = pickle.load(f)
        if not isinstance(records, list):
            raise ValueError("%s is not a final BEDLAM record-list .pkl file" % path)
        return len(records)

    with np.load(path, allow_pickle=True) as data:
        return len(data["imgname"])


def load_label_files(label_dir, pattern):
    files = expand_label_files(label_dir, pattern)
    if not files:
        raise FileNotFoundError("no label files found in %s with pattern %s" % (label_dir, pattern))
    counts = [count_label_file(path) for path in files]
    return files, counts


def load_label_data(path):
    path = Path(path)
    if path.suffix.lower() == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    return np.load(path, allow_pickle=True)


def close_label_data(data):
    if hasattr(data, "close"):
        data.close()


def rel_parts(path):
    return [part for part in str(path).replace("\\", "/").split("/") if part and part != "."]


def resolve_image_path(img_root, img_rel, scene_name=None):
    img_rel = str(img_rel)
    if os.path.isabs(img_rel):
        return Path(img_rel)

    parts = rel_parts(img_rel)
    img_root = Path(img_root)
    direct_path = img_root.joinpath(*parts)
    candidates = []
    if scene_name is not None:
        if len(parts) >= 2 and parts[0] == scene_name:
            candidates.append(direct_path)
        elif parts and parts[0] == "png":
            candidates.append(img_root.joinpath(scene_name, *parts))
        else:
            candidates.append(img_root.joinpath(scene_name, "png", *parts))
    candidates.append(direct_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def scene_from_ann_id(record):
    ann_id = record.get("ann_id")
    if ann_id is None:
        return None
    return Path(str(ann_id).split(":", 1)[0]).stem


def fill_pose_slice(pose, dst_slice, value):
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    size = dst_slice.stop - dst_slice.start
    pose[dst_slice.start : dst_slice.start + min(size, value.shape[0])] = value[:size]


def pose_from_smplx_param(smplx_param):
    if "pose_cam" in smplx_param:
        return np.asarray(smplx_param["pose_cam"], dtype=np.float32).reshape(-1)
    if "pose" in smplx_param:
        return np.asarray(smplx_param["pose"], dtype=np.float32).reshape(-1)

    pose = np.zeros((165,), dtype=np.float32)
    fill_pose_slice(pose, slice(0, 3), smplx_param.get("root_pose", np.zeros(3)))
    fill_pose_slice(pose, slice(3, 66), smplx_param.get("body_pose", np.zeros(63)))
    fill_pose_slice(pose, slice(66, 69), smplx_param.get("jaw_pose", np.zeros(3)))
    fill_pose_slice(pose, slice(69, 72), smplx_param.get("leye_pose", np.zeros(3)))
    fill_pose_slice(pose, slice(72, 75), smplx_param.get("reye_pose", np.zeros(3)))
    fill_pose_slice(pose, slice(75, 120), smplx_param.get("lhand_pose", np.zeros(45)))
    fill_pose_slice(pose, slice(120, 165), smplx_param.get("rhand_pose", np.zeros(45)))
    return pose


def cam_int_from_cam_param(cam_param):
    focal = np.asarray(cam_param["focal"], dtype=np.float32).reshape(2)
    princpt = np.asarray(cam_param["princpt"], dtype=np.float32).reshape(2)
    return np.array([[focal[0], 0, princpt[0]], [0, focal[1], princpt[1]], [0, 0, 1]], dtype=np.float32)


def cam_trans_from_companion_npz(label_path, idx):
    npz_path = Path(label_path).with_suffix(".npz")
    if not npz_path.is_file():
        return None
    with np.load(npz_path, allow_pickle=True) as data:
        if "cam_ext" not in data.files or idx >= len(data["cam_ext"]):
            return None
        cam_ext = np.asarray(data["cam_ext"][idx], dtype=np.float32).reshape(4, 4)
        return cam_ext[:3, 3]


def cam_trans_from_record(record, label_path=None, idx=None):
    if "cam_ext" in record:
        cam_ext = np.asarray(record["cam_ext"], dtype=np.float32).reshape(4, 4)
        return cam_ext[:3, 3]
    cam_param = record.get("cam_param", {})
    if "t" in cam_param:
        return np.asarray(cam_param["t"], dtype=np.float32).reshape(3)
    if label_path is not None and idx is not None:
        cam_trans = cam_trans_from_companion_npz(label_path, idx)
        if cam_trans is not None:
            return cam_trans
    return np.zeros((3,), dtype=np.float32)


def sample_from_npz(data, idx, label_path, img_root):
    scene_name = Path(label_path).stem
    img_rel = str(data["imgname"][idx])
    cam_ext = data["cam_ext"][idx].astype(np.float32) if "cam_ext" in data.files else np.eye(4, dtype=np.float32)
    return {
        "img_rel": img_rel,
        "img_path": resolve_image_path(img_root, img_rel, scene_name),
        "pose": np.asarray(data["pose_cam"][idx], dtype=np.float32).reshape(-1),
        "shape": np.asarray(data["shape"][idx], dtype=np.float32).reshape(-1),
        "trans": np.asarray(data["trans_cam"][idx], dtype=np.float32).reshape(3),
        "cam_int": np.asarray(data["cam_int"][idx], dtype=np.float32).reshape(3, 3),
        "cam_trans": cam_ext[:3, 3],
        "gender": data["gender"][idx] if "gender" in data.files else "neutral",
    }


def sample_from_companion_npz(label_path, idx, img_root):
    npz_path = Path(label_path).with_suffix(".npz")
    if not npz_path.is_file():
        return None
    with np.load(npz_path, allow_pickle=True) as data:
        required = {"imgname", "pose_cam", "shape", "trans_cam", "cam_int"}
        if not required.issubset(set(data.files)) or idx >= len(data["imgname"]):
            return None
        return sample_from_npz(data, idx, npz_path, img_root)


def sample_from_pkl(records, idx, label_path, img_root, prefer_companion_npz=False):
    if prefer_companion_npz:
        companion_sample = sample_from_companion_npz(label_path, idx, img_root)
        if companion_sample is not None:
            companion_sample["source"] = "companion_npz"
            return companion_sample

    record = records[idx]
    smplx_param = record["smplx_param"]
    scene_name = scene_from_ann_id(record) or Path(label_path).stem
    img_rel = str(record["img_path"])
    return {
        "img_rel": img_rel,
        "img_path": resolve_image_path(img_root, img_rel, scene_name),
        "pose": pose_from_smplx_param(smplx_param),
        "shape": np.asarray(smplx_param["shape"], dtype=np.float32).reshape(-1),
        "trans": np.asarray(smplx_param["trans"], dtype=np.float32).reshape(3),
        "cam_int": cam_int_from_cam_param(record["cam_param"]),
        "cam_trans": cam_trans_from_record(record, label_path, idx),
        "gender": smplx_param.get("gender", "neutral"),
        "source": "pkl",
    }


def get_sample(data, idx, label_path, img_root, prefer_companion_npz=False):
    if Path(label_path).suffix.lower() == ".pkl":
        return sample_from_pkl(data, idx, label_path, img_root, prefer_companion_npz)
    sample = sample_from_npz(data, idx, label_path, img_root)
    sample["source"] = "npz"
    return sample


def sample_positions(counts, num_samples, random_sample, seed):
    total = sum(counts)
    if total == 0:
        raise RuntimeError("processed labels contain no samples")
    num_samples = min(num_samples, total)
    if random_sample:
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(total, size=num_samples, replace=False).tolist())
    if num_samples == 1:
        return [0]
    return np.linspace(0, total - 1, num_samples, dtype=np.int64).tolist()


def map_global_position(pos, cumulative):
    file_idx = bisect.bisect_right(cumulative, pos)
    prev = cumulative[file_idx - 1] if file_idx > 0 else 0
    return file_idx, pos - prev


def create_flat_layers(cfg, smplx, num_betas):
    layer_arg = {
        "create_global_orient": False,
        "create_body_pose": False,
        "create_left_hand_pose": False,
        "create_right_hand_pose": False,
        "create_jaw_pose": False,
        "create_leye_pose": False,
        "create_reye_pose": False,
        "create_betas": False,
        "create_expression": False,
        "create_transl": False,
    }
    return {
        gender: smplx.create(
            cfg.human_model_path,
            "smplx",
            gender=gender,
            ext="npz",
            num_betas=num_betas,
            flat_hand_mean=True,
            use_pca=False,
            use_face_contour=True,
            **layer_arg,
        ).eval()
        for gender in ("neutral", "male", "female")
    }


def shape_for_layer(shape, num_betas):
    shape = np.asarray(shape, dtype=np.float32).reshape(-1)[:num_betas]
    if shape.shape[0] < num_betas:
        shape = np.pad(shape, (0, num_betas - shape.shape[0]))
    return shape


def split_pose_tensors(pose):
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    full = np.zeros((165,), dtype=np.float32)
    full[: min(165, pose.shape[0])] = pose[:165]
    return {
        "root_pose": torch.from_numpy(full[:3]).float().view(1, 3),
        "body_pose": torch.from_numpy(full[3:66]).float().view(1, -1),
        "jaw_pose": torch.from_numpy(full[66:69]).float().view(1, 3),
        "leye_pose": torch.from_numpy(full[69:72]).float().view(1, 3),
        "reye_pose": torch.from_numpy(full[72:75]).float().view(1, 3),
        "left_hand_pose": torch.from_numpy(full[75:120]).float().view(1, -1),
        "right_hand_pose": torch.from_numpy(full[120:165]).float().view(1, -1),
    }


def forward_layer(layer, pose, shape, trans, expression_dim=10):
    pose_parts = split_pose_tensors(pose)
    betas = torch.from_numpy(shape_for_layer(shape, layer.num_betas)).float().view(1, -1)
    transl = torch.from_numpy(np.asarray(trans, dtype=np.float32).reshape(3)).float().view(1, 3)
    expression = torch.zeros((1, expression_dim), dtype=torch.float32)
    with torch.no_grad():
        output = layer(
            betas=betas,
            global_orient=pose_parts["root_pose"],
            body_pose=pose_parts["body_pose"],
            left_hand_pose=pose_parts["left_hand_pose"],
            right_hand_pose=pose_parts["right_hand_pose"],
            jaw_pose=pose_parts["jaw_pose"],
            leye_pose=pose_parts["leye_pose"],
            reye_pose=pose_parts["reye_pose"],
            expression=expression,
        )
    return output.vertices[0].detach().cpu().numpy(), output.joints[0].detach().cpu().numpy()


def get_hand_means(default_layer):
    pose_mean = default_layer.pose_mean.detach().cpu().numpy().reshape(-1)
    return pose_mean[75:120].astype(np.float32), pose_mean[120:165].astype(np.float32)


def convert_flat_to_default_pose(pose, default_layer):
    pose = np.asarray(pose, dtype=np.float32).reshape(-1).copy()
    if pose.shape[0] < 165:
        pose = np.pad(pose, (0, 165 - pose.shape[0]))
    lhand_mean, rhand_mean = get_hand_means(default_layer)
    pose[75:120] -= lhand_mean
    pose[120:165] -= rhand_mean
    return pose


def point_error(a, b):
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    dist = np.sqrt(np.sum(diff * diff, axis=1))
    return {
        "max": float(dist.max()),
        "mean": float(dist.mean()),
        "rms": float(np.sqrt(np.mean(dist * dist))),
    }


def render_mesh_img(render_mesh, faces, img, vertices_cam, cam_int):
    cam_param = {
        "focal": np.array([cam_int[0, 0], cam_int[1, 1]], dtype=np.float32),
        "princpt": np.array([cam_int[0, 2], cam_int[1, 2]], dtype=np.float32),
    }
    return render_mesh(img.copy(), vertices_cam, faces, cam_param)


def put_label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 520), 34), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def resize_panel(img, max_width):
    if max_width <= 0 or img.shape[1] <= max_width:
        return img
    scale = max_width / img.shape[1]
    return cv2.resize(img, (max_width, int(round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def make_diff_img(flat_img, converted_img):
    diff = cv2.absdiff(flat_img, converted_img)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    heat = cv2.applyColorMap(np.clip(gray * 8, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    return heat


def visualize_comparison(args, render_mesh, faces, sample, flat_vertices, converted_vertices, out_path):
    img = cv2.imread(str(sample["img_path"]), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise FileNotFoundError(sample["img_path"])

    cam_int = np.asarray(sample["cam_int"], dtype=np.float32).reshape(3, 3)
    cam_trans = np.asarray(sample["cam_trans"], dtype=np.float32).reshape(1, 3)
    flat_img = render_mesh_img(render_mesh, faces, img, flat_vertices + cam_trans, cam_int)
    converted_img = render_mesh_img(render_mesh, faces, img, converted_vertices + cam_trans, cam_int)
    diff_img = make_diff_img(flat_img, converted_img)

    panels = [
        put_label(resize_panel(img, args.max_panel_width), "original"),
        put_label(resize_panel(flat_img, args.max_panel_width), "flat_hand_mean=True"),
        put_label(resize_panel(converted_img, args.max_panel_width), "converted -> myosx default"),
        put_label(resize_panel(diff_img, args.max_panel_width), "abs diff x8"),
    ]
    min_h = min(panel.shape[0] for panel in panels)
    panel = np.concatenate([panel[:min_h] for panel in panels], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel, [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])


def output_scene_name(label_path, img_rel):
    parts = rel_parts(img_rel)
    if len(parts) >= 3 and parts[1] == "png":
        return parts[0]
    return Path(label_path).stem


def verify_sample(args, smpl_x, flat_layers, render_mesh, sample, idx, label_path, out_dir):
    gender = normalize_gender(sample["gender"])
    if gender not in flat_layers:
        gender = "neutral"
    if gender not in smpl_x.layer:
        gender = "neutral"

    pose_flat = np.asarray(sample["pose"], dtype=np.float32).reshape(-1)
    shape = np.asarray(sample["shape"], dtype=np.float32).reshape(-1)
    trans = np.asarray(sample["trans"], dtype=np.float32).reshape(3)

    flat_layer = flat_layers[gender]
    default_layer = smpl_x.layer[gender]
    converted_pose = convert_flat_to_default_pose(pose_flat, default_layer)

    flat_vertices, flat_joints = forward_layer(flat_layer, pose_flat, shape, trans, expression_dim=smpl_x.expr_code_dim)
    converted_vertices, converted_joints = forward_layer(
        default_layer, converted_pose, shape, trans, expression_dim=smpl_x.expr_code_dim
    )

    vertex_error = point_error(flat_vertices, converted_vertices)
    joint_error = point_error(flat_joints, converted_joints)

    scene = output_scene_name(label_path, sample["img_rel"])
    stem = Path(sample["img_rel"]).stem
    out_path = Path(out_dir) / ("%s_%s_%06d_flat_hand_verify.jpg" % (scene, stem, idx))
    if args.save_vis:
        visualize_comparison(args, render_mesh, smpl_x.face, sample, flat_vertices, converted_vertices, out_path)

    return {
        "label_file": str(label_path),
        "index": int(idx),
        "img_rel": str(sample["img_rel"]),
        "img_path": str(sample["img_path"]),
        "source": sample.get("source", Path(label_path).suffix.lower().lstrip(".")),
        "gender": gender,
        "flat_layer_num_betas": int(flat_layer.num_betas),
        "default_layer_num_betas": int(default_layer.num_betas),
        "vertex_error": vertex_error,
        "joint_error": joint_error,
        "vis_path": str(out_path) if args.save_vis else None,
    }


def summarize(results):
    vertex_max = [item["vertex_error"]["max"] for item in results]
    vertex_mean = [item["vertex_error"]["mean"] for item in results]
    joint_max = [item["joint_error"]["max"] for item in results]
    joint_mean = [item["joint_error"]["mean"] for item in results]
    return {
        "num_samples": len(results),
        "vertex_max_abs_max": float(max(vertex_max)) if vertex_max else None,
        "vertex_mean_abs_max": float(max(vertex_mean)) if vertex_mean else None,
        "joint_max_abs_max": float(max(joint_max)) if joint_max else None,
        "joint_mean_abs_max": float(max(joint_mean)) if joint_mean else None,
    }


def main():
    default_project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Verify BEDLAM flat_hand_mean=True hand poses can be converted to myosx default flat_hand_mean=False by subtracting hand means."
    )
    parser.add_argument("--project-root", type=str, default=str(default_project_root))
    parser.add_argument("--img-root", type=str, default="/workspace/BEDLAM_Dataset")
    parser.add_argument("--label-dir", type=str, default="/workspace/BEDLAM_Dataset/processed_labels")
    parser.add_argument("--output-dir", type=str, default="/workspace/BEDLAM_Dataset/flat_hand_verify")
    parser.add_argument("--pattern", type=str, default="auto", help="'auto', '*.npz', '*.pkl', or comma-separated patterns")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--flat-num-betas", type=int, default=10, help="use same beta dim on both layers to isolate flat_hand_mean")
    parser.add_argument("--prefer-companion-npz", action="store_true", help="when input is pkl, read same-stem npz instead")
    parser.add_argument("--save-vis", action="store_true", default=True)
    parser.add_argument("--no-save-vis", action="store_false", dest="save_vis")
    parser.add_argument("--max-panel-width", type=int, default=420)
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    cfg, smpl_x, smplx, render_mesh = setup_project(args.project_root)
    flat_layers = create_flat_layers(cfg, smplx, args.flat_num_betas)

    files, counts = load_label_files(args.label_dir, args.pattern)
    cumulative = np.cumsum(counts).tolist()
    positions = sample_positions(counts, args.num_samples, args.random, args.seed)

    grouped = {}
    for pos in positions:
        file_idx, local_idx = map_global_position(pos, cumulative)
        grouped.setdefault(files[file_idx], []).append(local_idx)

    results = []
    for label_path, indices in grouped.items():
        data = load_label_data(label_path)
        try:
            for idx in indices:
                sample = get_sample(data, idx, label_path, args.img_root, args.prefer_companion_npz)
                result = verify_sample(args, smpl_x, flat_layers, render_mesh, sample, idx, label_path, args.output_dir)
                results.append(result)
                print(
                    "%s idx=%d gender=%s vertex_max=%.8g joint_max=%.8g"
                    % (
                        Path(label_path).name,
                        idx,
                        result["gender"],
                        result["vertex_error"]["max"],
                        result["joint_error"]["max"],
                    )
                )
        finally:
            close_label_data(data)

    summary = summarize(results)
    failed = False
    if summary["vertex_max_abs_max"] is not None and summary["vertex_max_abs_max"] > args.tolerance:
        failed = True
    if summary["joint_max_abs_max"] is not None and summary["joint_max_abs_max"] > args.tolerance:
        failed = True
    summary["tolerance"] = args.tolerance
    summary["passed"] = not failed

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "flat_hand_verify_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "samples": results}, f, indent=2, ensure_ascii=False)

    print("summary: %s" % json.dumps(summary, indent=2))
    print("wrote %s" % summary_path)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
