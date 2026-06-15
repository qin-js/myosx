import argparse
import bisect
import importlib.util
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def import_df_full_body(bedlam_code_root):
    data_processing = Path(bedlam_code_root).resolve() / "data_processing"
    script = data_processing / "df_full_body.py"
    if not script.is_file():
        raise FileNotFoundError(script)

    old_cwd = os.getcwd()
    os.chdir(str(data_processing))
    sys.path.insert(0, str(data_processing))
    try:
        spec = importlib.util.spec_from_file_location("bedlam_df_full_body_for_vis", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
    return module


def setup_project(project_root):
    project_root = Path(project_root).resolve()
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "main"))
    from common.utils.human_models import smpl_x

    return smpl_x


def setup_myosx_bedlam_smplx(project_root):
    project_root = Path(project_root).resolve()
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "main"))

    from config import cfg
    from common.utils.smplx import smplx
    from common.utils.vis import render_mesh

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
    layer = {
        "male": smplx.create(
            cfg.human_model_path,
            "smplx",
            gender="male",
            ext="npz",
            num_betas=10,
            flat_hand_mean=True,
            use_pca=False,
            **layer_arg,
        ).eval(),
        "female": smplx.create(
            cfg.human_model_path,
            "smplx",
            gender="female",
            ext="npz",
            num_betas=10,
            flat_hand_mean=True,
            use_pca=False,
            **layer_arg,
        ).eval(),
        "neutral": smplx.create(
            cfg.human_model_path,
            "smplx",
            gender="neutral",
            ext="npz",
            num_betas=11,
            flat_hand_mean=True,
            use_pca=False,
            **layer_arg,
        ).eval(),
    }
    return {
        "layer": layer,
        "faces": layer["neutral"].faces,
        "render_mesh": render_mesh,
    }


def setup_renderer(args):
    if args.renderer == "myosx":
        return {"type": "myosx", "model": setup_myosx_bedlam_smplx(args.project_root)}
    if args.renderer == "bedlam":
        return {"type": "bedlam", "df": import_df_full_body(args.bedlam_code_root)}
    return {"type": "wireframe", "smpl_x": setup_project(args.project_root)}


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
        if not item:
            continue
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
    pose[dst_slice.start:dst_slice.start + min(size, value.shape[0])] = value[:size]


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
    return np.array(
        [[focal[0], 0, princpt[0]], [0, focal[1], princpt[1]], [0, 0, 1]],
        dtype=np.float32,
    )


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


def load_label_data(path):
    path = Path(path)
    if path.suffix.lower() == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    return np.load(path, allow_pickle=True)


def close_label_data(data):
    if hasattr(data, "close"):
        data.close()


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


def project_points(points_3d, cam_int):
    z = points_3d[:, 2:3]
    valid = z[:, 0] > 1e-6
    proj = np.zeros((points_3d.shape[0], 2), dtype=np.float32)
    proj[valid] = points_3d[valid, :2] / z[valid]
    proj[valid] = np.einsum("ij,nj->ni", cam_int[:2, :2], proj[valid]) + cam_int[:2, 2]
    return proj, valid


def get_smplx_vertices(smpl_x, pose, shape, trans, gender):
    gender = normalize_gender(gender)
    if gender not in smpl_x.layer:
        gender = "neutral"

    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    shape = np.asarray(shape, dtype=np.float32).reshape(-1)[: smpl_x.shape_param_dim]
    if shape.shape[0] < smpl_x.shape_param_dim:
        shape = np.pad(shape, (0, smpl_x.shape_param_dim - shape.shape[0]))
    trans = np.asarray(trans, dtype=np.float32).reshape(3)

    root_pose = torch.FloatTensor(pose[:3]).view(1, 3)
    body_pose = torch.FloatTensor(pose[3:66]).view(1, -1)
    jaw_pose = torch.FloatTensor(pose[66:69]).view(1, 3)
    leye_pose = torch.FloatTensor(pose[69:72]).view(1, 3)
    reye_pose = torch.FloatTensor(pose[72:75]).view(1, 3)
    lhand_pose = torch.FloatTensor(pose[75:120]).view(1, -1)
    rhand_pose = torch.FloatTensor(pose[120:165]).view(1, -1)
    betas = torch.FloatTensor(shape).view(1, -1)
    transl = torch.FloatTensor(trans).view(1, 3)
    expr = torch.zeros((1, smpl_x.expr_code_dim)).float()

    with torch.no_grad():
        output = smpl_x.layer[gender](
            betas=betas,
            body_pose=body_pose,
            global_orient=root_pose,
            transl=transl,
            left_hand_pose=lhand_pose,
            right_hand_pose=rhand_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            expression=expr,
        )
    return output.vertices[0].detach().cpu().numpy()


def get_myosx_bedlam_vertices(model_ctx, pose, shape, trans, gender):
    gender = normalize_gender(gender)
    if gender not in model_ctx["layer"]:
        gender = "neutral"

    layer = model_ctx["layer"][gender]
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    shape = np.asarray(shape, dtype=np.float32).reshape(-1)[: layer.num_betas]
    if shape.shape[0] < layer.num_betas:
        shape = np.pad(shape, (0, layer.num_betas - shape.shape[0]))
    trans = np.asarray(trans, dtype=np.float32).reshape(3)

    root_pose = torch.FloatTensor(pose[:3]).view(1, 3)
    body_pose = torch.FloatTensor(pose[3:66]).view(1, -1)
    jaw_pose = torch.FloatTensor(pose[66:69]).view(1, 3)
    leye_pose = torch.FloatTensor(pose[69:72]).view(1, 3)
    reye_pose = torch.FloatTensor(pose[72:75]).view(1, 3)
    lhand_pose = torch.FloatTensor(pose[75:120]).view(1, -1)
    rhand_pose = torch.FloatTensor(pose[120:165]).view(1, -1)
    betas = torch.FloatTensor(shape).view(1, -1)
    transl = torch.FloatTensor(trans).view(1, 3)
    expr = torch.zeros((1, 10)).float()

    with torch.no_grad():
        output = layer(
            betas=betas,
            body_pose=body_pose,
            global_orient=root_pose,
            transl=transl,
            left_hand_pose=lhand_pose,
            right_hand_pose=rhand_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            expression=expr,
        )
    return output.vertices[0].detach().cpu().numpy()


def get_bedlam_vertices(df, pose, shape, trans, gender):
    gender = normalize_gender(gender)
    shape = np.asarray(shape, dtype=np.float32).reshape(-1)
    if gender == "neutral" and shape.shape[0] < 11:
        shape = np.pad(shape, (0, 11 - shape.shape[0]))
    elif gender in ("male", "female") and shape.shape[0] > 10:
        shape = shape[:10]

    vertices, _ = df.get_smplx_vertices(pose, shape, trans, gender)
    return vertices.detach().cpu().numpy()


def draw_keypoints(img, kpts, radius):
    out = img.copy()
    kpts = np.asarray(kpts)
    if kpts.ndim != 2 or kpts.shape[1] < 2:
        return out
    if kpts.shape[1] >= 3:
        valid = kpts[:, 2] > 0
    else:
        valid = np.ones((kpts.shape[0],), dtype=bool)
    h, w = out.shape[:2]
    for idx, pt in enumerate(kpts[:, :2]):
        x, y = int(round(pt[0])), int(round(pt[1]))
        if not valid[idx] or x < 0 or y < 0 or x >= w or y >= h:
            continue
        color = (0, 255 - (idx * 37) % 180, 80 + (idx * 53) % 175)
        cv2.circle(out, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    return out


def draw_mesh(img, vertices_cam, faces, cam_int, face_stride, vertex_stride):
    out = img.copy()
    pts, valid = project_points(vertices_cam, cam_int)
    h, w = out.shape[:2]

    faces = faces[:: max(1, face_stride)]
    for face in faces:
        if not (valid[face[0]] and valid[face[1]] and valid[face[2]]):
            continue
        poly = np.round(pts[face]).astype(np.int32)
        if ((poly[:, 0] < -w) | (poly[:, 0] > 2 * w) | (poly[:, 1] < -h) | (poly[:, 1] > 2 * h)).all():
            continue
        cv2.polylines(out, [poly.reshape(-1, 1, 2)], True, (40, 220, 255), 1, cv2.LINE_AA)

    for pt in pts[:: max(1, vertex_stride)]:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(out, (x, y), 1, (0, 80, 255), -1, lineType=cv2.LINE_AA)
    return out


def render_bedlam_mesh(df, img, vertices_cam, cam_int):
    h, w = img.shape[:2]
    renderer = df.Renderer(focal_length=cam_int[0][0], img_w=w, img_h=h, faces=df.smplx_model_neutral.faces)
    mesh_rgb = renderer.render_front_view(vertices_cam[None], bg_img_rgb=img[:, :, ::-1].copy())
    return mesh_rgb[:, :, ::-1].copy()


def render_myosx_mesh(model_ctx, img, vertices_cam, cam_int):
    cam_param = {
        "focal": np.array([cam_int[0, 0], cam_int[1, 1]], dtype=np.float32),
        "princpt": np.array([cam_int[0, 2], cam_int[1, 2]], dtype=np.float32),
    }
    return model_ctx["render_mesh"](img.copy(), vertices_cam, model_ctx["faces"], cam_param)


def put_label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 360), 34), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def resize_panel(img, max_width):
    if max_width <= 0 or img.shape[1] <= max_width:
        return img
    scale = max_width / img.shape[1]
    return cv2.resize(img, (max_width, int(round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def make_panel(img, kpt_img, mesh_img, both_img, max_panel_width):
    panels = [
        put_label(resize_panel(img, max_panel_width), "original"),
        put_label(resize_panel(kpt_img, max_panel_width), "2D keypoints"),
        put_label(resize_panel(mesh_img, max_panel_width), "SMPL-X mesh"),
        put_label(resize_panel(both_img, max_panel_width), "keypoints + mesh"),
    ]
    min_h = min(panel.shape[0] for panel in panels)
    panels = [panel[:min_h] for panel in panels]
    return np.concatenate(panels, axis=1)


def sample_from_npz(data, idx, label_path, img_root):
    scene_name = Path(label_path).stem
    img_rel = str(data["imgname"][idx])
    img_path = resolve_image_path(img_root, img_rel, scene_name)

    cam_ext = data["cam_ext"][idx].astype(np.float32) if "cam_ext" in data.files else np.eye(4, dtype=np.float32)
    return {
        "img_rel": img_rel,
        "img_path": img_path,
        "pose": data["pose_cam"][idx],
        "shape": data["shape"][idx],
        "trans": data["trans_cam"][idx],
        "cam_int": data["cam_int"][idx].astype(np.float32),
        "cam_trans": cam_ext[:3, 3],
        "gender": data["gender"][idx] if "gender" in data.files else "neutral",
        "kpts": data["gtkps"][idx] if "gtkps" in data.files else np.zeros((0, 3), dtype=np.float32),
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


def sample_from_pkl(records, idx, label_path, img_root):
    companion_sample = sample_from_companion_npz(label_path, idx, img_root)
    if companion_sample is not None:
        return companion_sample

    record = records[idx]
    smplx_param = record["smplx_param"]
    scene_name = scene_from_ann_id(record) or Path(label_path).stem
    img_rel = str(record["img_path"])
    img_path = resolve_image_path(img_root, img_rel, scene_name)

    return {
        "img_rel": img_rel,
        "img_path": img_path,
        "pose": pose_from_smplx_param(smplx_param),
        "shape": smplx_param["shape"],
        "trans": smplx_param["trans"],
        "cam_int": cam_int_from_cam_param(record["cam_param"]),
        "cam_trans": cam_trans_from_record(record, label_path, idx),
        "gender": smplx_param.get("gender", "neutral"),
        "kpts": record.get("joint_img", np.zeros((0, 3), dtype=np.float32)),
    }


def get_sample(data, idx, label_path, img_root):
    if Path(label_path).suffix.lower() == ".pkl":
        return sample_from_pkl(data, idx, label_path, img_root)
    return sample_from_npz(data, idx, label_path, img_root)


def output_scene_name(label_path, img_rel):
    parts = rel_parts(img_rel)
    if len(parts) >= 3 and parts[1] == "png":
        return parts[0]
    return Path(label_path).stem


def visualize_sample(sample, idx, label_path, out_dir, renderer_ctx, args):
    img_rel = sample["img_rel"]
    img_path = sample["img_path"]
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise FileNotFoundError(img_path)

    pose = sample["pose"]
    shape = sample["shape"]
    trans = sample["trans"]
    cam_int = sample["cam_int"]
    cam_trans = sample["cam_trans"]
    gender = sample["gender"]

    kpts = sample["kpts"]
    kpt_img = draw_keypoints(img, kpts, args.kpt_radius)

    if renderer_ctx["type"] == "myosx":
        model_ctx = renderer_ctx["model"]
        vertices = get_myosx_bedlam_vertices(model_ctx, pose, shape, trans, gender)
        vertices_cam = vertices + cam_trans.reshape(1, 3)
        mesh_img = render_myosx_mesh(model_ctx, img, vertices_cam, cam_int)
        both_img = draw_keypoints(mesh_img, kpts, args.kpt_radius)
    elif renderer_ctx["type"] == "bedlam":
        df = renderer_ctx["df"]
        vertices = get_bedlam_vertices(df, pose, shape, trans, gender)
        vertices_cam = vertices + cam_trans.reshape(1, 3)
        mesh_img = render_bedlam_mesh(df, img, vertices_cam, cam_int)
        both_img = draw_keypoints(mesh_img, kpts, args.kpt_radius)
    else:
        smpl_x = renderer_ctx["smpl_x"]
        vertices = get_smplx_vertices(smpl_x, pose, shape, trans, gender)
        vertices_cam = vertices + cam_trans.reshape(1, 3)
        mesh_img = draw_mesh(img, vertices_cam, smpl_x.face, cam_int, args.face_stride, args.vertex_stride)
        both_img = draw_mesh(kpt_img, vertices_cam, smpl_x.face, cam_int, args.face_stride, args.vertex_stride)

    panel = make_panel(img, kpt_img, mesh_img, both_img, args.max_panel_width)

    scene = output_scene_name(label_path, img_rel)
    stem = Path(img_rel).stem
    out_path = Path(out_dir) / ("%s_%s_%06d.jpg" % (scene, stem, idx))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel, [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])
    return out_path


def main():
    default_project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, default=str(default_project_root))
    parser.add_argument("--bedlam-code-root", type=str, default=os.environ.get("BEDLAM_CODE_ROOT", "/workspace/BEDLAM"))
    parser.add_argument("--img-root", type=str, default="/workspace/BEDLAM_Dataset")
    parser.add_argument("--label-dir", type=str, default="/workspace/BEDLAM_Dataset/processed_labels")
    parser.add_argument("--output-dir", type=str, default="/workspace/BEDLAM_Dataset/vis_check")
    parser.add_argument("--pattern", type=str, default="auto", help="'auto', '*.npz', '*.pkl', or comma-separated patterns")
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--face-stride", type=int, default=8)
    parser.add_argument("--vertex-stride", type=int, default=20)
    parser.add_argument("--kpt-radius", type=int, default=3)
    parser.add_argument("--max-panel-width", type=int, default=520)
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--renderer", choices=["myosx", "bedlam", "wireframe"], default="myosx")
    args = parser.parse_args()

    renderer_ctx = setup_renderer(args)
    files, counts = load_label_files(args.label_dir, args.pattern)
    cumulative = np.cumsum(counts).tolist()
    positions = sample_positions(counts, args.num_samples, args.random, args.seed)

    grouped = {}
    for pos in positions:
        file_idx, local_idx = map_global_position(pos, cumulative)
        grouped.setdefault(files[file_idx], []).append(local_idx)

    saved = []
    for label_path, indices in grouped.items():
        data = load_label_data(label_path)
        try:
            for idx in indices:
                sample = get_sample(data, idx, label_path, args.img_root)
                out_path = visualize_sample(sample, idx, label_path, args.output_dir, renderer_ctx, args)
                saved.append(out_path)
                print("saved %s" % out_path)
        finally:
            close_label_data(data)
    print("done: %d visualization files in %s" % (len(saved), args.output_dir))


if __name__ == "__main__":
    main()
