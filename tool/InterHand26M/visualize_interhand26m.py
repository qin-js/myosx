import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


RIGHT_HAND_CHAIN = {
    "thumb": [20, 3, 2, 1, 0],
    "index": [20, 7, 6, 5, 4],
    "middle": [20, 11, 10, 9, 8],
    "ring": [20, 15, 14, 13, 12],
    "pinky": [20, 19, 18, 17, 16],
}
LEFT_HAND_CHAIN = {
    name: [idx + 21 for idx in chain] for name, chain in RIGHT_HAND_CHAIN.items()
}
CHAIN_COLORS = {
    "thumb": (40, 180, 255),
    "index": (40, 255, 120),
    "middle": (255, 210, 70),
    "ring": (255, 120, 180),
    "pinky": (190, 110, 255),
}


class SimpleCOCO:
    def __init__(self, annotation_file):
        print("loading annotations into memory...")
        data = load_json(annotation_file)
        print("creating index...")
        self.dataset = data
        self.anns = {ann["id"]: ann for ann in data.get("annotations", [])}
        self.imgs = {img["id"]: img for img in data.get("images", [])}
        print("index created!")

    def loadImgs(self, ids):
        if isinstance(ids, (list, tuple)):
            return [self.imgs[i] for i in ids]
        return [self.imgs[ids]]


def get_coco_class():
    try:
        from pycocotools.coco import COCO
        return COCO
    except ModuleNotFoundError:
        print("pycocotools is not installed; using slower pure-Python COCO fallback.")
        return SimpleCOCO


def setup_project(project_root):
    project_root = Path(project_root).resolve()
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "main"))

    from config import cfg
    from common.utils.smplx import smplx
    from common.utils.vis import render_mesh

    return cfg, smplx, render_mesh


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def cam_param_from_interhand(cameras, capture_id, cam_id):
    capture = cameras[str(capture_id)]
    cam_id = str(cam_id)
    R = np.asarray(capture["camrot"][cam_id], dtype=np.float32).reshape(3, 3)
    campos = np.asarray(capture["campos"][cam_id], dtype=np.float32).reshape(3)
    t = -np.dot(R, campos.reshape(3, 1)).reshape(3)
    focal = np.asarray(capture["focal"][cam_id], dtype=np.float32).reshape(2)
    princpt = np.asarray(capture["princpt"][cam_id], dtype=np.float32).reshape(2)
    return {"R": R, "t": t, "focal": focal, "princpt": princpt}


def world2cam(world_coord, R, t):
    return np.dot(R, world_coord.transpose(1, 0)).transpose(1, 0) + t.reshape(1, 3)


def cam2pixel(cam_coord, focal, princpt):
    z = cam_coord[:, 2:3]
    valid = np.abs(z[:, 0]) > 1e-8
    img_coord = np.zeros((cam_coord.shape[0], 3), dtype=np.float32)
    img_coord[:, 2] = cam_coord[:, 2]
    img_coord[valid, 0] = cam_coord[valid, 0] / z[valid, 0] * focal[0] + princpt[0]
    img_coord[valid, 1] = cam_coord[valid, 1] / z[valid, 0] * focal[1] + princpt[1]
    return img_coord


def project_points(points_3d, focal, princpt):
    points_3d = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3)
    z = points_3d[:, 2:3]
    valid = z[:, 0] > 1e-8
    points_2d = np.zeros((points_3d.shape[0], 2), dtype=np.float32)
    points_2d[valid, 0] = points_3d[valid, 0] / z[valid, 0] * focal[0] + princpt[0]
    points_2d[valid, 1] = points_3d[valid, 1] / z[valid, 0] * focal[1] + princpt[1]
    return points_2d, valid


def build_mano_layers(smplx_module, human_model_path):
    import torch

    layer_arg = {
        "create_global_orient": False,
        "create_hand_pose": False,
        "create_betas": False,
        "create_transl": False,
    }
    layers = {
        "right": smplx_module.create(
            human_model_path,
            "mano",
            is_rhand=True,
            use_pca=False,
            flat_hand_mean=False,
            **layer_arg,
        ).eval(),
        "left": smplx_module.create(
            human_model_path,
            "mano",
            is_rhand=False,
            use_pca=False,
            flat_hand_mean=False,
            **layer_arg,
        ).eval(),
    }

    # MANO left-hand shapedirs bug fix used by Hand4Whole.
    if torch.sum(torch.abs(layers["left"].shapedirs[:, 0, :] - layers["right"].shapedirs[:, 0, :])) < 1:
        print("Fix shapedirs bug of MANO")
        layers["left"].shapedirs[:, 0, :] *= -1
    return layers


def get_mano_vertices(mano_layers, mano_param, cam_param, side, apply_extrinsic=True):
    import torch

    if mano_param is None:
        return None, None

    pose = torch.FloatTensor(np.asarray(mano_param["pose"], dtype=np.float32)).view(-1, 3)
    shape = torch.FloatTensor(np.asarray(mano_param["shape"], dtype=np.float32)).view(1, -1)
    trans = torch.FloatTensor(np.asarray(mano_param["trans"], dtype=np.float32)).view(1, 3)

    if apply_extrinsic and "R" in cam_param:
        R = np.asarray(cam_param["R"], dtype=np.float32).reshape(3, 3)
        root_pose = pose[0].detach().cpu().numpy()
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))
        pose[0] = torch.from_numpy(root_pose).view(3)

    root_pose = pose[0].view(1, 3)
    hand_pose = pose[1:].contiguous().view(1, -1)
    with torch.no_grad():
        output = mano_layers[side](
            global_orient=root_pose,
            hand_pose=hand_pose,
            betas=shape,
            transl=trans,
        )
    vertices = output.vertices[0].detach().cpu().numpy()
    joints = output.joints[0].detach().cpu().numpy()

    if apply_extrinsic and "R" in cam_param and "t" in cam_param:
        R = np.asarray(cam_param["R"], dtype=np.float32).reshape(3, 3)
        t = np.asarray(cam_param["t"], dtype=np.float32).reshape(1, 3) / 1000.0
        root_cam = joints[0:1]
        rotated_root = np.dot(R, root_cam.transpose(1, 0)).transpose(1, 0)
        vertices = vertices - root_cam + rotated_root + t
        joints = joints - root_cam + rotated_root + t

    return vertices.astype(np.float32), mano_layers[side].faces.astype(np.int64)


def draw_interhand_keypoints(img, keypoints, valid, radius=3):
    out = img.copy()
    keypoints = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    valid = np.asarray(valid, dtype=np.float32).reshape(-1) > 0
    h, w = out.shape[:2]

    for chains in (RIGHT_HAND_CHAIN, LEFT_HAND_CHAIN):
        for name, chain in chains.items():
            color = CHAIN_COLORS[name]
            for a, b in zip(chain[:-1], chain[1:]):
                if not (valid[a] and valid[b]):
                    continue
                p1 = tuple(np.round(keypoints[a]).astype(np.int32).tolist())
                p2 = tuple(np.round(keypoints[b]).astype(np.int32).tolist())
                if point_in_loose_image(p1, w, h) and point_in_loose_image(p2, w, h):
                    cv2.line(out, p1, p2, color, 2, cv2.LINE_AA)
            for idx in chain:
                if not valid[idx]:
                    continue
                x, y = np.round(keypoints[idx]).astype(np.int32).tolist()
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(out, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
                    cv2.circle(out, (x, y), radius + 1, (20, 20, 20), 1, lineType=cv2.LINE_AA)
    return out


def point_in_loose_image(point, width, height):
    x, y = point
    return -width <= x <= 2 * width and -height <= y <= 2 * height


def draw_mesh_wireframe(img, vertices, faces, focal, princpt, face_stride=2, vertex_stride=12):
    out = img.copy()
    pts, valid = project_points(vertices, focal, princpt)
    h, w = out.shape[:2]

    for face in faces[:: max(1, face_stride)]:
        if not (valid[face[0]] and valid[face[1]] and valid[face[2]]):
            continue
        poly = np.round(pts[face]).astype(np.int32)
        if ((poly[:, 0] < -w) | (poly[:, 0] > 2 * w) | (poly[:, 1] < -h) | (poly[:, 1] > 2 * h)).all():
            continue
        cv2.polylines(out, [poly.reshape(-1, 1, 2)], True, (0, 220, 255), 1, cv2.LINE_AA)

    for pt, is_valid in zip(pts[:: max(1, vertex_stride)], valid[:: max(1, vertex_stride)]):
        if not is_valid:
            continue
        x, y = np.round(pt).astype(np.int32).tolist()
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(out, (x, y), 1, (0, 90, 255), -1, lineType=cv2.LINE_AA)
    return out


def render_mano_meshes(img, mesh_items, cam_param, render_mesh, args):
    out = img.copy()
    focal = np.asarray(cam_param["focal"], dtype=np.float32)
    princpt = np.asarray(cam_param["princpt"], dtype=np.float32)
    renderer_failed = False

    for vertices, faces in mesh_items:
        if vertices is None:
            continue
        if args.renderer == "wireframe" or renderer_failed:
            out = draw_mesh_wireframe(out, vertices, faces, focal, princpt, args.face_stride, args.vertex_stride)
            continue
        try:
            out = render_mesh(
                out,
                vertices,
                faces,
                {"focal": focal, "princpt": princpt},
            )
        except Exception as exc:
            if args.strict_renderer:
                raise
            renderer_failed = True
            print("renderer failed once, fallback to wireframe: %s" % exc)
            out = draw_mesh_wireframe(out, vertices, faces, focal, princpt, args.face_stride, args.vertex_stride)
    return out


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
        put_label(resize_panel(kpt_img, max_panel_width), "InterHand keypoints"),
        put_label(resize_panel(mesh_img, max_panel_width), "MANO mesh"),
        put_label(resize_panel(both_img, max_panel_width), "keypoints + mesh"),
    ]
    min_h = min(panel.shape[0] for panel in panels)
    panels = [panel[:min_h] for panel in panels]
    return np.concatenate(panels, axis=1)


def aid_list(annot_dir, split, db, use_human_annot):
    aid_path = Path(annot_dir) / split / ("aid_human_annot_%s.txt" % split)
    if use_human_annot and aid_path.is_file():
        with open(aid_path, "r") as f:
            aids = [int(line.strip()) for line in f if line.strip()]
        return [aid for aid in aids if aid in db.anns]
    return sorted(db.anns.keys())


def resolve_img_path(img_root, split, file_name):
    return Path(img_root) / split / file_name


def sample_candidates(aids, seed, max_candidates):
    rng = np.random.default_rng(seed)
    aids = np.asarray(aids, dtype=np.int64)
    rng.shuffle(aids)
    if max_candidates is not None and max_candidates > 0:
        aids = aids[:max_candidates]
    return aids.tolist()


def visualize_one(aid, db, cameras, joints, mano_params, img_root, split, mano_layers, render_mesh, args):
    ann = db.anns[aid]
    img_info = db.loadImgs(ann["image_id"])[0]
    img_path = resolve_img_path(img_root, split, img_info["file_name"])
    if args.skip_missing_images and not img_path.is_file():
        return None

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise FileNotFoundError(str(img_path))

    capture_id = str(img_info["capture"])
    frame_idx = str(img_info["frame_idx"])
    cam_id = str(img_info["camera"])
    cam_param = cam_param_from_interhand(cameras, capture_id, cam_id)

    joint_frame = joints[capture_id][frame_idx]
    joint_valid = np.asarray(joint_frame["joint_valid"], dtype=np.float32).reshape(-1, 1)
    joint_world = np.asarray(joint_frame["world_coord"], dtype=np.float32).reshape(-1, 3)
    joint_cam = world2cam(joint_world, cam_param["R"], cam_param["t"])
    joint_img = cam2pixel(joint_cam, cam_param["focal"], cam_param["princpt"])

    kpt_img = draw_interhand_keypoints(img, joint_img[:, :2], joint_valid, args.kpt_radius)

    mano_frame = mano_params.get(capture_id, {}).get(frame_idx, {})
    mesh_items = []
    for side in ("right", "left"):
        vertices, faces = get_mano_vertices(
            mano_layers,
            mano_frame.get(side),
            cam_param,
            side,
            apply_extrinsic=not args.no_mano_extrinsic,
        )
        if vertices is not None:
            mesh_items.append((vertices, faces))

    mesh_img = render_mano_meshes(img, mesh_items, cam_param, render_mesh, args)
    both_img = draw_interhand_keypoints(mesh_img, joint_img[:, :2], joint_valid, args.kpt_radius)
    panel = make_panel(img, kpt_img, mesh_img, both_img, args.max_panel_width)

    rel_name = Path(img_info["file_name"])
    safe_scene = str(img_info.get("seq_name", rel_name.parent)).replace("\\", "_").replace("/", "_")
    out_name = "%s_aid%s_%s.jpg" % (safe_scene, aid, rel_name.stem)
    out_path = Path(args.output_dir) / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel, [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])
    return out_path


def parse_args():
    default_project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, default=str(default_project_root))
    parser.add_argument("--annot-root", type=str, default=None)
    parser.add_argument("--img-root", type=str, default=None)
    parser.add_argument("--human-model-path", type=str, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=str, default=str(default_project_root / "output" / "interhand26m_vis_check"))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-candidates", type=int, default=10000)
    parser.add_argument("--use-all-annots", action="store_true")
    parser.add_argument("--skip-missing-images", action="store_true", default=True)
    parser.add_argument("--no-skip-missing-images", dest="skip_missing_images", action="store_false")
    parser.add_argument("--renderer", choices=["myosx", "wireframe"], default="myosx")
    parser.add_argument("--strict-renderer", action="store_true")
    parser.add_argument("--no-mano-extrinsic", action="store_true")
    parser.add_argument("--face-stride", type=int, default=2)
    parser.add_argument("--vertex-stride", type=int, default=12)
    parser.add_argument("--kpt-radius", type=int, default=3)
    parser.add_argument("--max-panel-width", type=int, default=520)
    parser.add_argument("--jpg-quality", type=int, default=95)
    return parser.parse_args()


def main():
    args = parse_args()
    COCO = get_coco_class()

    cfg, smplx_module, render_mesh = setup_project(args.project_root)

    annot_root = args.annot_root or os.environ.get("INTERHAND_ANNOS", getattr(cfg, "interhand_annot_path"))
    img_root = args.img_root or os.environ.get("INTERHAND_IMG_DIR", getattr(cfg, "interhand_img_dir"))
    human_model_path = args.human_model_path or getattr(cfg, "human_model_path")

    annot_dir = Path(annot_root) / args.split
    data_json = annot_dir / ("InterHand2.6M_%s_data.json" % args.split)
    camera_json = annot_dir / ("InterHand2.6M_%s_camera.json" % args.split)
    joint_json = annot_dir / ("InterHand2.6M_%s_joint_3d.json" % args.split)
    mano_json = annot_dir / ("InterHand2.6M_%s_MANO_NeuralAnnot.json" % args.split)

    for path in (data_json, camera_json, joint_json, mano_json):
        if not path.is_file():
            raise FileNotFoundError(str(path))

    print("loading annotations from %s" % annot_dir)
    db = COCO(str(data_json))
    cameras = load_json(camera_json)
    joints = load_json(joint_json)
    mano_params = load_json(mano_json)

    print("loading MANO models from %s" % human_model_path)
    mano_layers = build_mano_layers(smplx_module, human_model_path)

    aids = aid_list(annot_root, args.split, db, use_human_annot=not args.use_all_annots)
    candidates = sample_candidates(aids, args.seed, args.max_candidates)

    saved = []
    missing = 0
    for aid in candidates:
        out_path = visualize_one(
            aid,
            db,
            cameras,
            joints,
            mano_params,
            img_root,
            args.split,
            mano_layers,
            render_mesh,
            args,
        )
        if out_path is None:
            missing += 1
            continue
        saved.append(out_path)
        print("saved %s" % out_path)
        if len(saved) >= args.num_samples:
            break

    if not saved:
        raise RuntimeError(
            "No visualization was saved. Checked %d candidates, missing images %d. "
            "Please set --img-root or INTERHAND_IMG_DIR to the downloaded InterHand image root."
            % (len(candidates), missing)
        )
    print("done: saved %d files to %s" % (len(saved), args.output_dir))


if __name__ == "__main__":
    main()
