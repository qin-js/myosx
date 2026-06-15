import argparse
import csv
import importlib.util
import os
import pickle
import sys
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import tqdm


SCALE_FACTOR_BBOX = 1.2


def import_df_full_body(bedlam_code_root):
    data_processing = Path(bedlam_code_root).resolve() / "data_processing"
    script = data_processing / "df_full_body.py"
    if not script.is_file():
        raise FileNotFoundError(script)

    old_cwd = os.getcwd()
    os.chdir(str(data_processing))
    sys.path.insert(0, str(data_processing))
    try:
        spec = importlib.util.spec_from_file_location("bedlam_df_full_body", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
    return module


def resolve_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return torch.device(device)


def patch_df_for_device(df, device):
    df.smplx_model_male = df.smplx_model_male.to(device).eval()
    df.smplx_model_female = df.smplx_model_female.to(device).eval()
    df.smplx_model_neutral = df.smplx_model_neutral.to(device).eval()

    def to_tensor(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def get_smplx_vertices(poses, betas, trans, gender):
        gender = normalize_gender(gender)
        if gender == "male":
            model = df.smplx_model_male
        elif gender == "female":
            model = df.smplx_model_female
        else:
            model = df.smplx_model_neutral

        poses_t = to_tensor(poses).reshape(-1)
        betas_t = to_tensor(betas).reshape(1, -1)
        trans_t = to_tensor(trans).reshape(1, 3)

        with torch.no_grad():
            model_out = model(
                betas=betas_t,
                global_orient=poses_t[:3].reshape(1, 3),
                body_pose=poses_t[3:66].reshape(1, -1),
                left_hand_pose=poses_t[75:120].reshape(1, -1),
                right_hand_pose=poses_t[120:165].reshape(1, -1),
                jaw_pose=poses_t[66:69].reshape(1, 3),
                leye_pose=poses_t[69:72].reshape(1, 3),
                reye_pose=poses_t[72:75].reshape(1, 3),
                transl=trans_t,
            )
        return model_out.vertices[0], model_out.joints[0]

    def project(points, cam_trans, cam_int):
        cam_trans_t = to_tensor(cam_trans).reshape(1, 3)
        cam_int_t = to_tensor(cam_int).reshape(3, 3)
        points_t = points.to(device) if torch.is_tensor(points) else to_tensor(points)
        points_t = points_t + cam_trans_t
        projected_points = points_t / points_t[:, -1:].clamp(min=1e-8)
        projected_points = torch.einsum("ij,kj->ki", cam_int_t, projected_points.float())
        return projected_points.detach().cpu().numpy()

    df.get_smplx_vertices = get_smplx_vertices
    df.project = project
    return df


def downsample_vertices(df, vertices3d):
    mat = df.downsample_mat
    if torch.is_tensor(mat):
        return torch.matmul(mat.to(vertices3d.device), vertices3d)
    return torch.as_tensor(mat.matmul(vertices3d.detach().cpu()), dtype=torch.float32, device=vertices3d.device)


def normalize_gender(gender):
    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")
    gender = str(gender).lower()
    if gender in ("m", "male"):
        return "male"
    if gender in ("f", "female"):
        return "female"
    return "neutral"


def sanitize_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 > x1 and y2 > y1:
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
    return None


def process_bbox(bbox, img_width, img_height, input_img_shape):
    bbox = sanitize_bbox(bbox, img_width, img_height)
    if bbox is None:
        return None

    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.0
    c_y = bbox[1] + h / 2.0
    aspect_ratio = input_img_shape[1] / input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * 1.25
    bbox[3] = h * 1.25
    bbox[0] = c_x - bbox[2] / 2.0
    bbox[1] = c_y - bbox[3] / 2.0
    return bbox.astype(np.float32)


def bbox_from_center_scale(center, scale):
    center = np.asarray(center, dtype=np.float32).reshape(2)
    side = float(scale) * 200.0
    return np.array(
        [center[0] - side / 2.0, center[1] - side / 2.0, side, side],
        dtype=np.float32,
    )


def frame_index_from_name(path, fallback_idx):
    stem = Path(path).stem
    return int(stem) if stem.isdigit() else fallback_idx


def rel_image_path(scene_folder, seq_name, image_path):
    name = Path(image_path).name
    return "%s/png/%s/%s" % (scene_folder, seq_name, name)


def split_pose(pose):
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    return {
        "root_pose": pose[:3],
        "body_pose": pose[3:66],
        "jaw_pose": pose[66:69] if pose.shape[0] >= 69 else np.zeros(3, dtype=np.float32),
        "lhand_pose": pose[75:120] if pose.shape[0] >= 120 else np.zeros(45, dtype=np.float32),
        "rhand_pose": pose[120:165] if pose.shape[0] >= 165 else np.zeros(45, dtype=np.float32),
    }


def make_record(
    ann_id,
    img_rel,
    img_shape,
    bbox,
    joints2d,
    pose_cam,
    beta,
    trans_cam,
    cam_int,
    gender_sub,
):
    shape = np.asarray(beta, dtype=np.float32).reshape(-1)[:10]
    if shape.shape[0] < 10:
        shape = np.pad(shape, (0, 10 - shape.shape[0]))

    smplx_param = split_pose(pose_cam)
    smplx_param.update(
        {
            "shape": shape,
            "expr": np.zeros(10, dtype=np.float32),
            "trans": np.asarray(trans_cam, dtype=np.float32).reshape(3),
            "lhand_valid": True,
            "rhand_valid": True,
            "face_valid": False,
            "gender": normalize_gender(gender_sub),
        }
    )

    cam_int = np.asarray(cam_int, dtype=np.float32).reshape(3, 3)
    return {
        "ann_id": ann_id,
        "img_path": img_rel,
        "img_shape": tuple(int(v) for v in img_shape),
        "bbox": np.asarray(bbox, dtype=np.float32),
        "joint_img": np.asarray(joints2d, dtype=np.float32),
        "smplx_param": smplx_param,
        "cam_param": {
            "focal": np.array([cam_int[0, 0], cam_int[1, 1]], dtype=np.float32),
            "princpt": np.array([cam_int[0, 2], cam_int[1, 2]], dtype=np.float32),
        },
    }


def process_body(
    df,
    image_folder,
    scene_folder,
    seq_name,
    fl,
    start_frame,
    gender_sub,
    smplx_param_orig,
    trans_body,
    body_yaw,
    cam_x,
    cam_y,
    cam_z,
    cam_pitch,
    cam_roll,
    cam_yaw,
    fps,
    sub,
    img_format,
    img_shape,
    input_img_shape,
    rotate_flag,
    out_stem,
    state,
    show_frame_progress=True,
):
    all_images = sorted(glob(os.path.join(image_folder, "*" + img_format)))
    frame_step = 30 // fps
    selected_images = [
        (fallback_idx, image_path)
        for fallback_idx, image_path in enumerate(all_images)
        if frame_index_from_name(image_path, fallback_idx) % frame_step == 0
    ]

    accepted_before = len(state["records"])
    iterator = selected_images
    if show_frame_progress:
        iterator = tqdm.tqdm(
            selected_images,
            desc="frames %s/%s" % (scene_folder, sub),
            leave=False,
            dynamic_ncols=True,
        )

    for fallback_idx, image_path in iterator:
        img_ind = frame_index_from_name(image_path, fallback_idx)

        smplx_param_ind = img_ind + start_frame
        cam_ind = img_ind
        if smplx_param_ind >= smplx_param_orig["poses"].shape[0]:
            break
        if cam_ind >= len(fl):
            continue

        pose = smplx_param_orig["poses"][smplx_param_ind]
        transl = smplx_param_orig["trans"][smplx_param_ind]
        beta = smplx_param_orig["betas"]
        motion_info = smplx_param_orig["motion_info"]
        gender = smplx_param_orig["gender"]

        cam_pitch_ind = -cam_pitch[cam_ind]
        cam_yaw_ind = -cam_yaw[cam_ind]
        cam_roll_ind = -cam_roll[cam_ind] + 90 if rotate_flag else -cam_roll[cam_ind]

        cam_int = df.get_cam_int(fl[cam_ind], df.SENSOR_W, df.SENSOR_H, df.IMG_W / 2.0, df.IMG_H / 2.0)
        _, cam_rotmat_for_trans = df.get_cam_rotmat(body_yaw, cam_pitch_ind, cam_yaw_ind, cam_roll_ind)
        cam_t = [cam_x[cam_ind], cam_y[cam_ind], cam_z[cam_ind]]
        cam_trans = df.get_cam_trans(trans_body, cam_t)
        cam_trans = np.matmul(cam_rotmat_for_trans, cam_trans.T).T

        w_global_orient, c_global_orient, c_trans, w_trans, cam_rotmat = df.get_global_orient(
            pose, beta, transl, gender, body_yaw, cam_pitch_ind, cam_yaw_ind, cam_roll_ind, cam_trans
        )
        cam_ext = np.zeros((4, 4), dtype=np.float32)
        cam_ext[:3, :3] = cam_rotmat
        cam_ext[:, 3] = np.concatenate([cam_trans, np.array([[1]])], axis=1)

        pose_cam = pose.copy()
        pose_cam[:3] = c_global_orient
        pose_world = pose.copy()
        pose_world[:3] = w_global_orient

        vertices3d, joints3d = df.get_smplx_vertices(pose_cam, beta, c_trans, gender)
        joints2d = df.project(joints3d, cam_trans, cam_int)
        vertices3d_downsample = downsample_vertices(df, vertices3d)
        proj_verts = df.project(vertices3d_downsample, cam_trans, cam_int)

        center, scale, num_vis_joints, _ = df.get_bbox_valid(
            joints2d[:22], rescale=SCALE_FACTOR_BBOX, img_width=df.IMG_W, img_height=df.IMG_H
        )
        if center[0] < 0 or center[1] < 0 or scale <= 0:
            continue
        if num_vis_joints < 8:
            continue

        verts_cam = vertices3d.detach().cpu().numpy() + cam_trans
        if (verts_cam[:, 2] < 0).any():
            continue

        bbox_src = bbox_from_center_scale(center, scale)
        bbox = process_bbox(bbox_src, img_shape[1], img_shape[0], input_img_shape)
        if bbox is None:
            continue

        img_rel = rel_image_path(scene_folder, seq_name, image_path)
        ann_id = "%s:%d" % (out_stem, len(state["imgname"]))
        record = make_record(
            ann_id,
            img_rel,
            img_shape,
            bbox,
            joints2d,
            pose_cam,
            beta,
            c_trans,
            cam_int,
            gender_sub,
        )

        state["records"].append(record)
        state["imgname"].append(img_rel)
        state["center"].append(center)
        state["scale"].append(scale)
        state["pose_cam"].append(pose_cam)
        state["pose_world"].append(pose_world)
        state["shape"].append(beta)
        state["trans_cam"].append(c_trans)
        state["trans_world"].append(w_trans)
        state["gtkps"].append(joints2d)
        state["cam_int"].append(cam_int)
        state["cam_ext"].append(cam_ext)
        state["gender"].append(gender_sub)
        state["proj_verts"].append(proj_verts)
        state["motion_info"].append(motion_info)
        state["sub"].append(sub)
        state["img_shape"].append(img_shape)

    return {
        "selected_frames": len(selected_images),
        "accepted": len(state["records"]) - accepted_before,
    }


def empty_state():
    return {
        "records": [],
        "imgname": [],
        "center": [],
        "scale": [],
        "pose_cam": [],
        "pose_world": [],
        "shape": [],
        "trans_cam": [],
        "trans_world": [],
        "gtkps": [],
        "cam_int": [],
        "cam_ext": [],
        "gender": [],
        "proj_verts": [],
        "motion_info": [],
        "sub": [],
        "img_shape": [],
    }


def checkpoint_payload(state, completed_bodies):
    return {
        "state": state,
        "completed_bodies": sorted(completed_bodies),
    }


def save_pickle_atomic(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def save_checkpoint(state, completed_bodies, checkpoint_path):
    save_pickle_atomic(checkpoint_payload(state, completed_bodies), checkpoint_path)


def load_checkpoint(checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        payload = pickle.load(f)
    state = payload.get("state", empty_state())
    completed_bodies = set(payload.get("completed_bodies", []))
    return state, completed_bodies


def save_outputs(state, npz_path, pkl_path, save_npz=True):
    Path(pkl_path).parent.mkdir(parents=True, exist_ok=True)
    save_pickle_atomic(state["records"], pkl_path)

    if save_npz:
        np.savez(
            npz_path,
            imgname=state["imgname"],
            img_shape=state["img_shape"],
            center=state["center"],
            scale=state["scale"],
            pose_cam=state["pose_cam"],
            pose_world=state["pose_world"],
            shape=state["shape"],
            trans_cam=state["trans_cam"],
            trans_world=state["trans_world"],
            gtkps=state["gtkps"],
            cam_int=state["cam_int"],
            cam_ext=state["cam_ext"],
            gender=state["gender"],
            proj_verts=state["proj_verts"],
            motion_info=state["motion_info"],
            sub=state["sub"],
        )


def make_body_key(scene_folder, seq_name, row_idx, body, start_frame):
    return "%s|%s|%06d|%s|%d" % (scene_folder, seq_name, row_idx, body, start_frame)


def collect_body_jobs(csv_data, cam_csv_base, scene_folder):
    jobs = []
    seq_name = ""
    cam_arrays = None
    for idx, comment in enumerate(csv_data["Comment"]):
        if "sequence_name" in comment:
            seq_name = comment.split(";")[0].split("=")[-1]
            cam_csv = pd.read_csv(cam_csv_base / (seq_name + "_camera.csv")).to_dict("list")
            cam_arrays = (
                cam_csv["x"],
                cam_csv["y"],
                cam_csv["z"],
                cam_csv["yaw"],
                cam_csv["pitch"],
                cam_csv["roll"],
                cam_csv["focal_length"],
            )
            continue

        if "start_frame" not in comment or cam_arrays is None:
            continue

        start_frame = int(comment.split(";")[0].split("=")[-1])
        body = csv_data["Body"][idx]
        jobs.append(
            {
                "key": make_body_key(scene_folder, seq_name, idx, body, start_frame),
                "row_idx": idx,
                "seq_name": seq_name,
                "cam_arrays": cam_arrays,
                "start_frame": start_frame,
                "body": body,
            }
        )
    return jobs


def iter_scene_rows(scene_csv):
    with open(scene_csv, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            yield row[0], row[1]


def main():
    parser = argparse.ArgumentParser(
        description="Process BEDLAM full-body annotations and save both .npz and training-ready .pkl cache."
    )
    parser.add_argument("--bedlam-code-root", type=str, default="/workspace/BEDLAM")
    parser.add_argument("--img_folder", type=str, default="/workspace/BEDLAM_Dataset")
    parser.add_argument("--output_folder", type=str, default="/workspace/BEDLAM_Dataset/processed_labels")
    parser.add_argument(
        "--smplx_gt_folder",
        type=str,
        default="/workspace/BEDLAM_Dataset/smplx_gt/neutral_ground_truth_motioninfo",
    )
    parser.add_argument("--scene_csv", type=str, default=None)
    parser.add_argument("--scene", action="append", default=None, help="process only this folder or scene name; can repeat")
    parser.add_argument("--fps", type=int, default=1, help="output fps; must divide 30")
    parser.add_argument("--img_format", type=str, default=".jpg")
    parser.add_argument("--input_img_shape", type=int, nargs=2, default=(512, 384), metavar=("H", "W"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no_npz", action="store_true", help="only save .pkl cache")
    parser.add_argument("--resume", action="store_true", default=True, help="resume from per-scene checkpoints; enabled by default")
    parser.add_argument("--no_resume", action="store_false", dest="resume", help="ignore checkpoints and restart scenes")
    parser.add_argument("--overwrite", action="store_true", help="reprocess scenes even if final .pkl already exists")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="save checkpoint every N completed bodies")
    parser.add_argument("--no_frame_progress", action="store_true", help="hide per-body frame progress bars")
    args = parser.parse_args()

    if args.fps <= 0 or 30 % args.fps != 0:
        raise ValueError("fps must be a positive divisor of 30")

    img_format = args.img_format if args.img_format.startswith(".") else "." + args.img_format
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print("Using device: %s" % device)
    df = patch_df_for_device(import_df_full_body(args.bedlam_code_root), device)
    scene_csv = args.scene_csv or str(Path(args.bedlam_code_root) / "data_processing" / "bedlam_scene_names.csv")
    scene_rows = list(iter_scene_rows(scene_csv))

    selected = set(args.scene or [])
    scene_iter = tqdm.tqdm(scene_rows, desc="scenes", dynamic_ncols=True)
    for scene_folder, scene_name in scene_iter:
        if selected and scene_folder not in selected and scene_name not in selected:
            continue

        npz_path = output_folder / (scene_folder + ".npz")
        pkl_path = output_folder / (scene_folder + ".pkl")
        checkpoint_path = output_folder / (scene_folder + ".checkpoint.pkl")
        if pkl_path.is_file() and not args.overwrite:
            print("skip existing scene: %s (%s)" % (scene_folder, pkl_path))
            continue

        rotate_flag = "closeup" in scene_name
        if rotate_flag:
            df.SENSOR_W, df.SENSOR_H, df.IMG_W, df.IMG_H = 20.25, 36, 720, 1280
        else:
            df.SENSOR_W, df.SENSOR_H, df.IMG_W, df.IMG_H = 36, 20.25, 1280, 720
        img_shape = (int(df.IMG_H), int(df.IMG_W))

        base_folder = Path(args.img_folder) / scene_folder
        image_folder_base = base_folder / "png"
        csv_path = base_folder / "be_seq.csv"
        cam_csv_base = base_folder / "ground_truth" / "camera"
        if not csv_path.is_file():
            print("skip missing csv: %s" % csv_path)
            continue

        csv_data = pd.read_csv(csv_path).to_dict("list")
        if args.resume and checkpoint_path.is_file() and not args.overwrite:
            state, completed_bodies = load_checkpoint(checkpoint_path)
            print(
                "resume scene %s: %d completed bodies, %d records"
                % (scene_folder, len(completed_bodies), len(state["records"]))
            )
        else:
            state = empty_state()
            completed_bodies = set()
        out_stem = scene_folder
        body_jobs = collect_body_jobs(csv_data, cam_csv_base, scene_folder)
        pending_jobs = [job for job in body_jobs if job["key"] not in completed_bodies]
        scene_iter.set_postfix_str("%s bodies=%d pending=%d records=%d" % (
            scene_folder,
            len(body_jobs),
            len(pending_jobs),
            len(state["records"]),
        ))

        body_bar = tqdm.tqdm(
            pending_jobs,
            desc="bodies %s" % scene_folder,
            leave=False,
            dynamic_ncols=True,
        )
        for body_count, job in enumerate(body_bar, 1):
            idx = job["row_idx"]
            seq_name = job["seq_name"]
            start_frame = job["start_frame"]
            body = job["body"]
            person_parts = body.split("_")
            person_id = "_".join(person_parts[:-1])
            sequence_id = person_parts[-1]
            smplx_path = Path(args.smplx_gt_folder) / person_id / sequence_id / "motion_seq.npz"
            if not smplx_path.is_file():
                print("skip missing smplx gt: %s" % smplx_path)
                completed_bodies.add(job["key"])
                save_checkpoint(state, completed_bodies, checkpoint_path)
                continue

            smplx_param_orig = np.load(smplx_path)
            gender_sub = smplx_param_orig["gender_sub"].item()
            trans_body = [csv_data["X"][idx], csv_data["Y"][idx], csv_data["Z"][idx]]
            body_yaw = csv_data["Yaw"][idx]
            image_folder = image_folder_base / seq_name
            cam_x, cam_y, cam_z, cam_yaw, cam_pitch, cam_roll, fl = job["cam_arrays"]
            stats = process_body(
                df,
                str(image_folder),
                scene_folder,
                seq_name,
                fl,
                start_frame,
                gender_sub,
                smplx_param_orig,
                trans_body,
                body_yaw,
                cam_x,
                cam_y,
                cam_z,
                cam_pitch,
                cam_roll,
                cam_yaw,
                args.fps,
                person_id,
                img_format,
                img_shape,
                tuple(args.input_img_shape),
                rotate_flag,
                out_stem,
                state,
                show_frame_progress=not args.no_frame_progress,
            )
            completed_bodies.add(job["key"])
            if body_count % max(1, args.checkpoint_interval) == 0:
                save_checkpoint(state, completed_bodies, checkpoint_path)
            body_bar.set_postfix_str(
                "accepted=%d total_records=%d frames=%d"
                % (stats["accepted"], len(state["records"]), stats["selected_frames"])
            )

        save_checkpoint(state, completed_bodies, checkpoint_path)
        save_outputs(state, npz_path, pkl_path, save_npz=not args.no_npz)
        if checkpoint_path.is_file():
            checkpoint_path.unlink()
        print("saved %d records: %s" % (len(state["records"]), pkl_path))


if __name__ == "__main__":
    main()
