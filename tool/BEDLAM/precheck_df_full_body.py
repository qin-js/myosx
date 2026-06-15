import argparse
import importlib.util
import os
import sys
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


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


def draw_2d_keypoints(image, joints2d, radius=3):
    out = image.copy()
    joints2d = np.asarray(joints2d)
    for idx, pt in enumerate(joints2d[:, :2]):
        x, y = int(round(pt[0])), int(round(pt[1]))
        if x < 0 or y < 0 or x >= out.shape[1] or y >= out.shape[0]:
            continue
        color = (0, 255 - (idx * 37) % 180, 80 + (idx * 53) % 175)
        cv2.circle(out, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    return out


def put_label(image, label):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 360), 34), (0, 0, 0), -1)
    cv2.putText(out, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def save_panel(df, image_path, joints2d, verts_cam, cam_int, output_dir, scene_name, sub, frame_idx, rotate_flag):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    if rotate_flag:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    h, w = img.shape[:2]
    kpt_img = draw_2d_keypoints(img, joints2d)
    renderer = df.Renderer(focal_length=cam_int[0][0], img_w=w, img_h=h, faces=df.smplx_model_neutral.faces)
    mesh_rgb = renderer.render_front_view(verts_cam[None], bg_img_rgb=img[:, :, ::-1].copy())
    mesh_img = mesh_rgb[:, :, ::-1].copy()
    both_img = draw_2d_keypoints(mesh_img, joints2d)

    panel = np.concatenate(
        [
            put_label(img, "original"),
            put_label(kpt_img, "2D keypoints"),
            put_label(mesh_img, "SMPL-X mesh"),
            put_label(both_img, "keypoints + mesh"),
        ],
        axis=1,
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / ("%s_%s_%06d.jpg" % (scene_name, sub, frame_idx))
    cv2.imwrite(str(out_path), panel, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return out_path


def frame_index_from_name(path, fallback_idx):
    stem = Path(path).stem
    return int(stem) if stem.isdigit() else fallback_idx


def process_one_frame(
    df,
    image_path,
    frame_idx,
    start_frame,
    smplx_param_orig,
    trans_body,
    body_yaw,
    cam_arrays,
    gender_sub,
    sub,
    scene_name,
    rotate_flag,
    output_dir,
):
    cam_x, cam_y, cam_z, cam_yaw, cam_pitch, cam_roll, focal_length = cam_arrays
    smplx_param_ind = frame_idx + start_frame
    if smplx_param_ind >= smplx_param_orig["poses"].shape[0]:
        return None
    if frame_idx >= len(focal_length):
        return None

    pose = smplx_param_orig["poses"][smplx_param_ind]
    transl = smplx_param_orig["trans"][smplx_param_ind]
    beta = smplx_param_orig["betas"]
    gender = smplx_param_orig["gender"]

    cam_pitch_ind = -cam_pitch[frame_idx]
    cam_yaw_ind = -cam_yaw[frame_idx]
    cam_roll_ind = -cam_roll[frame_idx] + 90 if rotate_flag else -cam_roll[frame_idx]

    cam_int = df.get_cam_int(focal_length[frame_idx], df.SENSOR_W, df.SENSOR_H, df.IMG_W / 2.0, df.IMG_H / 2.0)
    body_rotmat, cam_rotmat_for_trans = df.get_cam_rotmat(body_yaw, cam_pitch_ind, cam_yaw_ind, cam_roll_ind)
    cam_t = [cam_x[frame_idx], cam_y[frame_idx], cam_z[frame_idx]]
    cam_trans = df.get_cam_trans(trans_body, cam_t)
    cam_trans = np.matmul(cam_rotmat_for_trans, cam_trans.T).T

    _, c_global_orient, c_trans, _, _ = df.get_global_orient(
        pose, beta, transl, gender, body_yaw, cam_pitch_ind, cam_yaw_ind, cam_roll_ind, cam_trans
    )
    pose_cam = pose.copy()
    pose_cam[:3] = c_global_orient

    vertices3d, joints3d = df.get_smplx_vertices(pose_cam, beta, c_trans, gender)
    joints2d = df.project(joints3d, df.torch.tensor(cam_trans), cam_int)
    center, scale, num_vis_joints, _ = df.get_bbox_valid(
        joints2d[:22], rescale=df.SCALE_FACTOR_BBOX, img_width=df.IMG_W, img_height=df.IMG_H
    )
    if center[0] < 0 or center[1] < 0 or scale <= 0 or num_vis_joints < 8:
        return None

    verts_cam = vertices3d.detach().cpu().numpy() + cam_trans
    if (verts_cam[:, 2] < 0).any():
        return None

    return save_panel(df, image_path, joints2d, verts_cam, cam_int, output_dir, scene_name, sub, frame_idx, rotate_flag)


def iter_scene_rows(scene_csv):
    with open(scene_csv, "r") as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            folder, scene_name = line.split(",", 1)
            yield folder, scene_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bedlam-code-root", type=str, default="E:/BEDLAM/BEDLAM")
    parser.add_argument("--img-folder", type=str, default="/workspace/BEDLAM_Dataset")
    parser.add_argument("--smplx-gt-folder", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="/workspace/BEDLAM_Dataset/precheck_vis")
    parser.add_argument("--scene-csv", type=str, default=None)
    parser.add_argument("--scene", action="append", default=None, help="scene key from bedlam_scene_names.csv; can repeat")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--img-format", type=str, default=".jpg")
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--frames-per-body", type=int, default=1)
    args = parser.parse_args()

    if args.fps <= 0 or 30 % args.fps != 0:
        raise ValueError("fps must be a positive divisor of 30")
    frame_step = 30 // args.fps
    img_format = args.img_format if args.img_format.startswith(".") else "." + args.img_format

    df = import_df_full_body(args.bedlam_code_root)
    scene_csv = args.scene_csv or str(Path(args.bedlam_code_root) / "data_processing" / "bedlam_scene_names.csv")

    saved = []
    for folder, scene_name in iter_scene_rows(scene_csv):
        if args.scene is not None and scene_name not in args.scene and folder not in args.scene:
            continue

        rotate_flag = "closeup" in scene_name
        if rotate_flag:
            df.SENSOR_W, df.SENSOR_H, df.IMG_W, df.IMG_H = 20.25, 36, 720, 1280
        else:
            df.SENSOR_W, df.SENSOR_H, df.IMG_W, df.IMG_H = 36, 20.25, 1280, 720

        base_folder = Path(args.img_folder) / folder
        csv_path = base_folder / "be_seq.csv"
        cam_csv_base = base_folder / "ground_truth" / "camera"
        image_folder_base = base_folder / "png"
        if not csv_path.is_file():
            continue

        csv_data = pd.read_csv(csv_path).to_dict("list")
        seq_name = ""
        cam_arrays = None
        for row_idx, comment in enumerate(csv_data["Comment"]):
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
            body = csv_data["Body"][row_idx]
            person_parts = body.split("_")
            person_id = "_".join(person_parts[:-1])
            sequence_id = person_parts[-1]
            smplx_path = Path(args.smplx_gt_folder) / person_id / sequence_id / "motion_seq.npz"
            if not smplx_path.is_file():
                continue

            smplx_param_orig = np.load(smplx_path)
            gender_sub = smplx_param_orig["gender_sub"].item()
            trans_body = [csv_data["X"][row_idx], csv_data["Y"][row_idx], csv_data["Z"][row_idx]]
            body_yaw = csv_data["Yaw"][row_idx]
            image_folder = image_folder_base / seq_name
            images = sorted(glob(str(image_folder / ("*" + img_format))))

            saved_for_body = 0
            for fallback_idx, image_path in enumerate(images):
                frame_idx = frame_index_from_name(image_path, fallback_idx)
                if frame_idx % frame_step != 0:
                    continue
                out_path = process_one_frame(
                    df,
                    image_path,
                    frame_idx,
                    start_frame,
                    smplx_param_orig,
                    trans_body,
                    body_yaw,
                    cam_arrays,
                    gender_sub,
                    person_id,
                    scene_name,
                    rotate_flag,
                    args.output_dir,
                )
                if out_path is None:
                    continue
                saved.append(out_path)
                print("saved %s" % out_path)
                saved_for_body += 1
                if saved_for_body >= args.frames_per_body or len(saved) >= args.num_samples:
                    break
            if len(saved) >= args.num_samples:
                break
        if len(saved) >= args.num_samples:
            break

    print("done: saved %d precheck panels to %s" % (len(saved), args.output_dir))


if __name__ == "__main__":
    main()
