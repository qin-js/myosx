"""Per-epoch training visualization panel.

Saves one diagnostic image per source after each epoch:
  - InterHand: input crop with model-predicted hand boxes + GT hand keypoints,
    to verify the (possibly GT-injected) hand ROI actually tracks the hands.
  - BEDLAM: side-by-side [original | predicted mesh | GT mesh] on the same crop,
    to watch how far the prediction is from ground truth as training progresses.
  - UBody: face-box/keypoint diagnostics plus a zoomed face crop, to verify
    face-only fine-tuning sees valid face supervision.

The GT mesh that BEDLAM stores (``smplx_mesh_cam``) lives in the *original*
camera frame, while the model's predicted mesh lives in the normalized virtual
crop camera (focal=5000, predicted depth). To overlay both on the same crop we
re-derive a virtual-frame GT translation from the GT 2D<->3D joints (a small
least-squares solve), then run the SMPL-X layer on the GT params.

The whole module is defensive: any failure is swallowed and logged, so a
visualization problem can never interrupt a long training run. Solid mesh
rendering uses pytorch3d and falls back to a pure numpy/cv2 vertex scatter.
"""

import os
import os.path as osp

import numpy as np
import torch

from config import cfg
from common.utils.human_models import smpl_x


def _to_bgr_uint8(img_tensor):
    # dataset only applies img/255.0 (ToTensor does not rescale float input),
    # so undo exactly that. Convert RGB -> BGR for cv2.imwrite.
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img[:, :, ::-1].copy()


def _run_test_forward(model, inputs, targets, meta_info):
    batch = {"img": torch.from_numpy(np.asarray(inputs["img"]))[None].cuda()}
    t = {k: torch.from_numpy(np.asarray(v))[None].cuda() for k, v in targets.items()}
    m = {}
    for k, v in meta_info.items():
        arr = np.asarray(v, dtype=np.float32)
        m[k] = torch.from_numpy(arr)[None].cuda()
    with torch.no_grad():
        out = model(batch, t, m, "test")
    return out


def _save_interhand_panel(model, db, epoch, out_dir, cv2):
    inputs, targets, meta_info = db[0]
    out = _run_test_forward(model, inputs, targets, meta_info)
    img = _to_bgr_uint8(out["img"][0])
    h, w = img.shape[:2]

    # GT hand keypoints (output_hm space -> input_img_shape)
    ji = np.asarray(targets["joint_img"]).reshape(smpl_x.joint_num, 3)
    tr = np.asarray(meta_info["joint_trunc"]).reshape(smpl_x.joint_num, -1)
    for part, color in (("lhand", (0, 0, 255)), ("rhand", (0, 255, 0))):
        for j in smpl_x.joint_part[part]:
            if tr[j].sum() <= 0:
                continue
            px = int(ji[j, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1])
            py = int(ji[j, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0])
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(img, (px, py), 3, color, -1)

    # model-predicted hand boxes (already in input_img_shape space). Clip to the
    # image so an over-extended box (extend_ratio=2.0) still shows its edges
    # instead of being drawn off-screen and appearing absent.
    for name, color in (("lhand_bbox", (255, 0, 0)), ("rhand_bbox", (0, 255, 0))):
        b = out[name][0].detach().cpu().numpy().astype(np.float32)
        x0 = int(np.clip(b[0], 0, w - 1)); y0 = int(np.clip(b[1], 0, h - 1))
        x1 = int(np.clip(b[2], 0, w - 1)); y1 = int(np.clip(b[3], 0, h - 1))
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

    dst = osp.join(out_dir, "epoch%03d_interhand.jpg" % epoch)
    cv2.imwrite(dst, img)
    return dst


def _project_mesh_xy(mesh, w, h):
    # mesh_cam already has cam_trans added (see get_coord). Pinhole-project with
    # the virtual camera (focal/princpt are in input_body_shape), then scale to
    # input_img_shape to match the saved crop.
    z = mesh[:, 2] + 1e-4
    x = mesh[:, 0] / z * cfg.focal[0] + cfg.princpt[0]
    y = mesh[:, 1] / z * cfg.focal[1] + cfg.princpt[1]
    x = x / cfg.input_body_shape[1] * cfg.input_img_shape[1]
    y = y / cfg.input_body_shape[0] * cfg.input_img_shape[0]
    return x, y


def _scaled_cam_param():
    # render_mesh expects focal/princpt already in the *output image* pixel space.
    # Scaling the intrinsics to input_img_shape is equivalent to scaling the
    # projected x/y, so the solid render lines up with the scatter fallback.
    sx = cfg.input_img_shape[1] / cfg.input_body_shape[1]
    sy = cfg.input_img_shape[0] / cfg.input_body_shape[0]
    focal = [cfg.focal[0] * sx, cfg.focal[1] * sy]
    princpt = [cfg.princpt[0] * sx, cfg.princpt[1] * sy]
    return {"focal": focal, "princpt": princpt}


def _estimate_virtual_cam_trans(joints_cam, joints_2d_body, valid):
    """Solve the virtual-frame translation t s.t. projecting (J + t) with the
    virtual camera (cfg.focal / cfg.princpt) matches the GT 2D joints.

    joints_cam:     (J, 3) absolute SMPL-X joints in metres (same convention
                    get_coord projects before the root-relative subtraction).
    joints_2d_body: (J, 2) GT 2D joints in input_body_shape pixel space.
    valid:          (J,) boolean / float mask.

    Per joint: x = (Jx + tx)/(Jz + tz) * f + cx  ->  linear in (tx, ty, tz):
        f*tx - (x - cx)*tz = (x - cx)*Jz - f*Jx
        f*ty - (y - cy)*tz = (y - cy)*Jz - f*Jy
    """
    fx, fy = float(cfg.focal[0]), float(cfg.focal[1])
    cx, cy = float(cfg.princpt[0]), float(cfg.princpt[1])
    m = np.asarray(valid).reshape(-1) > 0
    if m.sum() < 6:
        return None
    J = joints_cam[m]
    uv = joints_2d_body[m]
    dx = uv[:, 0] - cx
    dy = uv[:, 1] - cy
    rows, rhs = [], []
    for i in range(J.shape[0]):
        rows.append([fx, 0.0, -dx[i]]); rhs.append(dx[i] * J[i, 2] - fx * J[i, 0])
        rows.append([0.0, fy, -dy[i]]); rhs.append(dy[i] * J[i, 2] - fy * J[i, 1])
    A = np.asarray(rows, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    t, *_ = np.linalg.lstsq(A, b, rcond=None)
    if not np.all(np.isfinite(t)) or t[2] <= 0:
        return None
    return t.astype(np.float32)


def _gt_mesh_virtual(model, targets):
    """Build the GT SMPL-X mesh in the virtual crop camera frame.

    Returns (V, 3) numpy vertices already translated by the estimated virtual
    cam_trans, or None if it cannot be reconstructed.
    """
    core = model.module if hasattr(model, "module") else model
    layer = getattr(core, "smplx_layer", None)
    if layer is None or "smplx_pose" not in targets:
        return None

    pose = torch.from_numpy(np.asarray(targets["smplx_pose"], dtype=np.float32))[None].cuda()
    shape = torch.from_numpy(np.asarray(targets["smplx_shape"], dtype=np.float32))[None].cuda()
    expr = torch.from_numpy(np.asarray(targets["smplx_expr"], dtype=np.float32))[None].cuda()
    # split: root(3) body(63) lhand(45) rhand(45) jaw(3)
    a = [3, 66, 111, 156, 159]
    root, body = pose[:, :a[0]], pose[:, a[0]:a[1]]
    lhand, rhand, jaw = pose[:, a[1]:a[2]], pose[:, a[2]:a[3]], pose[:, a[3]:a[4]]
    zero = torch.zeros((1, 3)).float().cuda()
    with torch.no_grad():
        output = layer(betas=shape, body_pose=body, global_orient=root, right_hand_pose=rhand,
                       left_hand_pose=lhand, jaw_pose=jaw, leye_pose=zero, reye_pose=zero, expression=expr)
    verts = output.vertices[0].detach().cpu().numpy()
    joints = output.joints[0].detach().cpu().numpy()[smpl_x.joint_idx, :]

    # GT 2D joints (output_hm space) -> input_body_shape pixels.
    gj = np.asarray(targets["smplx_joint_img"]).reshape(smpl_x.joint_num, 3)
    uv = np.stack([
        gj[:, 0] / cfg.output_hm_shape[2] * cfg.input_body_shape[1],
        gj[:, 1] / cfg.output_hm_shape[1] * cfg.input_body_shape[0],
    ], axis=1)
    # body joints give the most stable translation solve.
    valid = np.zeros((smpl_x.joint_num,), dtype=np.float32)
    valid[smpl_x.joint_part["body"]] = 1.0
    t = _estimate_virtual_cam_trans(joints, uv, valid)
    if t is None:
        return None
    return verts + t[None, :]


def _render_or_scatter(img, mesh, cv2):
    """Solid pytorch3d render onto img (BGR uint8); scatter fallback. Returns
    (img, rendered_bool)."""
    try:
        from common.utils.vis import render_mesh
        rgb = img[:, :, ::-1].astype(np.float32)
        rgb = render_mesh(rgb, mesh, smpl_x.face, _scaled_cam_param())
        return np.clip(rgb, 0, 255).astype(np.uint8)[:, :, ::-1].copy(), True
    except Exception:  # pragma: no cover
        out = img.copy()
        h, w = out.shape[:2]
        x, y = _project_mesh_xy(mesh, w, h)
        step = max(1, mesh.shape[0] // 2000)
        for i in range(0, mesh.shape[0], step):
            px, py = int(x[i]), int(y[i])
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(out, (px, py), 1, (0, 200, 255), -1)
        return out, False


def _label(img, text, cv2):
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _draw_xyxy(img, bbox, color, cv2, thickness=2):
    h, w = img.shape[:2]
    b = np.asarray(bbox, dtype=np.float32).reshape(4)
    x0 = int(np.clip(b[0], 0, w - 1)); y0 = int(np.clip(b[1], 0, h - 1))
    x1 = int(np.clip(b[2], 0, w - 1)); y1 = int(np.clip(b[3], 0, h - 1))
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)


def _bbox_center_size_hm_to_img(center, size):
    center = np.asarray(center, dtype=np.float32).reshape(2)
    size = np.asarray(size, dtype=np.float32).reshape(2)
    x0 = (center[0] - size[0] * 0.5) / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
    y0 = (center[1] - size[1] * 0.5) / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
    x1 = (center[0] + size[0] * 0.5) / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
    y1 = (center[1] + size[1] * 0.5) / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def _joints_hm_to_img(joints):
    joints = np.asarray(joints, dtype=np.float32).copy()
    joints[:, 0] = joints[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
    joints[:, 1] = joints[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
    return joints


def _draw_joints(img, joints, valid, color, cv2, radius=2):
    h, w = img.shape[:2]
    joints = np.asarray(joints, dtype=np.float32)
    valid = np.asarray(valid).reshape(-1) > 0
    for pt, ok in zip(joints, valid):
        if not ok or not np.isfinite(pt[:2]).all():
            continue
        px, py = int(pt[0]), int(pt[1])
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(img, (px, py), radius, color, -1)


def _crop_rect_from_boxes(boxes, img_shape, margin=0.35):
    h, w = img_shape[:2]
    valid = []
    for b in boxes:
        if b is None:
            continue
        b = np.asarray(b, dtype=np.float32).reshape(4)
        if np.isfinite(b).all() and b[2] > b[0] and b[3] > b[1]:
            valid.append(b)
    if not valid:
        return np.asarray([0, 0, w - 1, h - 1], dtype=np.float32)
    b = np.stack(valid)
    x0, y0 = b[:, 0].min(), b[:, 1].min()
    x1, y1 = b[:, 2].max(), b[:, 3].max()
    bw, bh = x1 - x0, y1 - y0
    pad = max(bw, bh) * margin
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    side = max(bw, bh) + 2 * pad
    x0, y0 = cx - side * 0.5, cy - side * 0.5
    x1, y1 = cx + side * 0.5, cy + side * 0.5
    return np.asarray([
        np.clip(x0, 0, w - 1),
        np.clip(y0, 0, h - 1),
        np.clip(x1, 1, w),
        np.clip(y1, 1, h),
    ], dtype=np.float32)


def _transform_xyxy_to_crop(bbox, crop, dst_w, dst_h):
    b = np.asarray(bbox, dtype=np.float32).reshape(4)
    x0, y0, x1, y1 = crop
    scale_x = dst_w / max(float(x1 - x0), 1.0)
    scale_y = dst_h / max(float(y1 - y0), 1.0)
    return np.asarray([
        (b[0] - x0) * scale_x,
        (b[1] - y0) * scale_y,
        (b[2] - x0) * scale_x,
        (b[3] - y0) * scale_y,
    ], dtype=np.float32)


def _transform_joints_to_crop(joints, crop, dst_w, dst_h):
    out = np.asarray(joints, dtype=np.float32).copy()
    x0, y0, x1, y1 = crop
    out[:, 0] = (out[:, 0] - x0) * (dst_w / max(float(x1 - x0), 1.0))
    out[:, 1] = (out[:, 1] - y0) * (dst_h / max(float(y1 - y0), 1.0))
    return out


def _save_ubody_panel(model, db, epoch, out_dir, cv2):
    sample = None
    for idx in range(min(len(db), 50)):
        cand = db[idx]
        meta_info = cand[2]
        face_valid = float(np.asarray(meta_info.get("face_bbox_valid", 0.0)).reshape(-1)[0])
        face_trunc = np.asarray(meta_info.get("joint_trunc", np.zeros((smpl_x.joint_num, 1))))
        if face_valid > 0 and face_trunc[smpl_x.joint_part["face"]].sum() > 0:
            sample = cand
            break
    if sample is None:
        sample = db[0]

    inputs, targets, meta_info = sample
    out = _run_test_forward(model, inputs, targets, meta_info)
    base = _to_bgr_uint8(out["img"][0])
    h, w = base.shape[:2]

    face_idx = smpl_x.joint_part["face"]
    face_valid = float(np.asarray(meta_info.get("face_bbox_valid", 0.0)).reshape(-1)[0])
    gt_bbox = None
    if face_valid > 0:
        gt_bbox = _bbox_center_size_hm_to_img(targets["face_bbox_center"], targets["face_bbox_size"])
    pred_bbox = out["face_bbox"][0].detach().cpu().numpy().astype(np.float32)

    gt_face_joints = _joints_hm_to_img(np.asarray(targets["joint_img"])[face_idx])
    gt_face_valid = np.asarray(meta_info["joint_trunc"])[face_idx, 0]
    pred_joints_hm = out["smplx_joint_proj"][0].detach().cpu().numpy()[face_idx]
    pred_face_joints = _joints_hm_to_img(pred_joints_hm)
    pred_face_valid = np.ones((len(face_idx),), dtype=np.float32)

    orig = _label(base.copy(), "orig", cv2)

    gt_panel = base.copy()
    if gt_bbox is not None:
        _draw_xyxy(gt_panel, gt_bbox, (0, 255, 255), cv2, 2)
    _draw_joints(gt_panel, gt_face_joints, gt_face_valid, (0, 255, 0), cv2, 2)
    _label(gt_panel, "GT face", cv2)

    pred_panel = base.copy()
    _draw_xyxy(pred_panel, pred_bbox, (255, 255, 0), cv2, 2)
    _draw_joints(pred_panel, pred_face_joints, pred_face_valid, (0, 0, 255), cv2, 2)
    _label(pred_panel, "pred face", cv2)

    crop = _crop_rect_from_boxes([gt_bbox, pred_bbox], base.shape)
    x0, y0, x1, y1 = crop.astype(np.int32)
    zoom = base[y0:y1, x0:x1].copy()
    if zoom.size == 0:
        zoom = base.copy()
        crop = np.asarray([0, 0, w, h], dtype=np.float32)
    zoom = cv2.resize(zoom, (w, h), interpolation=cv2.INTER_LINEAR)
    if gt_bbox is not None:
        _draw_xyxy(zoom, _transform_xyxy_to_crop(gt_bbox, crop, w, h), (0, 255, 255), cv2, 2)
    _draw_xyxy(zoom, _transform_xyxy_to_crop(pred_bbox, crop, w, h), (255, 255, 0), cv2, 2)
    _draw_joints(zoom, _transform_joints_to_crop(gt_face_joints, crop, w, h), gt_face_valid, (0, 255, 0), cv2, 2)
    _draw_joints(zoom, _transform_joints_to_crop(pred_face_joints, crop, w, h), pred_face_valid, (0, 0, 255), cv2, 2)
    _label(zoom, "face zoom", cv2)

    panel = np.concatenate([orig, gt_panel, pred_panel, zoom], axis=1)
    for k in (1, 2, 3):
        cv2.line(panel, (w * k, 0), (w * k, h), (255, 255, 255), 1)
    dst = osp.join(out_dir, "epoch%03d_ubody_face.jpg" % epoch)
    cv2.imwrite(dst, panel)
    return dst


def _save_bedlam_panel(model, db, epoch, out_dir, cv2):
    inputs, targets, meta_info = db[0]
    out = _run_test_forward(model, inputs, targets, meta_info)
    base = _to_bgr_uint8(out["img"][0])

    # left: predicted mesh
    pred_mesh = out["smplx_mesh_cam"][0].detach().cpu().numpy()
    left, rendered = _render_or_scatter(base, pred_mesh, cv2)
    _label(left, "pred", cv2)

    # right: GT mesh (reconstructed into the virtual crop frame)
    gt_mesh = None
    try:
        gt_mesh = _gt_mesh_virtual(model, targets)
    except Exception:  # pragma: no cover
        gt_mesh = None
    if gt_mesh is not None:
        right, _ = _render_or_scatter(base, gt_mesh, cv2)
        _label(right, "GT", cv2)
    else:
        right = base.copy()
        _label(right, "GT (n/a)", cv2)

    # leftmost: clean original crop for reference
    orig = base.copy()
    _label(orig, "orig", cv2)

    panel = np.concatenate([orig, left, right], axis=1)
    w = base.shape[1]
    for k in (1, 2):
        cv2.line(panel, (w * k, 0), (w * k, base.shape[0]), (255, 255, 255), 1)
    suffix = "" if rendered else "_scatter"
    dst = osp.join(out_dir, "epoch%03d_bedlam%s.jpg" % (epoch, suffix))
    cv2.imwrite(dst, panel)
    return dst


def save_epoch_panels(model, trainset_by_name, epoch, out_dir, logger=None):
    """Save one diagnostic image per available training source for this epoch.

    Never raises: every step is guarded so visualization cannot break training.
    """
    def _log(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        _log("[epoch-vis] skipped (cv2 import failed: %s)" % exc)
        return

    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as exc:
        _log("[epoch-vis] skipped (mkdir failed: %s)" % exc)
        return

    was_training = model.training
    model.eval()
    try:
        panels = (
            ("InterHand26M", _save_interhand_panel),
            ("BEDLAM", _save_bedlam_panel),
            ("UBody", _save_ubody_panel),
        )
        for name, fn in panels:
            db = trainset_by_name.get(name)
            if db is None or len(db) == 0:
                continue
            try:
                dst = fn(model, db, epoch, out_dir, cv2)
                _log("[epoch-vis] saved %s -> %s" % (name, dst))
            except Exception as exc:  # pragma: no cover
                _log("[epoch-vis] %s panel failed: %s" % (name, exc))
    finally:
        if was_training:
            model.train()
