import argparse
import os
import os.path as osp
import re
import sys

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

DEMO_DIR = osp.dirname(osp.abspath(__file__))
ROOT_DIR = osp.abspath(osp.join(DEMO_DIR, '..'))
sys.path.insert(0, osp.join(ROOT_DIR, 'main'))
sys.path.insert(0, osp.join(ROOT_DIR, 'data'))

from config import cfg

# os.environ["PYOPENGL_PLATFORM"] = "egl"

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='0')
    parser.add_argument(
        '--img_path',
        type=str,
        nargs='+',
        default=['input.png'],
        help='One or more image files and/or image directories.')
    parser.add_argument('--recursive', action='store_true',
                        help='Recursively scan image directories.')
    parser.add_argument('--output_folder', type=str, default='output')
    parser.add_argument('--encoder_setting', type=str, default='osx_l', choices=['osx_b', 'osx_l'])
    parser.add_argument('--decoder_setting', type=str, default='normal',
                        choices=['normal', 'wo_face_decoder', 'wo_decoder', 'pytorch'])
    parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_models/osx_l.pth.tar')
    parser.add_argument('--continue_train_path', type=str, default='')
    parser.add_argument('--continue_train_paths', type=str, nargs='+', default=None,
                        help='Multiple lightweight snapshots for side-by-side comparison.')
    parser.add_argument('--model_labels', type=str, nargs='+', default=None,
                        help='Optional labels for compared models. Count must match compared models.')
    parser.add_argument('--include_base', action='store_true',
                        help='Include the pytorch pretrained/no-snapshot model in comparison.')
    parser.add_argument('--compare_cols', type=int, default=0,
                        help='Columns in stitched comparison image. 0 means all models in one row.')
    parser.add_argument('--det_conf', type=float, default=0.5,
                        help='YOLO person confidence threshold.')
    parser.add_argument('--det_iou', type=float, default=0.4,
                        help='YOLO NMS IoU threshold.')
    parser.add_argument('--max_persons', type=int, default=0,
                        help='Maximum persons per image. 0 means all detected persons.')
    parser.add_argument('--detector_repo', type=str, default='ultralytics/yolov5',
                        help='torch.hub repo or local YOLOv5 repo path.')
    parser.add_argument('--detector_source', type=str, default='github', choices=['github', 'local'])
    parser.add_argument('--detector_model', type=str, default='yolov5s')
    parser.add_argument('--detector_weights', type=str, default='',
                        help='Optional custom YOLO weights. Uses torch.hub custom model when set.')
    parser.add_argument('--save_debug_npy', action='store_true',
                        help='Save per-person mesh/face/camera numpy debug files.')
    args = parser.parse_args()

    if not args.gpu_ids:
        raise ValueError("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args


def resolve_existing_path(path):
    if not path:
        return path
    if osp.isabs(path) or osp.exists(path):
        return path
    for base_dir in (DEMO_DIR, ROOT_DIR):
        candidate = osp.abspath(osp.join(base_dir, path))
        if osp.exists(candidate):
            return candidate
    return path


def is_image_file(path):
    return osp.isfile(path) and osp.splitext(path)[1].lower() in IMAGE_EXTS


def collect_image_paths(paths, recursive=False):
    image_paths = []
    for raw_path in paths:
        path = resolve_existing_path(raw_path)
        if osp.isdir(path):
            if recursive:
                for root, _, files in os.walk(path):
                    for file_name in files:
                        candidate = osp.join(root, file_name)
                        if is_image_file(candidate):
                            image_paths.append(candidate)
            else:
                for file_name in os.listdir(path):
                    candidate = osp.join(path, file_name)
                    if is_image_file(candidate):
                        image_paths.append(candidate)
        elif is_image_file(path):
            image_paths.append(path)
        else:
            print(f'[skip] not an image file or directory: {raw_path}')

    image_paths = sorted(osp.abspath(p) for p in image_paths)
    if not image_paths:
        raise FileNotFoundError('No input images found.')
    return image_paths


def safe_stem(path):
    stem = osp.splitext(osp.basename(path))[0]
    stem = re.sub(r'[^A-Za-z0-9_.-]+', '_', stem).strip('._')
    return stem or 'image'


def unique_output_stems(image_paths):
    counts = {}
    stems = []
    for path in image_paths:
        base = safe_stem(path)
        count = counts.get(base, 0)
        counts[base] = count + 1
        stems.append(base if count == 0 else f'{base}_{count:03d}')
    return stems


def safe_label(label):
    label = re.sub(r'[^A-Za-z0-9_.-]+', '_', label).strip('._')
    return label or 'model'


def label_from_snapshot(path):
    if not path:
        return 'base'
    name = osp.basename(path)
    for suffix in ('.pth.tar', '.pth', '.pt', '.tar'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return safe_label(name)


def unique_labels(labels):
    counts = {}
    unique = []
    for label in labels:
        base = safe_label(label)
        count = counts.get(base, 0)
        counts[base] = count + 1
        unique.append(base if count == 0 else f'{base}_{count:03d}')
    return unique


def make_model_specs(args):
    paths = []
    if args.include_base:
        paths.append('')
    if args.continue_train_paths:
        paths.extend(args.continue_train_paths)
    elif args.continue_train_path:
        paths.append(args.continue_train_path)
    elif not paths:
        paths.append('')

    paths = [resolve_existing_path(path) for path in paths]
    if args.decoder_setting != 'pytorch' and any(paths):
        raise ValueError('Snapshot comparison requires --decoder_setting pytorch.')

    labels = args.model_labels if args.model_labels else [label_from_snapshot(path) for path in paths]
    if len(labels) != len(paths):
        raise ValueError('--model_labels count must match the compared model count.')
    labels = unique_labels(labels)
    return [{'path': path, 'label': label} for path, label in zip(paths, labels)]


def make_output_dirs(output_folder, save_debug_npy=False, model_label=None):
    root = output_folder if model_label is None else osp.join(output_folder, 'models', model_label)
    dirs = {
        'root': root,
        'render': osp.join(root, 'render'),
        'kpts': osp.join(root, 'kpts'),
        'obj': osp.join(root, 'obj'),
    }
    if save_debug_npy:
        dirs['debug'] = osp.join(root, 'debug')
    for out_dir in dirs.values():
        os.makedirs(out_dir, exist_ok=True)
    return dirs


def make_compare_dirs(output_folder):
    dirs = {
        'render': osp.join(output_folder, 'compare', 'render'),
        'kpts': osp.join(output_folder, 'compare', 'kpts'),
    }
    for out_dir in dirs.values():
        os.makedirs(out_dir, exist_ok=True)
    return dirs


def load_detector(args):
    detector_repo = args.detector_repo
    if args.detector_source == 'local':
        detector_repo = resolve_existing_path(detector_repo)
    if args.detector_weights:
        weights = resolve_existing_path(args.detector_weights)
        return torch.hub.load(detector_repo, 'custom', path=weights, source=args.detector_source)
    return torch.hub.load(
        detector_repo,
        args.detector_model,
        pretrained=True,
        source=args.detector_source)


def flatten_nms_indices(indices):
    if indices is None or len(indices) == 0:
        return []
    return [int(i) for i in np.asarray(indices).reshape(-1)]


def detect_person_boxes(detector, original_img, det_conf, det_iou, max_persons):
    with torch.no_grad():
        results = detector(original_img)
    person_results = results.xyxy[0][results.xyxy[0][:, 5] == 0]

    boxes, confidences = [], []
    for detection in person_results:
        x1, y1, x2, y2, confidence, _ = detection.tolist()
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        confidences.append(float(confidence))

    if not boxes:
        return []

    selected = flatten_nms_indices(cv2.dnn.NMSBoxes(boxes, confidences, det_conf, det_iou))
    detections = [(boxes[i], confidences[i]) for i in selected]
    detections.sort(key=lambda x: x[1], reverse=True)
    if max_persons > 0:
        detections = detections[:max_persons]
    return detections


def detect_image(detector, image_path, args):
    from common.utils.preprocessing import load_img

    original_img = load_img(image_path)
    detections = detect_person_boxes(
        detector,
        original_img,
        args.det_conf,
        args.det_iou,
        args.max_persons)
    if not detections:
        print(f'[warn] no person detected: {image_path}')
    return detections


def run_image(demoer, transform, image_path, out_stem, out_dirs, args, detections):
    from common.utils.preprocessing import load_img, process_bbox, generate_patch_image
    from common.utils.vis import render_mesh, save_obj, vis_keypoints
    from common.utils.human_models import smpl_x

    original_img = load_img(image_path)
    original_img_height, original_img_width = original_img.shape[:2]

    vis_mesh = original_img.copy()
    vis_kpts = original_img.copy()
    processed_persons = 0

    for person_idx, (det_bbox, confidence) in enumerate(detections):
        bbox = process_bbox(det_bbox, original_img_width, original_img_height)
        if bbox is None:
            print(f'[warn] invalid bbox skipped: {image_path} person={person_idx}')
            continue

        img, _, bb2img_trans = generate_patch_image(
            original_img,
            bbox,
            1.0,
            0.0,
            False,
            cfg.input_img_shape)
        img = transform(img.astype(np.float32)) / 255
        img = img.cuda()[None, :, :, :]
        inputs = {'img': img}
        targets = {}
        meta_info = {}

        with torch.no_grad():
            out = demoer.model(inputs, targets, meta_info, 'test')

        mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]
        obj_name = f'{out_stem}_person_{person_idx:02d}.obj'
        save_obj(mesh, smpl_x.face, osp.join(out_dirs['obj'], obj_name))

        focal = [
            cfg.focal[0] / cfg.input_body_shape[1] * bbox[2],
            cfg.focal[1] / cfg.input_body_shape[0] * bbox[3],
        ]
        princpt = [
            cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0],
            cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1],
        ]

        try:
            vis_mesh = render_mesh(vis_mesh, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})
        except Exception as exc:
            print(f'[warn] render failed: {image_path} person={person_idx}: {exc}')

        joint_proj = out['smplx_joint_proj'].detach().cpu().numpy()[0]
        joint_proj[:, 0] = joint_proj[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
        joint_proj[:, 1] = joint_proj[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
        joint_proj = np.concatenate((joint_proj, np.ones_like(joint_proj[:, :1])), 1)
        joint_proj = np.dot(bb2img_trans, joint_proj.transpose(1, 0)).transpose(1, 0)
        vis_kpts = vis_keypoints(vis_kpts, joint_proj)

        if args.save_debug_npy:
            debug_prefix = osp.join(out_dirs['debug'], f'{out_stem}_person_{person_idx:02d}')
            np.save(debug_prefix + '_mesh.npy', mesh)
            np.save(debug_prefix + '_face.npy', smpl_x.face)
            np.save(debug_prefix + '_cam.npy', np.array([focal, princpt], dtype=np.float32))
            np.save(debug_prefix + '_confidence.npy', np.array([confidence], dtype=np.float32))

        processed_persons += 1

    cv2.imwrite(osp.join(out_dirs['render'], f'{out_stem}.jpg'), vis_mesh[:, :, ::-1])
    cv2.imwrite(osp.join(out_dirs['kpts'], f'{out_stem}.jpg'), vis_kpts[:, :, ::-1])
    return processed_persons


def annotate_panel(img, label):
    img = np.clip(img, 0, 255).astype(np.uint8)
    label_bar = 32
    canvas = np.full((img.shape[0] + label_bar, img.shape[1], 3), 255, dtype=np.uint8)
    canvas[label_bar:, :, :] = img
    cv2.putText(
        canvas,
        label,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        cv2.LINE_AA)
    return canvas


def pad_panel(img, height, width):
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    canvas[:img.shape[0], :img.shape[1], :] = img
    return canvas


def make_panel_grid(panels, cols):
    if cols <= 0:
        cols = len(panels)
    cols = max(1, min(cols, len(panels)))
    cell_h = max(panel.shape[0] for panel in panels)
    cell_w = max(panel.shape[1] for panel in panels)

    rows = []
    for start in range(0, len(panels), cols):
        row_panels = panels[start:start + cols]
        padded = [pad_panel(panel, cell_h, cell_w) for panel in row_panels]
        while len(padded) < cols:
            padded.append(np.full((cell_h, cell_w, 3), 255, dtype=np.uint8))
        rows.append(np.concatenate(padded, axis=1))
    return np.concatenate(rows, axis=0)


def stitch_comparisons(output_folder, model_specs, model_out_dirs, out_stems, compare_cols):
    compare_dirs = make_compare_dirs(output_folder)
    for kind in ('render', 'kpts'):
        for out_stem in out_stems:
            panels = []
            for spec in model_specs:
                label = spec['label']
                img_path = osp.join(model_out_dirs[label][kind], f'{out_stem}.jpg')
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    print(f'[warn] cannot read comparison panel: {img_path}')
                    continue
                panels.append(annotate_panel(img, label))
            if not panels:
                continue
            grid = make_panel_grid(panels, compare_cols)
            cv2.imwrite(osp.join(compare_dirs[kind], f'{out_stem}.jpg'), grid)


def release_model(demoer):
    del demoer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    args.pretrained_model_path = resolve_existing_path(args.pretrained_model_path)

    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True

    model_path = args.pretrained_model_path
    assert osp.exists(model_path), 'Cannot find model at ' + model_path

    image_paths = collect_image_paths(args.img_path, args.recursive)
    out_stems = unique_output_stems(image_paths)
    model_specs = make_model_specs(args)
    multi_model = len(model_specs) > 1
    transform = transforms.ToTensor()

    detector = load_detector(args)
    detections_by_path = {}
    for image_idx, (image_path, out_stem) in enumerate(zip(image_paths, out_stems), start=1):
        print(f'[detect {image_idx}/{len(image_paths)}] {image_path} -> {out_stem}')
        detections_by_path[image_path] = detect_image(detector, image_path, args)
    del detector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from common.base import Demoer

    model_out_dirs = {}
    total_persons_by_model = {}
    for model_idx, spec in enumerate(model_specs, start=1):
        label = spec['label']
        continue_path = spec['path']
        if continue_path and not osp.exists(continue_path):
            raise FileNotFoundError('Cannot find snapshot at ' + continue_path)

        cfg.set_additional_args(
            encoder_setting=args.encoder_setting,
            decoder_setting=args.decoder_setting,
            pretrained_model_path=args.pretrained_model_path,
            continue_train_path=continue_path)

        print(f'[model {model_idx}/{len(model_specs)}] label={label} continue={continue_path or "none"}')
        print('Load checkpoint from {}'.format(model_path))
        demoer = Demoer()
        demoer._make_model()
        demoer.model.eval()

        out_dirs = make_output_dirs(
            args.output_folder,
            args.save_debug_npy,
            model_label=label if multi_model else None)
        model_out_dirs[label] = out_dirs

        total_persons = 0
        for image_idx, (image_path, out_stem) in enumerate(zip(image_paths, out_stems), start=1):
            print(f'[{label} {image_idx}/{len(image_paths)}] {image_path} -> {out_stem}')
            persons = run_image(
                demoer,
                transform,
                image_path,
                out_stem,
                out_dirs,
                args,
                detections_by_path[image_path])
            total_persons += persons
        total_persons_by_model[label] = total_persons
        release_model(demoer)

    if multi_model:
        stitch_comparisons(args.output_folder, model_specs, model_out_dirs, out_stems, args.compare_cols)

    print(f'Done. images={len(image_paths)} models={len(model_specs)} output={args.output_folder}')
    for label, total_persons in total_persons_by_model.items():
        print(f'  {label}: persons={total_persons}')


if __name__ == '__main__':
    main()
