import argparse
import json
import os
import os.path as osp
import pickle
import sys
import types
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


def _install_pandas_pickle_compat():
    module = types.ModuleType('pandas.core.indexes.numeric')
    module.Int64Index = pd.Index
    module.UInt64Index = pd.Index
    module.Float64Index = pd.Index
    sys.modules.setdefault('pandas.core.indexes.numeric', module)


def _load_pickle_from_zip(zip_file, name):
    _install_pandas_pickle_compat()
    with zip_file.open(name) as f:
        data = pickle.load(f, encoding='latin1')
    return {k: list(v) for k, v in data.items()}


def _build_annotation_index(annotation_path):
    with open(annotation_path) as f:
        db = json.load(f)
    images = {img['id']: img for img in db['images']}
    anns_by_image = defaultdict(list)
    for ann in db['annotations']:
        image_name = osp.basename(images[ann['image_id']]['file_name_3840x2160'])
        anns_by_image[image_name].append(ann)
    for anns in anns_by_image.values():
        anns.sort(key=lambda ann: ann['id'])
    return anns_by_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agora_root', default='dataset/AGORA')
    parser.add_argument('--zip_path', default='')
    parser.add_argument('--annotation_path', default='')
    parser.add_argument('--out_data_dir', default='')
    parser.add_argument('--format', choices=('npy', 'json'), default='npy')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    zip_path = args.zip_path or osp.join(args.agora_root, 'validation_SMPLX.zip')
    annotation_path = args.annotation_path or osp.join(args.agora_root, 'data', 'AGORA_validation.json')
    out_data_dir = args.out_data_dir or osp.join(args.agora_root, 'data')

    anns_by_image = _build_annotation_index(annotation_path)
    written = skipped = missing = mismatched = 0

    with zipfile.ZipFile(zip_path) as zf:
        pkl_names = sorted(name for name in zf.namelist() if name.endswith('.pkl'))
        for pkl_name in tqdm(pkl_names, desc='validation SMPL-X pkl'):
            data = _load_pickle_from_zip(zf, pkl_name)
            rows = zip(data['imgPath'], data['gt_verts'], data['isValid'])
            for image_name, verts_list, valid_list in rows:
                anns = anns_by_image.get(image_name)
                if anns is None:
                    missing += 1
                    continue
                if len(anns) != len(verts_list):
                    mismatched += 1
                    continue
                for person_idx, (verts, is_valid) in enumerate(zip(verts_list, valid_list)):
                    ann = anns[person_idx]
                    if bool(ann.get('is_valid')) != bool(is_valid):
                        mismatched += 1
                        continue
                    if not is_valid:
                        continue

                    rel_path = ann['smplx_verts_path'].lstrip('./')
                    out_path = osp.join(out_data_dir, rel_path)
                    if args.format == 'npy':
                        out_path = osp.splitext(out_path)[0] + '.npy'
                    os.makedirs(osp.dirname(out_path), exist_ok=True)
                    if osp.isfile(out_path) and not args.overwrite:
                        skipped += 1
                        continue

                    verts_arr = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
                    if args.format == 'npy':
                        np.save(out_path, verts_arr)
                    else:
                        with open(out_path, 'w') as f:
                            json.dump(verts_arr.tolist(), f, separators=(',', ':'))
                    written += 1

    print('written:', written)
    print('skipped:', skipped)
    print('missing_images:', missing)
    print('mismatched_images_or_people:', mismatched)
    if missing or mismatched:
        raise RuntimeError('AGORA validation SMPL-X extraction had mapping errors')


if __name__ == '__main__':
    main()
