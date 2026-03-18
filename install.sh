#!/bin/bash
pip install openmim
mim install mmcv-full==1.7.1
pip install -r requirements.txt
cd main/transformer_utils && python setup.py install
conda install -y -c conda-forge ffmpeg

# cuda 12.1
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

pip install ultralytics==8.3.38
pip install scipy==1.9.0
pip install numpy==1.23.5
pip install chumpy==0.7.0
pip install matplotlib
pip install easydict
pip install einops
pip install timm,pycocotools,plyfile,trimesh
conda install pytorch3d -c pytorch3d

