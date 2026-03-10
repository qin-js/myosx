#!/bin/bash
pip install openmim
mim install mmcv-full==1.7.1
pip install -r requirements.txt
cd main/transformer_utils && python setup.py install
conda install -y -c conda-forge ffmpeg

pip install ultralytics==8.3.38
pip install scipy==1.9.0
pip install numpy==1.23.5
pip install chumpy==0.7.0
pip install matplotlib
pip install easydict
pip install einops
pip install timm,pycocotools,plyfile,trimesh
conda install pytorch3d -c pytorch3d

