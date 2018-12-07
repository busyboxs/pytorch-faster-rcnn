#!/usr/bin/env bash


SESSION=1
EPOCH=6
CHECKPOINT=6583
DATA_SET='kittivoc'
IMAGE_DIR='/media/yangshun/0008EB70000B0B9F/PycharmProjects/faster-rcnn.pytorch/data/KITTIVOC/JPEGImages'
LOAD_DIR='/media/yangshun/0008EB70000B0B9F/roi_roi_new/pytroch/kitti/multi_scale'
SAVE_ROOT='/media/yangshun/0008EB70000B0B9F/roi_roi_new/kitti/txt/pytorch/multi_scale/data'

python demo_kitti.py --net vgg16 \
               --dataset ${DATA_SET} \
               --checksession ${SESSION} \
               --checkepoch ${EPOCH} \
               --checkpoint ${CHECKPOINT} \
               --load_dir ${LOAD_DIR} \
               --image_dir ${IMAGE_DIR} \
               --save_root ${SAVE_ROOT} \
               --cuda