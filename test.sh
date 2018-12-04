#!/usr/bin/env bash

DATA_SET=pascal_voc
NET=vgg16
SESSION=1
EPOCH=6
LOAD_DIR="/media/yangshun/0008EB70000B0B9F/roi_roi_new/pytroch/voc/faster_rcnn"
CHECKPOINT=10021

python test_net.py \
    --dataset ${DATA_SET} \
    --net ${NET} \
    --load_dir ${LOAD_DIR} \
    --checksession ${SESSION} \
    --checkepoch ${EPOCH} \
    --checkpoint ${CHECKPOINT} \
    --cuda