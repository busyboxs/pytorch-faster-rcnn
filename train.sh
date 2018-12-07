#!/usr/bin/env bash

GPU_ID=0
BATCH_SIZE=1
WORKER_NUMBER=1
LEARNING_RATE=0.001
DECAY_STEP=5
MAX_EPOCH=6
SAVE_DIR="/media/yangshun/0008EB70000B0B9F/roi_roi_new/pytroch/kitti/multi_scale"

CUDA_VISIBLE_DEVICES=${GPU_ID} python trainval_net.py \
                   --dataset kittivoc \
                   --net vgg16 \
                   --bs ${BATCH_SIZE} \
                   --epochs ${MAX_EPOCH} \
                   --nw ${WORKER_NUMBER} \
                   --lr ${LEARNING_RATE} \
                   --lr_decay_step ${DECAY_STEP} \
                   --save_dir ${SAVE_DIR} \
                   --cuda