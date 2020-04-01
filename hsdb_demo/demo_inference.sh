#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=${CONFIG:-"hsdb_demo/checkpoints_configs/cascade_rcnn_hrnetv2p_w32_20e_gastric_syn_rand_v2.py"}
CHECKPOINT=${CHECKPOINT:-"hsdb_demo/checkpoints_configs/cascade_rcnn_hrnetv2p_w32_20e_gastric_syn_rand_v2_epoch_20.pth"}
FRAME_IN_DIR=${FRAME_IN_DIR:-"hsdb_demo/sample_images/gastrec"}
FRAME_OUT_DIR=${FRAME_OUT_DIR:-"hsdb_demo/sample_results"}
SCORE_THRESHOLD=${SCORE_THRESHOLD:-0.5}
$PYTHON hsdb_demo/demo_inference.py $CONFIG $CHECKPOINT $FRAME_IN_DIR $FRAME_OUT_DIR --score_thr $SCORE_THRESHOLD