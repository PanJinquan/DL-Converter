#!/usr/bin/env bash

python demo.py \
    --model_path "data/yolov3_micro0.25_320_320_freeze_head_optimize_float16.tflite" \
    --input_size 320  \
    --prob_threshold=0.2  \
    --iou_threshold=0.3  \
    --image_dir "data/test_image"
