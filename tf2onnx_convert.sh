#!/usr/bin/env bash
name="yolov3_simple0.35_256"
pb_model=data/$name".pb"
out_model=data/$name".onnx"

python -m tf2onnx.convert\
    --input $pb_model \
    --inputs Input:0 \
    --outputs Identity:0,Identity_1:0 \
    --output $out_model \
    --verbose
#    --opset 11