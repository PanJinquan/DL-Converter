#!/usr/bin/env bash
# step
# docker run --rm --volume=$(pwd):/opt/TNN/tools/onnx2tnn/onnx-converter/workspace  -it tnn-convert:latest  /bin/bash
# cd /opt/TNN/tools/onnx2tnn/onnx-converter
# bash workspace/utils/onnx2tnn.sh

model_name="rfb1.0_face_320_320_freeze_header"
#model_name="best_model_164_0.9478"
onnx_path="workspace/data/pretrained/onnx/"$model_name".onnx"
sim_onnx_path="workspace/data/pretrained/onnx/"$model_name"_sim.onnx"
tnn_model="workspace/data/pretrained/tnn/"


# https://github.com/daquexian/onnx-simplifier
# pip3 install onnx-simplifier
# pip install --upgrade onnx
# python -m onnxsim path/to/src.onnx path/to/src_sim.onnx 0(不做check) --input-shape 1,112,112,3
python3 -m onnxsim  \
    $onnx_path \
    $sim_onnx_path \
    0 \
    --input-shape 1,3,320,320
#
#onnx_path=$sim_onnx_path


# https://github.com/Tencent/TNN/blob/master/doc/cn/user/onnx2tnn.md
#python3 converter.py onnx2tnn \
#    $onnx_path  \
#    -optimize \
#    -v=v3.0 \
#    -o $tnn_model \
#    -align \
#    -input_file in.txt \
#    -ref_file ref.txt


python3 onnx2tnn.py \
    $onnx_path \
    -version=v3.0 \
    -optimize=1 \
    -half=0 \
    -o $tnn_model \
