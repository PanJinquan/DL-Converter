#!/usr/bin/env bash

model_name="rbf_300_300"
onnx_path=data/pretrained/onnx_ncnn/$model_name".onnx"
simplifier_onnx_path=data/pretrained/onnx_ncnn/$model_name"_sim.onnx"

ncnn_out=data/pretrained/ncnn/$model_name
ncnn_sim_out=data/pretrained/ncnn/$model_name"_sim"

# https://github.com/daquexian/onnx-simplifier
# pip3 install onnx-simplifier
# pip install --upgrade onnx
# python -m onnxsim path/to/src.onnx path/to/src_sim.onnx 0(不做check) --input-shape 1,112,112,3
python3 -m onnxsim  \
    $onnx_path \
    $simplifier_onnx_path \
#    0 \
#    --input-shape 1,112,112,3