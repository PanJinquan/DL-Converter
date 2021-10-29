#!/usr/bin/env bash

model_name=rbf_1000_1000
onnx_path=data/pretrained/onnx/$model_name".onnx"
simplifier_onnx_path=data/pretrained/onnx/$model_name"_sim.onnx"

mnn_out=data/pretrained/mnn/$model_name".mnn"
mnn_sim_out=data/pretrained/mnn/$model_name"_sim.mnn"

# https://github.com/daquexian/onnx-simplifier
# pip3 install onnx-simplifier
python3 -m onnxsim  \
    $onnx_path \
    $simplifier_onnx_path \

# https://www.yuque.com/mnn/cn/usage_in_python
# pip install -U MNN
mnnconvert -f ONNX \
    --modelFile $onnx_path \
    --MNNModel  $mnn_out

mnnconvert -f ONNX \
    --modelFile $simplifier_onnx_path \
    --MNNModel  $mnn_sim_out

