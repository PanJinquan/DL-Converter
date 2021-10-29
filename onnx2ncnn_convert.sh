#!/usr/bin/env bash

#model_name="ir_mobilenet_v2_model"
#model_name="face_ldmks/rfb_old_149_300_300"
model_name="face/rfb1.0_face_320_320"
onnx_path=data/onnx/$model_name".onnx"
sim_onnx_path=data/onnx/$model_name"_sim.onnx"

ncnn_out=data/ncnn/$model_name
ncnn_sim_out=data/ncnn/$model_name"_sim"

# https://github.com/daquexian/onnx-simplifier
# pip3 install onnx-simplifier
# pip install --upgrade onnx
# python -m onnxsim path/to/src.onnx path/to/src_sim.onnx 0(不做check) --input-shape 1,112,112,3
python3 -m onnxsim  \
    $onnx_path \
    $sim_onnx_path \
    0 \
    --input-shape 1,3,320,320

onnx_path=$sim_onnx_path


#ncnn/tools/onnx/onnx2ncnn $onnx_path  $ncnn_out".param" $ncnn_out".bin"
ncnn/tools/onnx/onnx2ncnn $onnx_path  $ncnn_sim_out".param" $ncnn_sim_out".bin"

parampath=$ncnn_sim_out".param"
binpath=$ncnn_sim_out".bin"

# 量化fp32
parampath_fp32=$ncnn_sim_out"_fp32.param"
binpath_fp32=$ncnn_sim_out"_fp32.bin"

ncnn/tools/ncnnoptimize \
    $parampath \
    $binpath \
    $parampath_fp32 \
    $binpath_fp32 \
    0 # 0--> fp32,1-->fp16


# 量化fp16
parampath_fp16=$ncnn_sim_out"_fp16.param"
binpath_fp16=$ncnn_sim_out"_fp16.bin"
ncnn/tools/ncnnoptimize \
    $parampath \
    $binpath \
    $parampath_fp16 \
    $binpath_fp16 \
    65536 # 0--> fp32,1-->fp16 (65536)


# 生成量化表
#imagepath="/home/dm/panjinquan3/dataset/finger_keypoint/finger_v1/JPEGImages/"
#imagepath="/home/dm/panjinquan3/release/AIT/finger-keypoint-detection/data/test_image/"
#calibration_table=$ncnn_sim_out".table"
#utils/ncnn/tools/quantize/ncnn2table \
#    --param=$parampath_fp32 \
#    --bin=$binpath_fp32 \
#    --images=$imagepath \
#    --output=$calibration_table \
#    --mean=127.5,127.5,127.5 \
#    --norm=0.007843137,0.007843137,0.007843137  \
#    --size=256,256 \
##    --swapRB \
##    --thread=2



# 量化网络模型
#parampath_int8=$ncnn_sim_out"_int8.param"
#binpath_int8=$ncnn_sim_out"_int8.bin"
#utils/ncnn/tools/quantize/ncnn2int8 \
#    $parampath \
#    $binpath \
#    $parampath_int8 \
#    $binpath_int8 \
#    $calibration_table

