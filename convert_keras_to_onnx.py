# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: tf-Face-Recognize-Pipeline
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-06-03 12:27:37
# --------------------------------------------------------
"""
# -*-coding: utf-8 -*-
import os
import sys

sys.path.append(os.getcwd())
import glob
import argparse
import sys
import tensorflow as tf
import datetime
import tensorflow as tf
import keras2onnx
import onnx
from onnx import optimizer

print("TF version:{}".format(tf.__version__))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def converer_keras_to_onnx_v2(keras_path, outputs_layer=None, out_onnx=None, Optimize=True):
    '''
    :param keras_path: keras *.h5 files
    :param outputs_layer: default last layer
    :param out_tflite: output *.onnx file
    :return:
    '''
    model_dir = os.path.dirname(keras_path)
    model_name = os.path.basename(keras_path)[:-len(".h5")]
    # 加载keras模型, 结构打印
    model = tf.keras.models.load_model(keras_path)
    # model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf}, compile=False)
    model.summary()
    prefix = [model_name]
    if outputs_layer:
        # 从keras模型中提取层,转换成tflite
        model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(outputs_layer).output)
        model.summary()
        prefix += [outputs_layer]
    print(model.name)
    onnx_model = keras2onnx.convert_keras(model, model.name)
    # print(onnx_model)
    prefix = "_".join(prefix)
    if not out_onnx:
        out_onnx = os.path.join(model_dir, "{}.onnx".format(prefix))
    onnx.save_model(onnx_model, out_onnx)

    if Optimize:
        # 去掉identity层
        all_passes = optimizer.get_available_passes()
        print("Available optimization passes:")
        for p in all_passes:
            print('\t{}'.format(p))
        onnx_optimized = os.path.join(model_dir, "{}_optimized.onnx".format(prefix))
        passes = ['eliminate_identity']
        optimized_model = optimizer.optimize(onnx_model, passes)
        onnx.save(optimized_model, onnx_optimized)


def parse_args():
    # model_path = "/home/dm/panjinquan3/FaceDetector/tf-yolov3-detection/data/yolov3_simple0.5_320.h5"
    model_path = "data/yolov3_lite0.25_320_320_freeze_head.h5"
    # model_path = "/home/dm/panjinquan3/FaceDetector/tf-yolov3-detection/yolov3-micro.h5"
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--model_path", help="model_path", default=model_path, type=str)
    # parser.add_argument("--outputs_layer", help="outputs_layer", default='out_feature', type=str)
    parser.add_argument("--outputs_layer", help="outputs_layer", default=None, type=str)
    parser.add_argument("-o", "--out_onnx", help="out onnx model path", default=None, type=str)
    parser.add_argument("-opt", "--Optimize", help="Optimize model", default=False, type=bool)
    return parser.parse_args()


def get_model(model_path):
    if not model_path:
        model_path = os.path.join(os.getcwd(), "*.h5")
        model_list = glob.glob(model_path)
    else:
        model_list = [model_path]
    return model_list


if __name__ == '__main__':
    from utils import tf_tools

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf_tools.set_device_memory(0.9)
    args = parse_args()

    # outputs_layer = "fc1"
    outputs_layer = args.outputs_layer
    model_list = get_model(args.model_path)
    out_onnx = args.out_onnx
    Optimize = args.Optimize
    for model_path in model_list:
        converer_keras_to_onnx_v2(model_path, outputs_layer, out_onnx=out_onnx, Optimize=Optimize)
