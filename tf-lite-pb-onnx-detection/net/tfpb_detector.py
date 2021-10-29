# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: tf-object-detection
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-04-01 18:17:26
# --------------------------------------------------------
"""

import numpy as np
import cv2
import tensorflow as tf


class TFPBDetector(object):
    """ PB Detector"""

    def __init__(self, pb_path):
        """
        初始化模型
        :param pb_path:pb_path模型路径
        """
        # 加载模型权重
        self.pb_graph = self.load_weights(pb_path)
        self.input_details = self.pb_graph.inputs
        self.output_details = self.pb_graph.outputs
        # self.input_shape = self.input_details[0]['shape']
        # print("input_shape:{}".format(self.input_shape))
        print("input_details:{}".format(self.input_details))
        print("output_details:{}".format(self.output_details))

    def load_weights(self, pb_path):
        """
        :param pb_path:pb_path模型路径
        :return:
        """
        # Load frozen graph using TensorFlow 1.x functions
        with tf.io.gfile.GFile(pb_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            loaded = graph_def.ParseFromString(f.read())
        # Wrap frozen graph to ConcreteFunctions
        pb_graph = self.wrap_frozen_graph(graph_def=graph_def,
                                          inputs=["Input:0"],
                                          outputs=["Identity:0", "Identity_1:0"],
                                          print_graph=True)
        return pb_graph

    def __call__(self, input_data):
        """
        num_bboxes=1500
        num_class=1
        :param input_data: shape = (1, 320, 320, 3)
        :return: outputs[0] = bboxes = (1, num_bboxes, 4),bbox=[xmin,ymin,xmax,ymax]
                 outputs[1] = scores = (1, num_bboxes, num_class)

        """
        outputs = self.forward(input_data)
        return outputs

    def forward(self, input_data):
        """
        num_bboxes=1500
        num_class=1
        :param input_data: shape = (1, 320, 320, 3)
        :return: outputs[0] = bboxes = (1, num_bboxes, 4),bbox=[xmin,ymin,xmax,ymax]
                 outputs[1] = scores = (1, num_bboxes, num_class)

        """
        predictions = self.pb_graph(Input=tf.constant(input_data))
        # outputs = predictions[0].numpy()
        return predictions

    @staticmethod
    def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
        def _imports_graph_def():
            tf.compat.v1.import_graph_def(graph_def, name="")

        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
        import_graph = wrapped_import.graph

        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        if print_graph == True:
            for layer in layers:
                print(layer)
        print("-" * 50)
        return wrapped_import.prune(
            tf.nest.map_structure(import_graph.as_graph_element, inputs),
            tf.nest.map_structure(import_graph.as_graph_element, outputs))


if __name__ == "__main__":
    keras_path = "./yolov3-micro_freeze_head.h5"
    pb_path = "/home/dm/panjinquan3/FaceDetector/tf-yolov3-detection/utils/convert_tools/yolov3-micro_freeze_head.pb"
    # convert_h5to_pb(keras_path)
    pbd = TFPBDetector(pb_path)
    test_images = np.zeros(shape=(1, 320, 320, 3), dtype=np.float32)
    out = pbd(test_images)
    print(out)
