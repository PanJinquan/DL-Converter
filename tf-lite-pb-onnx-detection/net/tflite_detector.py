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
import tensorflow.lite as lite


class TFliteDetector(object):
    """ TFlite Detector"""

    def __init__(self, tflite_path):
        """
        初始化模型
        :param tflite_path:TF-lite模型路径
        """
        # 加载模型权重
        self.interpreter = self.load_weights(tflite_path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_dtype = self.input_details[0]['dtype']
        self.output_dtype = self.output_details[0]['dtype']
        self.input_shape = self.input_details[0]['shape']
        self.input_scale, self.input_zero_point = self.input_details[0]["quantization"]

        print("input_shape   :{}".format(self.input_shape))
        print("input_details :{},input_dtype :{}".format(self.input_details, self.input_dtype))
        print("output_details:{},output_dtype:{}".format(self.output_details, self.output_dtype))
        print("input_scale:{},input_zero_point:{}".format(self.input_scale, self.input_zero_point))

    def load_weights(self, tflite_path):
        """
        :param tflite_path:TF-lite模型路径
        :return:
        """
        interpreter = lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        return interpreter

    def __call__(self, image_tensor):
        """
        num_bboxes=1500
        num_class=1
        :param image_tensor: shape = (1, 320, 320, 3)
        :return: outputs[0] = bboxes = (1, num_bboxes, 4),bbox=[xmin,ymin,xmax,ymax]
                 outputs[1] = scores = (1, num_bboxes, num_class)

        """
        outputs = self.forward(image_tensor)
        return outputs

    def forward(self, image_tensor):
        """
        num_bboxes=1500
        num_class=1
        :param image_tensor: shape = (1, 320, 320, 3)
        :return: outputs[0] = bboxes = (1, num_bboxes, 4),bbox=[xmin,ymin,xmax,ymax]
                 outputs[1] = scores = (1, num_bboxes, num_class)

        """
        if self.input_dtype == np.uint8:
            image_tensor = image_tensor / self.input_scale + self.input_zero_point
        image_tensor = np.asarray(image_tensor).astype(self.input_dtype)
        self.interpreter.set_tensor(self.input_details[0]['index'], image_tensor)
        self.interpreter.invoke()
        outputs = []
        for node in self.output_details:
            output = self.interpreter.get_tensor(node['index'])
            if self.output_dtype == np.uint8:
                output = np.asarray(output, np.float32) / 255.0
            outputs.append(output)
        outputs = [outputs[1], outputs[0]]  # [scores,bboxes]->[bboxes,scores]
        return outputs
