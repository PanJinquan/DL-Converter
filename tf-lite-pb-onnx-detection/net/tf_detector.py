# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: tf-object-detection
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-04-01 18:17:26
# --------------------------------------------------------
"""
import os
import numpy as np
import cv2
import tensorflow as tf


class TFDetector(object):
    """ saved_model or *.h5 Detector"""

    def __init__(self, model_path):
        """
        初始化模型
        :param model_path: saved_model.pb or *.h5 model
        """
        # 加载模型权重
        self.model = self.load_weights(model_path)

    def load_weights(self, model_path):
        """
        :param model_path: saved_model.pb or *.h5 model
        :return:
        """
        if os.path.exists(os.path.join(model_path, "saved_model.pb")):
            model = tf.saved_model.load(model_path)
        elif "h5" in model_path:
            model = tf.keras.models.load_model(model_path, compile=False)
        else:
            Exception("Error:{}".format(model_path))
        return model

    def __call__(self, image_tensor, training=False):
        """
        num_bboxes=1500
        num_class=1
        :param image_tensor: shape = (1, 320, 320, 3)
        :return: outputs[0] = bboxes = (1, num_bboxes, 4),bbox=[xmin,ymin,xmax,ymax]
                 outputs[1] = scores = (1, num_bboxes, num_class)

        """
        outputs = self.forward(image_tensor, training=False)
        return outputs

    def forward(self, image_tensor, training=False):
        """
        320*320 out_tensor[0].shape=(1, 10, 10, 3, 6)
                out_tensor[1].shape=(1, 20, 20, 3, 6)

        320*320 out_tensor[0].shape=(1, 8, 8, 3, 6))
                out_tensor[1].shape=(1, 16, 16, 3, 6))
        :param image_tensor:
        :return:
        """
        out_tensor = self.model(image_tensor, training=training)
        out_tensor = out_tensor.numpy()
        return out_tensor
