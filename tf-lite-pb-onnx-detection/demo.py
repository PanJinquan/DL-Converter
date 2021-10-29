# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: tf-object-detection
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-04-01 18:17:26
# --------------------------------------------------------
"""
import sys
import os

sys.path.append(os.getcwd())
import cv2
import argparse
import numpy as np
import tensorflow as tf
from net import tflite_detector, tfpb_detector, tf_detector, onnx_detector
from utils.nms import py_bbox_nms
from utils import image_processing, file_processing, debug, tf_tools

print("TF:{}".format(tf.__version__))


def get_parser():
    # model_path = "data/yolov3_micro0.25_320_320_freeze_head.pb"
    # model_path = "data/yolov3_micro0.25_320_320_freeze_head_optimize.tflite"
    # model_path = "data/yolov3_lite0.25_320_320_freeze_head_optimize_float16.tflite"
    model_path = "data/yolov3_lite0.25_320_320_freeze_head.onnx"
    image_dir = "data/test_image"
    parser = argparse.ArgumentParser(description='detect_imgs')
    parser.add_argument('--model_path', default=model_path, type=str, help='model_path')
    parser.add_argument('--input_size', help="--input size 320", type=int, default=320)
    parser.add_argument('--prob_threshold', default=0.2, type=float, help='score threshold')
    parser.add_argument('--iou_threshold', default=0.3, type=float, help='iou_threshold')
    parser.add_argument('--image_dir', default=image_dir, type=str, help='directory or image path')
    args = parser.parse_args()
    return args


class YOLOLiteDetector():
    def __init__(self, model_path, input_size=320, prob_threshold=0.2, iou_threshold=0.3):
        """
        :param model_path: path/to/tflite model file
        :param input_size: model input size
        :param prob_threshold:
        :param iou_threshold:
        """
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.model = self.load_model(model_path=model_path)

    def load_model(self, model_path):
        if "tflite" in model_path:
            model = tflite_detector.TFliteDetector(tflite_path=model_path)
        elif "pb" in model_path:
            model = tfpb_detector.TFPBDetector(pb_path=model_path)
        elif "onnx" in model_path:
            model = onnx_detector.ONNXModel(onnx_path=model_path)
        elif os.path.exists(os.path.join(model_path, "saved_model.pb")):
            model = tf_detector.TFDetector(tf_path=model_path)
        else:
            Exception("Error:{}".format(model_path))
        return model

    @debug.run_time_decorator("forward")
    def forward(self, image_tensor):
        """
        num_bboxes=1500
        num_class=1
        :param image_tensor: shape = (1, 320, 320, 3)
        :return: outputs[0] = bboxes = (1, num_bboxes, 4),bbox=[xmin,ymin,xmax,ymax]
                 outputs[1] = scores = (1, num_bboxes, num_class)

        """
        out_tensor = self.model(image_tensor)
        return out_tensor

    @debug.run_time_decorator("predict")
    def predict(self, rgb_image):
        """
        :param rgb_image: input RGB image
        :return: preds_tensor=[boxes, scores]
        """
        image_tensor = self.pre_process(rgb_image)
        preds_tensor = self.forward(image_tensor)
        height, width, _ = rgb_image.shape
        boxes, labels, probs = self.post_process(preds_tensor, width, height,
                                                 prob_threshold=self.prob_threshold,
                                                 iou_threshold=self.iou_threshold)
        return boxes, labels, probs

    @debug.run_time_decorator("pre_process")
    def pre_process(self, image):
        """
        输入图像预处理:缩放到input_size,并归一化到[0,1]
        :param image:
        :return:
        """
        # use opencv to process image
        input_image = cv2.resize(image, dsize=(self.input_size, self.input_size))
        input_image = np.asarray(input_image / 255.0, dtype=np.float32)
        image_tensor = input_image[np.newaxis, :]

        # use tf.image to process image
        # image_tensor = tf.image.resize(image, (self.input_size, self.input_size))
        # image_tensor = image_tensor / 255
        # image_tensor = tf.expand_dims(image_tensor, 0)
        return image_tensor

    @debug.run_time_decorator("post_process")
    def post_process(self, preds_tensor, width, height, prob_threshold=0.2, iou_threshold=0.3, top_k=100):
        """
        NMS后处理,并将boxes映射为图像的真实坐标
        :param preds_tensor:
        :param width: orig image width
        :param height: orig image height
        :param prob_threshold:
        :param iou_threshold:
        :param top_k:  keep top_k results. If k <= 0, keep all the results.
        :return: boxes  : (num_boxes, 4),[xmin,ymin,xmax,ymax]
                 labels : (num_boxes, )
                 probs  : (num_boxes, )
        """
        boxes, scores = preds_tensor
        boxes, labels, probs = self.np_bboxes_nms(boxes,
                                                  scores,
                                                  prob_threshold=prob_threshold,
                                                  iou_threshold=iou_threshold,
                                                  top_k=top_k,
                                                  keep_top_k=top_k)
        boxes_scale = [width, height] * 2
        boxes = boxes * boxes_scale
        return boxes, labels, probs

    @staticmethod
    def np_bboxes_nms(input_bboxes, input_scores, prob_threshold, iou_threshold, top_k=200, keep_top_k=100):
        """
        :param input_bboxes: (num_boxes, 4)
        :param input_scores: (num_boxes,num_class)
        :param prob_threshold:
        :param iou_threshold:
        :param top_k: keep top_k results. If k <= 0, keep all the results.
        :param keep_top_k: keep_top_k<=top_k
        :return: boxes  : (num_boxes, 4),[xmin,ymin,xmax,ymax]
                 labels : (num_boxes, )
                 probs  : (num_boxes, )
        """
        if not isinstance(input_bboxes, np.ndarray):
            input_bboxes = np.asarray(input_bboxes)
        if not isinstance(input_scores, np.ndarray):
            input_scores = np.asarray(input_scores)
        input_bboxes = input_bboxes[0]
        input_scores = input_scores[0]
        boxes, labels, probs = py_bbox_nms.bboxes_nms(input_bboxes=input_bboxes,
                                                      input_scores=input_scores,
                                                      prob_threshold=prob_threshold,
                                                      iou_threshold=iou_threshold,
                                                      top_k=top_k,
                                                      keep_top_k=keep_top_k)
        return boxes, labels, probs

    def detect_image(self, rgb_image, isshow=True):
        """
        :param rgb_image:  input RGB Image
        :param isshow:
        :return:
        """
        boxes, labels, probs = self.predict(rgb_image)
        if isshow:
            print("boxes:{}\nlabels:{}\nprobs:{}".format(boxes, labels, probs))
            self.show_image(rgb_image, boxes, labels, probs)
        return boxes, labels, probs

    def infer(self, rgb_image, isshow=False):
        """
        :param rgb_image:
        :param isshow:
        :return: prediction:(num_bboxes, 5),[xmin,ymin,xmax,ymax,probs]
        """
        boxes, labels, probs = self.detect_image(rgb_image, isshow=isshow)
        labels = labels.reshape(-1, 1)
        probs = probs.reshape(-1, 1)
        prediction = np.hstack((boxes, probs))
        # prediction = np.hstack((boxes, labels, probs))
        return prediction

    def detect_image_dir(self, image_dir, isshow=True):
        """
        :param image_dir: directory or image file path
        :param isshow:<bool>
        :return:
        """
        if os.path.isdir(image_dir):
            image_list = file_processing.get_files_list(image_dir, postfix=["*.jpg", "*.png"])
        elif os.path.isfile(image_dir):
            image_list = [image_dir]
        else:
            raise Exception("Error:{}".format(image_dir))
        for img_path in image_list:
            orig_image = cv2.imread(img_path)
            rgb_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            # rgb_image = tf.image.decode_image(open(img_path, 'rb').read(), channels=3)
            # rgb_image = np.asarray(rgb_image)
            # boxes, labels, probs = self.detect_image(rgb_image, isshow=isshow)
            prediction = self.infer(rgb_image, isshow=isshow)

    def show_image(self, image, bboxes: np.asarray, classes, scores):
        """
        :param image: RGB image
        :param bboxes:<np.ndarray>: (num_bboxes, 4), box=[xmin,ymin,xmax,ymax]
        :param scores:<np.ndarray>: (num_bboxes,)
        :param classes:<np.ndarray>: (num_bboxes,)
        :return:
        """
        image = image_processing.draw_image_detection_bboxes(image, bboxes, scores, classes)
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Det", bgr_image)
        cv2.imwrite("result.jpg", bgr_image)
        cv2.waitKey(0)


if __name__ == "__main__":
    tf_tools.set_device_memory(0.90)
    args = get_parser()
    det = YOLOLiteDetector(model_path=args.model_path,
                           input_size=args.input_size,
                           prob_threshold=args.prob_threshold,
                           iou_threshold=args.iou_threshold)
    det.detect_image_dir(args.image_dir, isshow=True)
