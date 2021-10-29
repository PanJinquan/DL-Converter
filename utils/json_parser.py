# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: hook-circle-pytorch
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-11-25 18:25:37
# --------------------------------------------------------
"""
import numpy as np
from utils import file_processing


def get_annotation(json_path, class_dict=None):
    data = file_processing.read_json_data(json_path)
    anno_points = data["shapes"]
    bboxes, labels, points = get_anno_content(anno_points, class_dict=class_dict)
    bboxes = np.asarray(bboxes)
    labels = np.asarray(labels)
    points = np.asarray(points)
    return bboxes, labels, points


def points2bbox(keypoints):
    joints_bbox = []
    for joints in keypoints:
        joints = np.asarray(joints)
        xmin = min(joints[:, 0])
        ymin = min(joints[:, 1])
        xmax = max(joints[:, 0])
        ymax = max(joints[:, 1])
        joints_bbox.append([xmin, ymin, xmax, ymax])
    return joints_bbox


def get_anno_content(anno_points: dict, class_dict=None):
    dst_bbox = []
    dst_labels = []
    dst_points = []
    for item in anno_points:
        label = item["label"]
        if class_dict is None:
            points = np.asarray(item["points"])
            box = points2bbox([points])[0]
            dst_labels.append(label)
            dst_bbox.append(box)
            dst_points.append(points)
        elif label in class_dict:
            label = class_dict[label]
            points = np.asarray(item["points"])
            box = points2bbox([points])[0]
            dst_labels.append(label)
            dst_bbox.append(box)
            dst_points.append(points)
    return dst_bbox, dst_labels, dst_points


if __name__ == "__main__":
    json_path = "path/to/json"
    bboxes, gt_labels, points = get_annotation(json_path, class_dict=None)
