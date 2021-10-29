# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import tensorflow as tf
import numpy as np


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


# tf-api nms
def py_cpu_nms_tf_api(dets_tf, thresh):
    dets_tf = tf.cast(dets_tf, tf.float32)
    x1 = dets_tf[:, 0]
    y1 = dets_tf[:, 1]
    x2 = dets_tf[:, 2]
    y2 = dets_tf[:, 3]
    scores = dets_tf[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    top_nn = tf.shape(scores)[0]
    scores_topn, order = tf.nn.top_k(scores, top_nn, sorted=True)

    def my_cond(loop_i, order_input, result):
        cur_order_len = tf.shape(order_input)[0]
        flag = tf.cond(tf.equal(cur_order_len, 0), lambda: False, lambda: True)
        return flag

    def my_body(loop_i, order_input, tmp_tf):
        i = order_input[0]
        i = tf.cast(i, tf.int32)
        xx1 = tf.maximum(x1[i], tf.gather(x1, order_input[1:]))
        yy1 = tf.maximum(y1[i], tf.gather(y1, order_input[1:]))
        xx2 = tf.minimum(x2[i], tf.gather(x2, order_input[1:]))
        yy2 = tf.minimum(y2[i], tf.gather(y2, order_input[1:]))
        w = tf.maximum(0.0, xx2 - xx1 + 1)
        h = tf.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + tf.gather(areas, order_input[1:]) - inter)  # iou
        inds = tf.where(ovr <= thresh)
        inds = tf.reshape(inds, [-1])
        order_input = tf.gather(order_input, inds + 1)
        result = tf.concat([tmp_tf, tf.reshape(i, [1])], 0)

        return loop_i + 1, order_input, result

    ii = tf.constant(0)
    tmp = tf.constant(0, shape=[1])
    _, _, tmp_result = tf.while_loop(cond=my_cond,
                                     body=my_body,
                                     loop_vars=[ii, order, tmp],
                                     shape_invariants=[tf.TensorShape(None), tf.TensorShape(None),
                                                       tf.TensorShape(None)])

    keep = tmp_result[1:]
    keep = tf.reshape(keep, [-1])
    return keep


def per_class_nms(boxes, scores, prob_threshold, iou_threshold, top_k=200, keep_top_k=100):
    """
    :param boxes: (num_boxes, 4)
    :param scores:(num_boxes,)
    :param landms:(num_boxes, 10)
    :param prob_threshold:
    :param iou_threshold:
    :param top_k: keep top_k results. If k <= 0, keep all the results.
    :param keep_top_k: keep_top_k<=top_k
    :return: dets:shape=(num_bboxes,5),[xmin,ymin,xmax,ymax,scores]
             landms:(num_bboxes,10),[x0,y0,x1,y1,...,x4,y4]
    """
    # ignore low scores
    inds = np.where(scores > prob_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]
    # keep top-K before NMS
    if top_k >= 0:
        order = scores.argsort()[::-1][:top_k]
    else:
        order = scores.argsort()[::-1]
    boxes = boxes[order]
    scores = scores[order]
    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, iou_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    return dets


def bboxes_nms(input_bboxes, input_scores, prob_threshold, iou_threshold, top_k=200, keep_top_k=100):
    """
    :param input_bboxes: (num_boxes, 4)
    :param input_scores: (num_boxes,num_class)
    :param prob_threshold:
    :param iou_threshold:
    :param top_k: keep top_k results. If k <= 0, keep all the results.
    :param keep_top_k: keep_top_k<=top_k
    :return: dets:shape=(num_bboxes,5),[xmin,ymin,xmax,ymax,scores]
    """
    if not isinstance(input_bboxes, np.ndarray):
        input_bboxes = np.asarray(input_bboxes)
    if not isinstance(input_scores, np.ndarray):
        input_scores = np.asarray(input_scores)

    picked_boxes_probs = []
    picked_labels = []
    for class_index in range(0, input_scores.shape[1]):
        probs = input_scores[:, class_index]
        index = probs > prob_threshold
        subset_probs = probs[index]
        if probs.shape[0] == 0:
            continue
        subset_boxes = input_bboxes[index, :]
        sub_boxes_probs = per_class_nms(subset_boxes,
                                        subset_probs,
                                        prob_threshold=prob_threshold,
                                        iou_threshold=iou_threshold,
                                        top_k=top_k,
                                        keep_top_k=keep_top_k)
        picked_boxes_probs.append(sub_boxes_probs)
        picked_labels += [class_index] * sub_boxes_probs.shape[0]

    if len(picked_boxes_probs) == 0:
        return np.asarray([]), np.asarray([]), np.asarray([])
    picked_boxes_probs = np.concatenate(picked_boxes_probs)
    boxes = picked_boxes_probs[:, :4]
    probs = picked_boxes_probs[:, 4]
    labels = np.asarray(picked_labels)
    return boxes, labels, probs
