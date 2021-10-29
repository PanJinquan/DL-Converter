# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: tf-yolov3-detection
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-07-14 09:59:37
# --------------------------------------------------------
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def convert_h5to_pb(keras_path, out_pb=None):
    if not os.path.exists(keras_path):
        raise Exception("Error:{}".format(keras_path))
    model_dir = os.path.dirname(keras_path)
    model_name = os.path.basename(keras_path)[:-len(".h5")]
    if not out_pb:
        out_pb = os.path.join(model_dir, "{}.pb".format(model_name))

    model = tf.keras.models.load_model(keras_path, compile=False, custom_objects={'tf': tf})
    # model = tf.keras.models.load_model(keras_path, compile=False)
    model.summary()
    full_model = tf.function(lambda Input: model(Input))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=model_dir,
                      name=out_pb,
                      as_text=False)
    print("successfully convert to PB done")
    print("save model at: {}".format(out_pb))


def convert_h5to_saved_model(keras_path, out_pb=None):
    if not os.path.exists(keras_path):
        raise Exception("Error:{}".format(keras_path))
    model_dir = os.path.dirname(keras_path)
    model_name = os.path.basename(keras_path)[:-len(".h5")]
    if not out_pb:
        out_pb = os.path.join(model_dir, "{}".format(model_name))

    model = tf.keras.models.load_model(keras_path, compile=False, custom_objects={'tf': tf})
    # model = tf.keras.models.load_model(keras_path, compile=False)
    model.summary()
    tf.saved_model.save(model, out_pb)
    print("successfully convert to save_model done")
    print("save model at: {}".format(out_pb))


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


def forward_pb_model(pb_path):
    # Load frozen graph using TensorFlow 1.x functions
    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    # Wrap frozen graph to ConcreteFunctions
    frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=["Input:0"],
                                    outputs=["Identity:0", "Identity_1:0"],
                                    print_graph=True)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)
    test_images = np.zeros(shape=(1, 320, 320, 3), dtype=np.float32)
    # Get predictions for test images
    predictions = frozen_func(Input=tf.constant(test_images))[0]

    # Print the prediction for the first image
    print("-" * 50)
    print("Example prediction reference:")
    print(predictions[0].numpy())


if __name__ == "__main__":
    keras_path = "/home/dm/panjinquan3/release/AIT/finger-keypoint-detection/data/pretrained/model_model_mobilenet_v2.h5"
    pb_path = None
    convert_h5to_pb(keras_path)
    convert_h5to_saved_model(keras_path)
    # forward_pb_model(pb_path)
