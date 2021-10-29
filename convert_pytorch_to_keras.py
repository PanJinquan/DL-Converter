import os
import sys
import numpy as np
import tensorflow as tf
print("TF:{}".format(tf.__version__))

import numpy as np
import torch
import torch
import keras
from keras_preprocessing import image
from torch.autograd import Variable
from pytorch2keras import converter
from easydict import EasyDict as edict
from collections import OrderedDict
from configs import val_config
from keras.models import load_model
from models.nets.build_nets import build_nets
from utils import torch_tools


def torch2keras(model_path, input_size, out_keras_model=None, device="cpu"):
    """
    https://github.com/nerox8664/pytorch2keras
    pip install
     - tensorflow==2.3.0
     - keras==2.3.1
    =======================================================================
    To use the converter properly, please, make changes in your ~/.keras/keras.json:

    {
        "floatx": "float32",
        "epsilon": 1e-07,
        "backend": "tensorflow",
        "image_data_format": "channels_last"
    }
     =======================================================================
    :param model_path: torch model file
    :param input_size: torch model input_size
    :param out_keras_model: out_keras_model
    :param device: cpu
    :return:
    """

    if not os.path.exists(model_path):
        raise Exception("Error:{}".format(model_path))
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)
    if not out_keras_model:
        model_name = model_name[:-len(".pth")] + ".h5"
        out_keras_model = os.path.join(model_dir, model_name)

    # load torch Model
    t_model = build_model(net_type, model_path, config)

    # create random inputs datas
    np.random.seed(200)
    inputs = np.random.uniform(0, 1, (1, 3, input_size[1], input_size[0]))
    t_inputs = Variable(torch.FloatTensor(inputs)).to(device)
    k_inputs = inputs.transpose(0, 2, 3, 1)  # [B,C,H,W]-->[B,H,W,C]

    # forward torch
    t_model = t_model.to(device)
    t_model = t_model.eval()
    t_output = t_model(t_inputs)

    # convert torch weight to keras weight
    k_model = converter.pytorch_to_keras(model=t_model,
                                         args=t_inputs,
                                         input_shapes=[(3, input_size[1], input_size[0],)],
                                         verbose=True,
                                         change_ordering=True,  # change CHW to HWC
                                         )
    k_model.summary()
    # 保存模型
    k_model.save(out_keras_model)
    # 重新载入模型
    del k_model
    # load keras model
    k_model = tf.keras.models.load_model(out_keras_model)
    k_output = k_model(k_inputs, training=False)

    t_output = np.asarray(t_output.detach().numpy(), dtype=np.float32)
    k_output = np.asarray(k_output, dtype=np.float32).transpose(0, 3, 1, 2)
    # print("t_output:{}".format(t_output.shape))
    # print("k_output:{}".format(k_output.shape))
    print("t_output:{},{}".format(t_output.shape, t_output[0, 0, :, :]))
    print("k_output:{},{}".format(k_output.shape, k_output[0, 0, :, :]))
    print("successfully convert to keras model")
    print("torch model at: {}".format(model_path))
    print("save  model at: {}".format(out_keras_model))


def build_model(net_type, model_path, config):
    """
    build model
    :param net_type:
    :param model_path:
    :return:
    """
    model = build_nets(net_type=net_type, config=config, is_train=False)
    state_dict = torch_tools.load_state_dict(model_path, module=False)
    model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    config = val_config.custom_coco_finger4_model_mbv2_192_256
    config = edict(config)
    input_size = tuple(config.MODEL.IMAGE_SIZE)  # w,h
    net_type = config.MODEL.NAME
    # model_path = config.TEST.MODEL_FILE
    model_path = "../../work_space/finger4/custom_coco/model_mobilenet_v2_1.0_256x192_0.001_finger_2020-10-16-17-32/model/best_model_140_0.9306.pth"
    torch2keras(model_path, input_size)
