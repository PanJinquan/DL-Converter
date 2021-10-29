"""
This code is used to convert the pytorch model into an onnx format model.
"""
import sys
import os
import torch.onnx
import demo
import onnx
from models.nets import nets
from utils import torch_tools


def build_net(model_path, net_type, input_size, num_classes):
    """
    :param net_type:
    :param input_size:
    :param num_classes:
    :return:
    """
    model = nets.build_net(net_type,
                           input_size,
                           num_classes,
                           width_mult=1.0,
                           pretrained=False
                           )
    state_dict = torch_tools.load_state_dict(model_path, module=False)
    model.load_state_dict(state_dict)
    return model


def convert2onnx(model_path, net_type, input_size, num_classes, device="cuda:0", onnx_type="kp"):
    model = build_net(model_path,net_type, input_size, num_classes)
    model = model.to(device)
    model.eval()
    model_name = os.path.basename(model_path).split(".")[0]
    onnx_path = os.path.join(os.path.dirname(model_path), model_name + ".onnx")
    # dummy_input = torch.randn(1, 3, 240, 320).to("cuda")
    dummy_input = torch.randn(1, 3, input_size[1], input_size[0]).to(device)
    # torch.onnx.export(model, dummy_input, onnx_path, verbose=False,
    #                   input_names=['input'],output_names=['scores', 'boxes'])
    if onnx_type == "default":
        torch.onnx.export(model, dummy_input, onnx_path, verbose=False, export_params=True)
    elif onnx_type == "det":
        torch.onnx.export(model,
                          dummy_input,
                          onnx_path,
                          export_params=True,
                          verbose=False,
                          input_names=['input'],
                          output_names=['scores', 'boxes', 'ldmks'])
    elif onnx_type == "kp":
        torch.onnx.export(model,
                          dummy_input,
                          onnx_path,
                          export_params=True,
                          verbose=False,
                          input_names=['input'],
                          output_names=['output'])
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(onnx_path)


if __name__ == "__main__":
    input_size = [112, 112]
    model_path = "/home/dm/data3/git_project/torch-image-classification-pipeline/work_space/sitting/best_model_064_82.8069.pth"
    net_type = "ir_mobilenet_v2"
    num_classes = 3
    convert2onnx(model_path, net_type, input_size, num_classes)
