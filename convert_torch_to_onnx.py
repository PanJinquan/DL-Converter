"""
This code is used to convert the pytorch model into an onnx format model.
"""
import sys
import os
import torch.onnx
import demo
import onnx
from utils import torch_tools


def build_net(model_path, net_type, priors_type, input_size, freeze_header=False, device="cuda:0"):
    det = demo.Detector(model_path,
                        net_type=net_type,
                        priors_type=priors_type,
                        input_size=input_size,
                        freeze_header=freeze_header,
                        device=device)
    model = det.model
    return model


def convert2onnx(model_path,
                 net_type,
                 priors_type,
                 input_size,
                 num_classes=None,
                 device="cuda",
                 onnx_type="landm",
                 freeze_header=True):
    model = build_net(model_path, net_type, priors_type, input_size, freeze_header=freeze_header, device=device)
    model = model.to(device)
    model.eval()
    # model_name = os.path.basename(model_path)[:-len(".pth")]
    model_name = "_".join([net_type.lower(), priors_type.lower(), str(input_size[0]), str(input_size[1])])
    if freeze_header:
        model_name = model_name + "_freeze.onnx"
    else:
        model_name = model_name + ".onnx"
    onnx_path = os.path.join(os.path.dirname(model_path), model_name)
    # dummy_input = torch.randn(1, 3, 240, 320).to("cuda")
    dummy_input = torch.randn(1, 3, input_size[1], input_size[0]).to(device)
    if onnx_type == "default":
        torch.onnx.export(model, dummy_input, onnx_path,
                          verbose=False,
                          export_params=True)
    elif onnx_type == "det":
        torch.onnx.export(model,
                          dummy_input,
                          onnx_path,
                          export_params=True,
                          verbose=False,
                          input_names=['input'],
                          output_names=['boxes', 'scores'])
    elif onnx_type == "landm":
        torch.onnx.export(model,
                          dummy_input,
                          onnx_path,
                          export_params=True,
                          verbose=False,
                          input_names=['input'],
                          output_names=['boxes', 'scores', 'landm'])
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
    input_size = [320, 320]
    model_path ="/home/dm/data3/FaceDetector/torch-Slim-Detection-Landmark/work_space/RFB_landms_v2/RFB_landm1.0_face_320_320_wider_face_add_lm_10_10_dmai_data_FDDB_v2_ssd_20210624145405/model/best_model_RFB_landm_183_loss7.6508.pth"
    net_type = "RFB_landm"
    priors_type = "face"
    convert2onnx(model_path, net_type, priors_type, input_size, onnx_type="landm")
