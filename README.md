# Convert-Tools

## Pytorch --> ONNX
- docs: https://pytorch.org/docs/stable/onnx.html
- torch.onnx: 将模型导出为ONNX格式。这个导出器运行你的模型一次，以获得其导出的执行轨迹; 目前，它不支持动态模型（例如，RNN）。
```python
torch.onnx.export(model, args, f, export_params=True, verbose=False, training=False)
"""
 --model模型（torch.nn.Module）:要导出的模型。
 --args（参数元组）:模型的输入，例如，这-model(*args)是模型的有效调用。任何非变量参数将被硬编码到导出的模型中; 任何变量参数都将成为输出模型的输入，按照它们在参数中出现的顺序。如果args是一个变量，这相当于用该变量的一个元组来调用它。（注意：将关键字参数传递给模型目前还不支持，如果需要，给我们留言。）
 --f 类文件对象（必须实现返回文件描述符的fileno）或包含文件名的字符串。一个二进制Protobuf将被写入这个文件。
 --export_params（布尔，默认为True）:如果指定，所有参数将被导出。如果要导出未经训练的模型，请将其设置为False。在这种情况下，导出的模型将首先将其所有参数作为参数，按照指定的顺序model.state_dict().values()
 --verbose（布尔，默认为False）: 如果指定，我们将打印出一个调试描述的导出轨迹。
 --training（布尔，默认为False）: 在训练模式下导出模型。目前，ONNX只是为了推导出口模型，所以你通常不需要将其设置为True。
"""

```


## onnx-simplifier
- 简化网络结构
- docs: https://github.com/daquexian/onnx-simplifier
- install: `pip3 install onnx-simplifier`
- eg : `python3 -m onnxsim input_onnx_model output_onnx_model`

```python
"""
positional arguments:
  input_model           Input ONNX model
  output_model          Output ONNX model
  check_n               Check whether the output is correct with n random
                        inputs

optional arguments:
  -h, --help            show this help message and exit
  --enable-fuse-bn      Enable ONNX fuse_bn_into_conv optimizer. In some cases
                        it causes incorrect model
                        (https://github.com/onnx/onnx/issues/2677).
  --skip-fuse-bn        This argument is deprecated. Fuse-bn has been skippped
                        by default
  --skip-optimization   Skip optimization of ONNX optimizers.
  --input-shape         INPUT_SHAPE [INPUT_SHAPE ...]
                        The manually-set static input shape, useful when the
                        input shape is dynamic. The value should be
                        "input_name:dim0,dim1,...,dimN" or simply
                        "dim0,dim1,...,dimN" when there is only one input, for
                        example, "data:1,3,224,224" or "1,3,224,224". Note:
                        you might want to use some visualization tools like
                        netron to make sure what the input name and dimension
                        ordering (NCHW or NHWC) is.

"""

```


## ONNX --> NCNN
- docs: https://github.com/Tencent/ncnn
- `onnx2ncnn $onnx_path  $ncnn_out".param" $ncnn_out".bin"`


## Pytorch2Keras
- docs   : https://github.com/nerox8664/pytorch2keras
- install: pip install pytorch2keras 

## darknet2onnx+darknet2pytorch+onnx2tensorflow
- https://github.com/Tianxiaomo/pytorch-YOLOv4/tree/master/tool

## 转换工具
- https://convertmodel.com/ 
