#!/usr/bin/env bash
# https://github.com/Tencent/TNN/blob/master/doc/cn/user/convert.md
docker pull turandotkay/tnn-convert
# 重命名images
docker tag turandotkay/tnn-convert:latest tnn-convert:latest
docker rmi turandotkay/tnn-convert:latest
# 首先验证下 docker 镜像能够正常使用，首先我们通过下面的命令来看下 convert2tnn 的帮助信息：
docker run --rm -it tnn-convert:latest  python3 ./converter.py -h
#
#docker run --rm --volume=$(pwd):/opt/TNN/tools/onnx2tnn/onnx-converter/workspace  -it tnn-convert:latest  /bin/bash
#docker run --rm --volume=$(pwd):/opt/TNN/tools/convert2tnn/workspace  -it tnn-convert:latest  /bin/bash

