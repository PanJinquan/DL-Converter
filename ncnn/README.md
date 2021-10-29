# 1. Install NCNN
- 编译NCNN
```
   a. git clone -b 20200413 https://github.com/Tencent/ncnn.git
   b. cd ncnn
   c. mkdir build && cd build && cmake .. (根据需要修改编译选项)
   d. make && make install
   e. cp -r install/* ${PROJECT}/3rdparty/ncnn
```

# 2. NCNN量化之ncnn2table和ncnn2int8
NCNN的量化工具包含两部分，ncnn2table和ncnn2int8;
在进行ncnn2int8量化过程之前需要进行ncnn2table操作，生成量化表；
下面首先介绍量化表的生成步骤
#### （1）ncnn2table生成量化表
- １、首先准备工作，参考NCNN深度学习框架之Optimize优化器
- ２、终端进入ncnn/build/tools/quantize目录
- ３、example: 

```
./ncnn2table \
	--param=squeezenet-fp32.param \
	--bin=squeezenet-fp32.bin \
	--images=path/to/images/dir/  \
	--output=newModel.table \
	--mean=104.0,117.0,123.0 \
	--norm=1.0,1.0,1.0 \
	--size=224,224 \
	--swapRB \
	--thread=2
注：在执行命令时，在后面还可以添加mean, norm, size, thread参数，这边工具里已设为默认，就没有修改；
注：这里的image指的是图片集，并且是图片数量较多的图片集；
注：NCNN和Pytorch的mean和std的对应关系： 
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
const float mean[3] = {0.5 * 255.f, 0.5 * 255.f, 0.5 * 255.f};
const float std[3] = {1 / 0.5 / 255.f, 1 / 0.5 / 255.f, 1 / 0.5 / 255.f};
```
- ４、执行命令后，即可看见原文件目录下生成newModel.table的量化表

#### （2）ncnn2int8量化网络模型
- １、执行可执行文件ncnn2table，生成量化表
- ２、终端进入ncnn/build/tools/quantizw目录
- ３、./ncnn2int8 [inparam] [inbin] [outparam] [outbin] [calibration table]
- ４、执行命令后，即可在原文件目录下生成param和bin的输出文件(即进行int8量化后的ncnn模型)


#### （3）NCNN vulkan（GPU）
- 参考： https://github.com/nihui/ncnn-android-styletransfer
