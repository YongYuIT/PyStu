# 整体目标

使用CNN判断图片里面的动物类型，与Task1中使用全链接的模型进行比较

# 训练资料准备（Data）

1. 使用Task1中的训练资料
2. 为了后续更好处理，将图片缩放从原来的100*100改为128*128（MainPreprocessing）
3. 为了识别更多细节，输入图片将不再转为灰度图，直接处理彩色图片（MainPreprocessing，ImagesSaveToTensers）
4. 为了增强泛化特性，将图片亮度增强范围从1~1.5改为0.5~1.5（ImageEnhanceToMany）

# 模型设计

参考：https://zh.d2l.ai/chapter_convolutional-neural-networks/lenet.html#lenet

参考LeNet实现

* 输入层，3 * 128 * 128的张量
* 第一卷积层，16 * 3 * 9 * 9卷积核（16通道输出，每通道3 * 9 * 9），步幅1，填充4，ReLU激活
  步幅移动次数=128+4-(9-4)+1=128
  所以卷积层输出为16 * 128 * 128的特征张量
  在此步骤中，需要学习的参数包括：
  16 * 3 * 9 * 9=3888 个卷积权重
  16个卷积核偏置
  一共3904个参数
* 第一汇聚层，4 * 4汇聚窗口，步幅4，最大汇聚
  步幅移动次数=128/4=32
  所以汇聚层输出为16 * 32 * 32的特征张量
* 第二卷积层，24 * 16 * 5 * 5卷积核，步幅1，填充2，ReLU激活
  步幅移动次数=32+2-(5-2)+1=32
  所以卷积层输出为24 * 32 * 32的特征张量
  此步骤中，需要学习的参数包括：
  24 * 16 * 5 * 5=9600 个卷积权重
  24个卷积核偏置
  一共9624个参数
* 第二汇聚层，4 * 4汇聚窗口，步幅4，最大汇聚
  步幅移动次数=32/4=8
  所以汇聚层输出为24 * 8 * 8的特征张量
* 第三卷积层，32 * 24 * 3 * 3卷积核，步幅1，填充1，ReLU激活
  步幅移动次数=8+1-(3-1)+1=8
  所以卷积层输出为32 * 8 * 8的特征张量
  此步骤中，需要学习的参数包括：
  32 * 24 * 3 * 3=6912 个卷积权重
  32个卷积核偏置
  一共9644个参数
* 第三汇聚层，2 * 2汇聚窗口，步幅2，最大汇聚
  步幅移动次数=8/2=4
  所以汇聚层输出为32 * 4 * 4的特征张量
* 展平层 将32 * 4 * 4的特征张量展开为512向量
* 第一全连接层，512-->200，ReLU激活
* 第二全连接层，200-->100，ReLU激活
* 第三全连接层，100-->50，ReLU激活
* 第四全连接层，50-->5，ReLU激活

全部模型一共512 * 200 + 200 * 100 +... 约十万个参数

对比LessLevelModel为10000 * 5000 + 5000 + 5000 * 2500 + 2500 + 2500 * 1250 + 1250... 约8千万个参数

超参
~~~
batchSize = 100
learningRate = 0.8
numEpochs = 60
~~~

实验结果：
![LeNetModelDef.png](OptRecords%2FLeNetModelDef.png)

看上去还行，比LessLevelModel收敛结果好，但是过程不够平滑

## 在验证集上验证模型（main_use）

验证集跟LessLevelModel一样

结果比LessLevelModel稍好一点，错误率14/50

## 在GPU上运算

在系统命令行中运行

~~~
nvidia-smi
~~~

可以看到显卡配置

执行代码

~~~
torch.cuda.device_count()
~~~

显示GPU数量为0，检查CUDA是否安装正确

在系统命令行中运行
~~~
nvcc --version
~~~

命令不识别，安装CUDA

CUDA（Compute Unified Device Architecture）是由 NVIDIA 开发的并行计算平台和编程模型。它允许开发者使用 NVIDIA 的 GPU（图形处理单元）进行通用目的的并行计算。

下载 cuda-toolkit

https://developer.nvidia.com/cuda-toolkit

在下载界面选择正确的操作系统类型、CPU架构、操作系统版本、安装类型之后，会自动启动下载

CUDA装好之后，nvcc --version也有了信息，运行torch.cuda.device_count()，输出GPU数量还是0

重新安装pytorch

https://pytorch.org/

在“INSTALL PYTORCH”选项中，选择正确的参数配置，自动获取安装命令

~~~
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
~~~

如果速度慢，切换国内的源

~~~
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c https://mirrors.aliyun.com/anaconda/cloud/pytorch -c https://mirrors.aliyun.com/anaconda/cloud/nvidia
~~~

在空白的python中执行这行命令

安装完成之后，再运行torch.cuda.device_count()，就看到输出了1，说明仅有的一个英伟达显卡有了

将模型放在GPU上训练发现速度快了很多

超参
~~~
batchSize = 400
learningRate = 0.8
numEpochs = 400
~~~

实验结果：
![LeNetGPUModelDef.png](OptRecords%2FLeNetGPUModelDef.png)


## 对训练集数据进行人工筛选，在高质量数据集上训练

~~~
picRootPath = "pic_check/"
~~~

超参
~~~
batchSize = 400
learningRate = 0.8
numEpochs = 400
~~~

实验结果：
![LeNetGPUModelDef-PicCheck.png](OptRecords%2FLeNetGPUModelDef-PicCheck.png)

在使用Spider-Stu\AI_Task1_DataPrepare\Step3_DownloadValidationData，从另一个数据源下载验证数据

通过批量验证，错误率在44.5%

# 分析两个模型泛化性能

* 是否过拟合

