# 整体目标

使用全链接判断图片里面的动物类型

# 数据准备（DataPrepare）

1. Spider-Stu\AI_Task1_DataPrepare项目分批下载几百张图片
    * 下载约200张小狗的图片存在dog文件夹下，每张图片生成唯一id并作为文件名命名图片，用一个文本文件记录这些id
    * 下载约200张猪的图片存在pig文件夹下，每张图片生成唯一id并作为文件名命名图片，用一个文本文件记录这些id
    * 下载约200张小鸟的图片存在bird文件夹下，每张图片生成唯一id并作为文件名命名图片，用一个文本文件记录这些id
    * 下载约200张美女的图片存在girl文件夹下，每张图片生成唯一id并作为文件名命名图片，用一个文本文件记录这些id
    * 下载约200张蛇的图片存在snake文件夹下，每张图片生成唯一id并作为文件名命名图片，用一个文本文件记录这些id
2. 手动剔除无关图片，得到原始数据

# 数据输入

## 数据预处理（DataPreprocessing）

1. 图片标准化：将图片扩展为1:1，不够的部分补上黑色
2. 图片大小一致：将图片缩放为100*100像素
3. 图片黑白化：将彩色图片转化为黑白图片
4. 图像增广：
    * 将图片左右颠倒
    * 将图片亮度进行1.1~1.4倍缩放

## 数据张量化存储（DataStorage）

参考：https://zh.d2l.ai/chapter_deep-learning-computation/read-write.html#id2

1. 将图像增广之后的图片命名为type_id_index，其中:
    * type为图片中的动物类型
    * id为图片的文件名
    * index为增广之后的图片的下标
2. 将 type_id_index --> 图片数据张量 字典借助 torch.save 保存为 pic_dict
3. 将来需要用到的时候，使用torch.load加载

# 模型训练

参考：AI-Stu\02-Classification\mlp_simple_impl

## 数据规范化读取（DataStdRead）

1. 继承 pytorch 里面的 Dataset，自定义数据库 ImgClassDataSet
2. 读取pic_dict字典里面的数据，将其转换为 ImgClassDataSet
3. 将数据集 ImgClassDataSet 按照8:2的比例随机划分成训练集和测试集
4. 使用 pytorch data.DataLoader 批量读取数据

## 模型设计（ModelDesign）

1. 采用全链接模型：
   * 展平层
   * 10000:5000 ReLU
   * 5000:2500 ReLU
   * ...（features减半）
   * 10:5

# 模型优化

## 原始状况（Start Status）

模型：ModelDef，全链接，每层Leatures减半

超参
~~~
batchSize = 100
learningRate = 0.1
numEpochs = 20
~~~



实验结果：
![StartStatus.png](OptRecords%2FStartStatus.png)

分析：准确度在低水平稳定徘徊，需要动态变动学习速率

## 优化1: 引入动态学习速率（DynamicLearningRate）

模型：DynamicLRModelDef

超参
~~~
batchSize = 100
learningRate = 0.5
numEpochs = 20
~~~
实验结果：
![DynamicLearningRate.png](OptRecords%2FDynamicLearningRate.png)

准确率没有明显改善

## 优化2：这种特征应该是比较多的features，减少模型输入端的features递减层数（LessLevel）

模型：LessLevelModelDef

超参
~~~
batchSize = 100
learningRate = 0.5
numEpochs = 60
~~~

实验结果：
![LessLevel.png](OptRecords%2FLessLevel.png)

这个结果还行

# 实战验证

## 保存和加载模型参数

参考：https://zh.d2l.ai/chapter_deep-learning-computation/read-write.html#id3


## 在验证集上验证模型（main_use）

* 从验证文件夹 pic_check 中读取图片
* 将图片进行标准化，包括：
  1. 图片标准化：将图片扩展为1:1，不够的部分补上黑色 
  2. 图片大小一致：将图片缩放为100*100像素 
  3. 图片黑白化：将彩色图片转化为黑白图片
* 使用模型进行预测，并显示结果

实际使用的准确率在70%左右（错误率16/50）