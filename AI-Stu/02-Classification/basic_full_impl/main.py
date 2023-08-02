import matplotlib.pyplot as plt
import torch

import DataLoader as thk_dataLoader
import ModelDef as tnk_modelDef

# 定义批量大小
batch_size = 256

# 由于待处理的图片都是28*28规格，此处将每个像素视为独立的feature，所以 features总数=28*28=784
num_inputs = 784
# 由于一共有10个分类，所以输出向量长度等于10
num_outputs = 10

# 随机选取一个学习起点
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# 定义学习速率
lr = 0.1

# 定义学习终点
num_epochs = 10

model = tnk_modelDef.ModelDef(batch_size, W, b, lr, num_epochs)

# 导入训练数据
train_iter, test_iter = thk_dataLoader.load_data_fashion_mnist(model.batch_size)

# 执行深度学习，得到W和b
tnk_modelDef.train_ch3(model, train_iter, test_iter)

# 用训练结果（W和b）进行预测，展示结果
tnk_modelDef.predict_ch3(model, test_iter)
plt.show()
