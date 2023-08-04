import matplotlib.pyplot as plt
import DataLoader as thk_dataLoader
import TrainModel as thk_trainModel
import ModelDef as tnk_modelDef

# 与full_impl相比，不同的是
# main
# Accumulator
# ModelDef
# 三个文件，其他文件均相同

# 定义批量大小
batch_size = 256

# 定义学习速率
lr = 0.1

# 定义学习终点
num_epochs = 10

model = tnk_modelDef.ModelDef(batch_size, lr, num_epochs)

# 导入训练数据
train_iter, test_iter = thk_dataLoader.load_data_fashion_mnist(model.batch_size)

# 执行深度学习
thk_trainModel.train_ch3(model, train_iter, test_iter)

# 用训练结果进行预测，展示结果
thk_trainModel.predict_ch3(model, test_iter)
plt.show()
