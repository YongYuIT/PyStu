import matplotlib.pyplot as plt
import DataLoader as thk_dataLoader
import TrainModel as thk_trainModel
import ModelDef as tnk_modelDef

# 正常拟合---------------------------------------------------------------------------------

# 定义批量大小
batch_size = 10

# 定义学习速率
lr = 0.01

# 定义学习终点
num_epochs = 400

# 设置input层形状（features数目）
num_features = 4

model = tnk_modelDef.ModelDef(batch_size, lr, num_epochs, num_features)

# 导入训练数据
train_iter, test_iter = thk_dataLoader.load_data(model)

# 执行深度学习
thk_trainModel.train_ch3(model, train_iter, test_iter)

plt.show()

# 欠拟合---------------------------------------------------------------------------------


# 定义批量大小
batch_size = 10

# 定义学习速率
lr = 0.01

# 定义学习终点
num_epochs = 200

# 设置input层形状（features数目）
num_features = 2

model = tnk_modelDef.ModelDef(batch_size, lr, num_epochs, num_features)

# 导入训练数据
train_iter, test_iter = thk_dataLoader.load_data(model)

# 执行深度学习
thk_trainModel.train_ch3(model, train_iter, test_iter)

plt.show()

# 过拟合---------------------------------------------------------------------------------


# 定义批量大小
batch_size = 10

# 定义学习速率
lr = 0.01

# 定义学习终点
num_epochs = 1000

# 设置input层形状（features数目）
num_features = 8

model = tnk_modelDef.ModelDef(batch_size, lr, num_epochs, num_features)

# 导入训练数据
train_iter, test_iter = thk_dataLoader.load_data(model)

# 执行深度学习
thk_trainModel.train_ch3(model, train_iter, test_iter)

plt.show()