import matplotlib.pyplot as plt
import DataLoader as thk_dataLoader
import TrainModel as thk_trainModel
import ModelDef as tnk_modelDef

# 正常拟合---------------------------------------------------------------------------------

# 定义批量大小
batch_size = 5

# 定义学习速率
lr = 0.001

# 定义学习终点
num_epochs = 500

# 设置input层形状（features数目）
num_features = 200

# 设置正则化系数lambd=0
lambd = 0

# 导入训练数据
train_iter, test_iter = thk_dataLoader.load_data(2000, 0.1, num_features, batch_size)

model = tnk_modelDef.ModelDef(batch_size, lr, num_epochs, num_features, lambd)

# 执行深度学习
thk_trainModel.train_ch3(model, train_iter, test_iter)

plt.show()

####################################################################
# 训练数据集设置为200，正则化系数lambd=0的时候出现明显的过拟合
# 下面将正则化系数改成100
lambd = 10

model = tnk_modelDef.ModelDef(batch_size, lr, num_epochs, num_features, lambd)

# 执行深度学习
thk_trainModel.train_ch3(model, train_iter, test_iter)

plt.show()