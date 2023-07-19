import torch
from torch.utils import data
# nn是神经网络的缩写
from torch import nn


# 生成测试数据，跟basic_full_impl一样
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    print("X rand as")
    print(X.shape)
    y = torch.matmul(X, w) + b
    print("y=Xw+b as")
    print(y.shape)
    y += torch.normal(0, 0.01, y.shape)
    print("y=Xw+b+噪声 as")
    print(y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print("y向量化")
print(labels.shape)

# 读取数据集，采用更先进的DataLoader

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 尝试读取数据集的第一项
print(next(iter(data_iter)))

# 定义模型
# Linear(2,1)表示线性模型，输入两个feature，输出一个标量
# Sequential是一个模型容器，下面的代码表示这个模型只包含有一个2->1的线性模型
net = nn.Sequential(nn.Linear(2, 1))

# 初始化模型参数
# net[0]表示对第0层（也就是输入层）进行设定
# normal_是将张量的元素按照指定的正态分布进行随机初始化，后面两个参数分别是正态分布的均值和标准差
# fill_是将张量填充为设定的值
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 定义损失还是
loss = nn.MSELoss()

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练模型
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)