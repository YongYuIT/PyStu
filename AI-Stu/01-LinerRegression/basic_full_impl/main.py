# 相当于 %matplotlib inline
# 设置Matplotlib图形库的显示方式
# 它会告诉解释器将Matplotlib的图形直接嵌入到Notebook中的输出单元格中，而不是弹出一个新的窗口显示图形。
# 这种方式被称为"内嵌显示"（inline display），它允许你在Notebook中直接看到绘制的图形，而无需切换到其他窗口
import matplotlib

matplotlib.use('TkAgg')

import torch
import matplotlib.pyplot as plt

import random


def synthetic_data(w, b, num_examples):
    # 生成y=Xw+b+噪声
    # torch.normal是一个用于生成服从正态分布（高斯分布）的随机数的函数
    # torch.normal(mean, std, size=None, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    # mean：正态分布的均值（期望值），填0说明关于y轴对称
    # std：正态分布的标准差（方差的平方根）
    # size：指定要生成的随机数的形状，可以是一个整数或元组。默认为None，表示生成一个单个随机数。
    #       填(num_examples, len(w))说明生成的生成的随机数是一个二维张量（即矩阵），行数是num_examples，列数是len(w)
    X = torch.normal(0, 1, (num_examples, len(w)))
    print("X rand as")
    print(X.shape)
    y = torch.matmul(X, w) + b
    print("y=Xw+b as")
    print(y.shape)
    y += torch.normal(0, 0.01, y.shape)
    print("y=Xw+b+噪声 as")
    print(y.shape)
    # y.reshape((-1, 1))表示将y转换成列向量（即数学意义上的向量）
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print("y向量化")
print(labels.shape)

print('features:', features[0], '\nlabel:', labels[0])

# detach：将张量当时的数值深度拷贝出来，张量后续的变化不会影响这份数值拷贝
# numpy：将张量转换成一个numpy数组
plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
plt.show()


# 读取数据集
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    # rang(0,1000,10)，从0开始，以10为步长，生成不超过1000的整数序列。0,10,20,...,980,990
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        # ("batch_indices-",i)
        # print(batch_indices)
        yield features[batch_indices], labels[batch_indices]


batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# 随机找个初始点
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# 定义模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b


# 定义损失函数，MSE
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 定义优化算法，SGD
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
