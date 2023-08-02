import torch
import DataLoader as thk_dataLoader
import Accumulator as thk_accumulator
import Accuracy as thk_accuracy
import Animator as thk_animator
import ImageShow as thk_imageShow
import matplotlib.pyplot as plt

# 定义批量大小
batch_size = 256
# 导入训练数据
train_iter, test_iter = thk_dataLoader.load_data_fashion_mnist(batch_size)

# 由于待处理的图片都是28*28规格，此处将每个像素视为独立的feature，所以 features总数=28*28=784
num_inputs = 784
# 由于一共有10个分类，所以输出向量长度等于10
num_outputs = 10

# 随机选取一个学习起点
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


def net(X):
    # y=XW
    y = torch.matmul(X.reshape((-1, W.shape[0])), W)
    # y=XW+b
    y = y + b
    # y=sofrmax(XW+b)
    y = softmax(y)
    return y


# 这里需要定义一下样本对应的标签、预测值的表示方式
# 例如这里有三个样本A，B，C
# 对应3个预测值y_hat=[[0.1,0.1,0.9],[0.2,0.8,0],[1,0,0]]
# 本来这三个样本对应的标签形式上也应该跟y_hat相同，是一个3*3的张量
# 但是为了便于计算，标签使用索引表示，y=[2,1,0]
# 表示第一个样本其标签是第三个选项，第二个样本其标签是第二个选项，第二个样本其标签是第一个选项
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])


# 训练模型一个迭代周期（定义见第3章）
def train_epoch_ch3(net, train_iter, loss, updater):
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = thk_accumulator.Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), thk_accuracy.accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


# 训练模型（定义见第3章）
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  # @save
    animator = thk_animator.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                                     legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = thk_accumulator.evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

# 定义优化算法，SGD
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 定义学习速率
lr = 0.1
def updater(batch_size):
    return sgd([W, b], lr, batch_size)

# 定义学习终点
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

# 预测标签（定义见第3章）
def predict_ch3(net, test_iter, n=6):  #@save
    for X, y in test_iter:
        break
    trues = thk_dataLoader.get_fashion_mnist_labels(y)
    preds = thk_dataLoader.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    thk_imageShow.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
plt.show()