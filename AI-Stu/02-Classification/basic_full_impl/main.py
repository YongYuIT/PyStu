import torch
import torchvision
from torchvision import transforms

# 定义批量大小
batch_size = 256
# 导入训练数据
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)

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

# 计算预测正确的数量
# 例如y_hat=[[0.2,0.3,0.5],[0.1,0.9,0],[0.1,0.01,0.99]]
# y=[2,1,1]
# 经过 y_hat = y_hat.argmax(axis=1)
# y_hat=[2, 1, 2]
# 所以计算正确的有两个（前面两个），计算不正确的有一个（最后一个）
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())