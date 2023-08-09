import torch
import Relu as thk_relu


class ModelDef:
    def __init__(self, batch_size, W, b, W1, b1, lr, num_epochs):
        self.W = W
        self.b = b
        self.W1 = W1
        self.b1 = b1
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs

    @staticmethod
    def softmax(X):
        X_exp = torch.exp(X)
        partition = X_exp.sum(1, keepdim=True)
        return X_exp / partition

    # 这里需要定义一下样本对应的标签、预测值的表示方式
    # 例如这里有三个样本A，B，C
    # 对应3个预测值y_hat=[[0.1,0.1,0.9],[0.2,0.8,0],[1,0,0]]
    # 本来这三个样本对应的标签形式上也应该跟y_hat相同，是一个3*3的张量
    # 但是为了便于计算，标签使用索引表示，y=[2,1,0]
    # 表示第一个样本其标签是第三个选项，第二个样本其标签是第二个选项，第二个样本其标签是第一个选项
    @staticmethod
    def cross_entropy(y_hat, y):
        return - torch.log(y_hat[range(len(y_hat)), y])

    # 定义优化算法，SGD
    @staticmethod
    def sgd(W, b, W1, b1, lr, batch_size):
        params = [W, b, W1, b1]
        with torch.no_grad():
            for param in params:
                param -= lr * param.grad / batch_size
                param.grad.zero_()

    # 定义网络
    def net(self, X):
        # H=XW
        H = torch.matmul(X.reshape((-1, self.W.shape[0])), self.W)
        # H=XW+b
        H = H + self.b
        # H=Relu(XW+b)
        H = thk_relu.relu(H)
        # y=HW1+b1
        y = torch.matmul(H, self.W1) + self.b1
        # y=sofrmax(HW1+b1)
        y = ModelDef.softmax(y)
        return y

    # 定义优化器
    def updater(self):
        return self.sgd(self.W, self.b, self.W1, self.b1, self.lr, self.batch_size)

    def loss(self, y_hat, y):
        return ModelDef.cross_entropy(y_hat, y)
