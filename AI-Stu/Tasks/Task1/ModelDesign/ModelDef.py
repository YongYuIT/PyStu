import torch
from torch import nn


class ModelDef:

    # 模型初始化
    @staticmethod
    def init_W(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    # 模型定义
    def __init__(self, batch_size, learning_rate, num_epochs):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.net = nn.Sequential(nn.Flatten(),
                                 nn.Linear(10000, 5000),
                                 nn.ReLU(),
                                 nn.Linear(5000, 2500),
                                 nn.ReLU(),
                                 nn.Linear(2500, 1250),
                                 nn.ReLU(),
                                 nn.Linear(1250, 625),
                                 nn.ReLU(),
                                 nn.Linear(625, 312),
                                 nn.ReLU(),
                                 nn.Linear(312, 156),
                                 nn.ReLU(),
                                 nn.Linear(156, 78),
                                 nn.ReLU(),
                                 nn.Linear(78, 40),
                                 nn.ReLU(),
                                 nn.Linear(40, 20),
                                 nn.ReLU(),
                                 nn.Linear(20, 10),
                                 nn.ReLU(),
                                 nn.Linear(10, 5),
                                 )
        # 定义模型初始化方法（即规定学习起点）
        self.net.apply(ModelDef.init_W)
        # 定义损失函数（即固定交叉熵损失函数为模型损失函数，reduction='none' 意味着不对损失进行任何聚合操作，而是返回每个样本的独立损失值，保持了每个样本的损失信息。）
        self.loss = nn.CrossEntropyLoss(reduction='none')
        # 定义优化器
        self.updater = torch.optim.SGD(self.net.parameters(), self.learning_rate)

    # 模型训练
    def train(self, train_iter, test_iter):
        num_layers = len(list(self.net.children()))
        print("current model has levels:", num_layers)
        dictTrainRecords = {}
        for epoch_index in range(self.num_epochs):
            self.train_epoch(train_iter)
            correct_rate = self.evaluate(test_iter)
            print("epoch index-->", epoch_index, "||correct rate-->", correct_rate)
            dictTrainRecords[epoch_index] = correct_rate
        return dictTrainRecords

    # 单独的一轮epoch
    def train_epoch(self, train_iter):
        # 将模型设置为训练模式
        self.net.train()
        for y, X in train_iter:
            y_hat = self.net(X)
            # 前向传播，计算loss
            loss = self.loss(y_hat, y)
            # 将梯度矩阵置0，避免梯度积累
            self.updater.zero_grad()
            # 后向传播，记录grad
            loss.mean().backward()
            # 更新模型参数
            self.updater.step()

    # 模型评估，返回正确识别的百分比
    def evaluate(self, test_iter):
        self.net.eval()  # 将模型设置为评估模式
        with torch.no_grad():
            totalSamples = 0
            notEqualSamples = 0
            for y, X in test_iter:
                y_hat = self.net(X)
                # print("y_hat-->", y_hat)
                # print("y-->", y)
                y_hat_max = y_hat.argmax(axis=1)
                y_max = y.argmax(axis=1)
                notEqual = torch.sum(y_hat_max != y_max).item()
                totalSamples += len(y)
                notEqualSamples += notEqual
        return 1 - (notEqualSamples / totalSamples)


def test():
    y_hat = torch.Tensor([[0.1, 0.1, 0.5, 0.1, 0.2], [0.9, 0, 0, 0, 0.1]])
    print(y_hat.argmax(axis=1))
