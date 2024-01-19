import torch
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F


class LeNetModelDef(nn.Module):

    # 模型初始化
    @staticmethod
    def init_W(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    def __init__(self, learning_rate=0., num_epochs=0):
        super().__init__()
        # 保存超参
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        # 模型各层定义，_modules是一个字典对象
        self._modules['first_conv'] = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self._modules['first_active'] = nn.ReLU()
        self._modules['first_pool'] = nn.AvgPool2d(kernel_size=2, stride=2)
        self._modules['second_conv'] = nn.Conv2d(6, 16, kernel_size=5, padding=0)
        self._modules['second_active'] = nn.ReLU()
        self._modules['second_pool'] = nn.AvgPool2d(kernel_size=2, stride=2)
        self._modules['flatten'] = nn.Flatten()
        self._modules['first_full_conn'] = nn.Linear(400, 120)
        self._modules['first_fc_active'] = nn.ReLU()
        self._modules['second_full_conn'] = nn.Linear(120, 84)
        self._modules['second_fc_active'] = nn.ReLU()
        self._modules['third_full_conn'] = nn.Linear(84, 10)
        # 定义损失函数
        self.loss = nn.CrossEntropyLoss()

    # 由于这些参数跟具体运算设备相关，所以需要在运算设备确定之后才能初始化
    def initModel(self):
        # 定义模型初始化方法（即规定学习起点）
        self.apply(LeNetModelDef.init_W)
        # 定义优化器
        self.optimizer = torch.optim.SGD(self.parameters(), self.learning_rate)
        # 定义学习速率修改器
        self.learning_rate_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.8)

    def forward(self, X):
        X_1_1 = self._modules['first_conv'](X)
        X_1_2 = self._modules['first_active'](X_1_1)
        X_1_3 = self._modules['first_pool'](X_1_2)
        X_2_1 = self._modules['second_conv'](X_1_3)
        X_2_2 = self._modules['second_active'](X_2_1)
        X_2_3 = self._modules['second_pool'](X_2_2)
        X_F = self._modules['flatten'](X_2_3)
        X_FU_1 = self._modules['first_full_conn'](X_F)
        X_FU_1_A = self._modules['first_fc_active'](X_FU_1)
        X_FU_2 = self._modules['second_full_conn'](X_FU_1_A)
        X_FU_2_A = self._modules['second_fc_active'](X_FU_2)
        y = self._modules['third_full_conn'](X_FU_2_A)
        return y

    # 单独的一轮epoch
    def train_epoch(self, train_iter):
        current_lr = self.optimizer.param_groups[0]['lr']
        print("start epoch current learning rate:", current_lr)
        # 将模型设置为训练模式
        self.train()
        for X, y in train_iter:
            self.train_batch(X, y)
        # 更新学习速率
        self.learning_rate_scheduler.step()

    # 单独的一次随机小批量
    def train_batch(self, X, y):
        # 数据集中，标签y是[1,2,3...]这种，其中每个分量（一个整型值，如1）需要变成长度为10的以为张量（如[0,1,0,0,0,0,0,0,0,0]）
        # 这样才能使用交叉熵函数求loss
        y = F.one_hot(y, 10).float()
        # 前向传播，计算y的预测值
        y_hat = self(X)  # 等价于 y_hat = self.forward(X)
        # 根据计算出的预测值y_hat，结合标签y，计算loss
        loss = self.loss(y_hat, y)
        # 将梯度矩阵归零，避免梯度积累
        self.optimizer.zero_grad()
        # 后向传播，计算本次随机小批量的梯度
        loss.backward()
        # 根据计算的梯度矩阵，更新模型参数
        self.optimizer.step()

    # 模型评估，返回正确识别的百分比
    def evaluate_model(self, test_iter):
        # 将模型设置为评估模式
        self.eval()
        with torch.no_grad():
            # 初始化统计参数
            totalSamples = 0
            equalSamples = 0
            totalLoss = 0
            for X, y in test_iter:
                equal_num, loss = self.evaluate_batch(X, y)
                totalSamples += y.size(0)
                totalLoss += loss
                equalSamples += equal_num
        return totalLoss / totalSamples, equalSamples / totalSamples

    # 小批量评估
    def evaluate_batch(self, X, y):
        # 等价于 y_hat = self.forward(X)
        y_hat = self(X)
        # 计算测准样本数
        y_hat_max = y_hat.argmax(axis=1)
        equal_num = torch.sum(y_hat_max == y).item()
        # 计算loss
        y = F.one_hot(y, 10).float()
        loss = self.loss(y_hat, y).item()
        return equal_num, loss

    def train_model(self, train_iter, test_iter):
        dictTrainRecords = {}
        for epoch_index in range(self.num_epochs):
            self.train_epoch(train_iter)
            avgLoss, correct = self.evaluate_model(test_iter)
            print("epoch index-->", epoch_index, "||avgLoss-->", avgLoss, "||correct-->", correct)
            dictTrainRecords[epoch_index] = [avgLoss * 1000, correct]
        return dictTrainRecords

    def saveModel(self, saveName):
        torch.save(self.state_dict(), saveName)

    def loadModel(self, saveName):
        self.load_state_dict(torch.load(saveName))
