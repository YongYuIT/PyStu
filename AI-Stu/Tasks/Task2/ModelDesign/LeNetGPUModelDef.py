from ModelDesign import LeNetModelDef as MD
from torch import nn
import torch
import torch.optim.lr_scheduler as lr_scheduler


def checkCUDA():
    print(torch.cuda.current_device())  # 当前使用的 CUDA 设备序号
    print(torch.cuda.device_count())  # 可用的 CUDA 设备数量
    print(torch.cuda.get_device_name(0))  # 第一个 CUDA 设备的名称


def test():
    print("test1:", torch.device('cpu').type)
    print("test2:", torch.device('cuda').type)
    print("test3:", torch.device('cuda:1').type)
    print("test4:", torch.device('cuda:2').type)
    print("test5:", torch.cuda.device_count())


class LeNetGPUModelDef(MD.LeNetModelDef):
    def __init__(self, batch_size=0, learning_rate=0, num_epochs=0):
        super().__init__(batch_size, learning_rate, num_epochs)

        self.net = self.net.to(torch.cuda.current_device())

        # 定义模型初始化方法（即规定学习起点）
        self.net.apply(LeNetGPUModelDef.init_W)
        # 定义损失函数（即固定交叉熵损失函数为模型损失函数，reduction='none' 意味着不对损失进行任何聚合操作，而是返回每个样本的独立损失值，保持了每个样本的损失信息。）
        self.loss = nn.CrossEntropyLoss(reduction='none')
        # 定义优化器
        self.updater = torch.optim.SGD(self.net.parameters(), self.learning_rate)
        # 每5个epoch学习率乘以0.5
        self.learning_rate_scheduler = lr_scheduler.StepLR(self.updater, step_size=5, gamma=0.5)

    # 单独的一轮epoch
    def train_epoch(self, train_iter):
        # 将模型设置为训练模式
        self.net.train()
        for y, X in train_iter:
            X = X.to('cuda')
            y = y.to('cuda')
            y_hat = self.net(X)
            # 前向传播，计算loss
            loss = self.loss(y_hat, y)
            # 将梯度矩阵置0，避免梯度积累
            self.updater.zero_grad()
            # 后向传播，记录grad
            loss.mean().backward()
            # 更新模型参数
            self.updater.step()
        print("do epoch on GPU-->", len(train_iter))

    # 模型评估，返回正确识别的百分比
    def evaluate(self, test_iter):
        self.net.eval()  # 将模型设置为评估模式
        with torch.no_grad():
            totalSamples = 0
            EqualSamples = 0
            totalLoss = 0
            for y, X in test_iter:
                X = X.to('cuda')
                y = y.to('cuda')
                y_hat = self.net(X)
                y_hat_max = y_hat.argmax(axis=1)
                y_max = y.argmax(axis=1)
                Equal = torch.sum(y_hat_max == y_max).item()
                totalSamples += len(y)
                EqualSamples += Equal
                loss = self.loss(y_hat, y)
                totalLoss = loss.sum()
        print("do evaluate on GPU-->", len(test_iter))
        return EqualSamples / totalSamples, totalLoss / totalSamples
