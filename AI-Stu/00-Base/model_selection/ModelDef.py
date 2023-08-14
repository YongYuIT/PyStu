import torch
from torch import nn


class ModelDef:
    def __init__(self, batch_size, lr, num_epochs, num_features):
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.num_features = num_features

        # 定义网络
        # 不设置偏置，因为我们已经在多项式中实现了它
        self.net = nn.Sequential(nn.Linear(self.num_features, 1, bias=False))

        # 定义损失函数
        self.loss = nn.MSELoss(reduction='none')

        # 定义优化器
        self.updater = torch.optim.SGD(self.net.parameters(), self.lr)
