import torch
from torch import nn


class ModelDef:
    # 新增正则化系数lambd
    def __init__(self, batch_size, lr, num_epochs, num_features, lambd):
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.num_features = num_features
        self.lambd = lambd

        # 定义网络
        # 不设置偏置
        self.net = nn.Sequential(nn.Linear(self.num_features, 1))

        # 定义损失函数
        self.loss = nn.MSELoss(reduction='none')

        # 定义优化器，新增L2范数惩罚项
        self.updater = torch.optim.SGD(
            [{"params": self.net[0].weight, 'weight_decay': self.lambd}, {"params": self.net[0].bias}], lr=self.lr)
