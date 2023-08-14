import torch
from torch import nn


class ModelDef:
    def __init__(self, batch_size, lr, num_epochs):

        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs

        # 定义网络
        input_shape = 1
        # 不设置偏置，因为我们已经在多项式中实现了它
        self.net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))

        # 定义损失函数
        self.loss = nn.MSELoss(reduction='none')

        # 定义优化器
        self.updater = torch.optim.SGD(self.net.parameters(), self.lr)
