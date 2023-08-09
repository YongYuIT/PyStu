import torch
from torch import nn


class ModelDef:
    def __init__(self, batch_size, lr, num_epochs):
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs

        # 定义网络
        # PyTorch不会隐式地调整输入的形状。因此，
        # 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.Linear(784, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 10)
                                 )
        self.net.apply(ModelDef.init_weights)

        # 定义损失函数
        self.loss = nn.CrossEntropyLoss(reduction='none')

        # 定义优化器
        self.updater = torch.optim.SGD(self.net.parameters(), self.lr)

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
