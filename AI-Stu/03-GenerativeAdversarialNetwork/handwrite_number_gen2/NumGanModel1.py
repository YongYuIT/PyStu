import torch
from torch import nn

from NumGanModel import NumGanModel


class NumGanModel1(NumGanModel):
    def __init__(self):
        super().__init__()
        # 定义判别器模型
        self.DiscModel = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 200),
            nn.LeakyReLU(0.02),
            nn.Linear(200, 1),
            nn.Sigmoid(),
        )
        # 定义生成器模型
        self.GenModel = nn.Sequential(
            nn.Linear(1, 200),
            nn.LeakyReLU(0.02),
            nn.Linear(200, 28 * 28),
            nn.Sigmoid(),
        )
        # 将两个模型都移动到GPU执行
        self.DiscModel.to(torch.cuda.current_device())
        self.GenModel.to(torch.cuda.current_device())
        # 定义两个模型的优化器
        self.DiscOptimiser = torch.optim.Adam(self.DiscModel.parameters(), lr=0.0001)
        self.GenOptimiser = torch.optim.Adam(self.GenModel.parameters(), lr=0.0001)
