from typing import Optional, List

import torch
from torch import nn
from torch.nn import functional as F

from NumGanModel4 import NumGanModel4


class Interpolate(nn.Module, ):
    def __init__(self, size: Optional[int] = None, scale_factor: Optional[List[float]] = None, mode: str = 'nearest',
                 align_corners: Optional[bool] = None, recompute_scale_factor: Optional[bool] = None,
                 antialias: bool = False):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias

    def forward(self, inputTensor):
        return F.interpolate(inputTensor, self.size, self.scale_factor, self.mode, self.align_corners,
                             self.recompute_scale_factor, self.antialias)


class NumGanModel5(NumGanModel4):
    def __init__(self):
        super().__init__()
        # 定义判别器模型
        self.DiscModel = nn.Sequential(
            # 输入的是n*1*28*28的张量，即n张1通道28*28的图片
            # 参考handwrite_number_gen项目中，LeNetModelDef的网络架构
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
            nn.Sigmoid(),
        )
        # 定义生成器模型
        self.GenModel = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Unflatten(1, (16, 5, 5)),
            Interpolate(scale_factor=2, mode='nearest'),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 6, kernel_size=5, padding=0),
            nn.ReLU(),
            Interpolate(scale_factor=2, mode='nearest'),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 1, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )
        # 将两个模型都移动到GPU执行
        self.DiscModel.to(torch.cuda.current_device())
        self.GenModel.to(torch.cuda.current_device())
        # 定义模型的Loss函数，由于GAN自始至终仅用到判别器的Loss函数，所以只需定义判别器的Loss，无需定义生成器的Loss
        # 改进采用二元交叉熵损失函数
        self.DiscLoss = nn.BCELoss()
        # 定义两个模型的优化器
        self.DiscOptimiser = torch.optim.Adam(self.DiscModel.parameters(), lr=0.0001)
        self.GenOptimiser = torch.optim.Adam(self.GenModel.parameters(), lr=0.0001)
