import sys

sys.path.append('../../Task1')
from ModelDesign import DynamicLRModelDef as MD
from torch import nn
import torch
import torch.optim.lr_scheduler as lr_scheduler


class LeNetModelDef(MD.DynamicLRModelDef):
    def __init__(self, batch_size=0, learning_rate=0, num_epochs=0):
        super().__init__(batch_size, learning_rate, num_epochs)

        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(16, 24, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(512, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 5),
        )

        # 定义模型初始化方法（即规定学习起点）
        self.net.apply(LeNetModelDef.init_W)
        # 定义损失函数（即固定交叉熵损失函数为模型损失函数，reduction='none' 意味着不对损失进行任何聚合操作，而是返回每个样本的独立损失值，保持了每个样本的损失信息。）
        self.loss = nn.CrossEntropyLoss(reduction='none')
        # 定义优化器
        self.updater = torch.optim.SGD(self.net.parameters(), self.learning_rate)
        # 每5个epoch学习率乘以0.5
        self.learning_rate_scheduler = lr_scheduler.StepLR(self.updater, step_size=5, gamma=0.5)
