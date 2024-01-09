from ModelDesign import DynamicLRModelDef as MD
from torch import nn
import torch
import torch.optim.lr_scheduler as lr_scheduler


class LessLevelModelDef(MD.DynamicLRModelDef):
    def __init__(self, batch_size=0, learning_rate=0, num_epochs=0):
        super().__init__(batch_size, learning_rate, num_epochs)
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.Linear(10000, 5000),
                                 nn.ReLU(),
                                 nn.Linear(5000, 2500),
                                 nn.ReLU(),
                                 nn.Linear(2500, 1250),
                                 nn.ReLU(),
                                 nn.Linear(1250, 625),
                                 nn.ReLU(),
                                 nn.Linear(625, 5)
                                 )
        # 定义模型初始化方法（即规定学习起点）
        self.net.apply(LessLevelModelDef.init_W)
        # 定义损失函数（即固定交叉熵损失函数为模型损失函数，reduction='none' 意味着不对损失进行任何聚合操作，而是返回每个样本的独立损失值，保持了每个样本的损失信息。）
        self.loss = nn.CrossEntropyLoss(reduction='none')
        # 定义优化器
        self.updater = torch.optim.SGD(self.net.parameters(), self.learning_rate)
        # 每5个epoch学习率乘以0.5
        self.learning_rate_scheduler = lr_scheduler.StepLR(self.updater, step_size=5, gamma=0.5)
