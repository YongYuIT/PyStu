from ModelDesign import DynamicLRModelDef as MD
from torch import nn
import torch


class MoreLevelModelDef(MD.DynamicLRModelDef):
    def __init__(self, batch_size=0, learning_rate=0, num_epochs=0):
        super().__init__(batch_size, learning_rate, num_epochs)
        self.net = nn.Sequential()
        self.net.append(nn.Flatten())
        levels = getNumOfLevels(10000, 1.2)
        for index in range(len(levels)):
            levelInfo = levels[index]
            self.net.append(nn.Linear(levelInfo[0], levelInfo[1]))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(levels[len(levels) - 1][1], 5))
        # 定义模型初始化方法（即规定学习起点）
        self.net.apply(MoreLevelModelDef.init_W)
        # 定义损失函数（即固定交叉熵损失函数为模型损失函数，reduction='none' 意味着不对损失进行任何聚合操作，而是返回每个样本的独立损失值，保持了每个样本的损失信息。）
        self.loss = nn.CrossEntropyLoss(reduction='none')
        # 定义优化器
        self.updater = torch.optim.SGD(self.net.parameters(), self.learning_rate)


def getNumOfLevels(allFeatures, subSpeed):
    squ = []
    while allFeatures > 5:
        squ.append(allFeatures)
        allFeatures = int(allFeatures / subSpeed)
    level = []
    for index in range(len(squ) - 1):
        level.append([squ[index], squ[index + 1]])
    return level


def test():
    print(getNumOfLevels(10000, 1.5))
