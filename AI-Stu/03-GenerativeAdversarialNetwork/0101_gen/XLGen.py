# 生成器

import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader

from XLDataSet import XLDataSet
from XLDisc import XLDisc
from ShowDict import showDict


class XLGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.netWork = nn.Sequential(
            nn.Linear(1, 2),
            nn.Sigmoid(),
            nn.Linear(2, 4),
            nn.Sigmoid(),
            nn.Linear(4, 8),
            nn.Sigmoid(),
        )
        self.netWork.to(torch.cuda.current_device())
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self, X):
        X = X.to(device='cuda', dtype=torch.float)
        return self.netWork.forward(X)

    # X 输入的反向生成值，float
    # y 全1张量，训练过程需要惩罚生成的序列不是0101模式，即鉴别器输出0的模式
    #   所以生成输出越接近1生成器越优，反之越差
    #   所以y设置为全1，惩罚鉴别器输出0
    def trainModel(self, Desc, X, y):
        X = X.to(device='cuda', dtype=torch.float)
        y = y.to(device='cuda', dtype=torch.float)

        self.netWork.train()
        Desc.netWork.train()
        # 通过值反向生成序列
        y_hat = self(X)
        # 鉴别器辨别序列
        desc = Desc(y_hat)
        # 计算鉴别器损失
        dLoss = Desc.lossFunc(desc, y)
        self.optimiser.zero_grad()
        # 由于dLoss的计算图中包含鉴别器的forward，所以backward的时候会导致鉴别器中的模型梯度重新计算
        # 但是由于optimiser里面不包含鉴别器的模型参数，所以这些参数不会被更新
        dLoss.backward()
        self.optimiser.step()


def checkSampleGPU(sample):
    for index in range(sample.size(0)):
        if sample[index] <= 1.1 and sample[index] >= 0.9:
            sample[index] = 1
        if sample[index] <= 0.1 and sample[index] >= -0.1:
            sample[index] = 0
    std = torch.FloatTensor([0., 1., 0., 1., 0., 1., 0., 1.]).to(device='cuda', dtype=torch.float)
    return torch.equal(std, sample)


def testTrain():
    dictTrainRecords = {}
    Disc = XLDisc()
    Gen = XLGen()

    currentTrainSize = 100
    # 生成器伪造数据的随机种子恒定
    # 这个种子也用于生成器训练
    g_train_data_seed = torch.rand(currentTrainSize, 1)

    for index in range(1000):
        epochNum = 10
        # 先用 currentTrainSize 个样本训练鉴别器，训练 epochNum/2 轮
        train_data_set = XLDataSet(currentTrainSize, 0.5)
        train_loader = DataLoader(train_data_set, batch_size=(currentTrainSize))
        Disc.trainModel(int(epochNum / 2), train_loader)
        # 再用 currentTrainSize 个生成样本训练鉴别器，训练 epochNum/2 轮
        g_train_data_set = Gen(g_train_data_seed).detach()
        g_train_loader = [
            (
                g_train_data_set,
                torch.full((currentTrainSize,), 0, dtype=torch.float)
            )
        ]
        Disc.trainModel(int(epochNum / 2), g_train_loader)
        # 最后用 currentTrainSize 个样本训练生成器，训练 epochNum 轮
        for epoch_index in range(epochNum):
            Gen.trainModel(Disc, g_train_data_seed,
                           torch.full((currentTrainSize, 1), 1, dtype=torch.float)
                           )
        # 每次博弈完成，输出生成器效率
        Gen.netWork.eval()
        with torch.no_grad():
            gen_test_data = Gen(torch.rand(100, 1)).detach()
            okSum = 0
            for indexGen in range(gen_test_data.size(0)):
                xl = gen_test_data[indexGen]
                if checkSampleGPU(xl):
                    okSum += 1
            okRate = okSum / gen_test_data.size(0)
            print("index-->", index, " || current rate-->", okRate)
            dictTrainRecords[index] = [okRate]
    showDict("XLGen", dictTrainRecords, 'epochTimes', ['okRate'])


testTrain()
