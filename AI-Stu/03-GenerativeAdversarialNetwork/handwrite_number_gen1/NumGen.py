# 生成器

import torch.nn as nn
import torch.optim


class NumGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.netWork = nn.Sequential(
            nn.Linear(1, 200),
            nn.Sigmoid(),
            nn.Linear(200, 28 * 28),
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
        # print('dloss-->', dLoss.item())
        self.optimiser.zero_grad()
        # 由于dLoss的计算图中包含鉴别器的forward，所以backward的时候会导致鉴别器中的模型梯度重新计算
        # 但是由于optimiser里面不包含鉴别器的模型参数，所以这些参数不会被更新
        dLoss.backward()
        self.optimiser.step()
