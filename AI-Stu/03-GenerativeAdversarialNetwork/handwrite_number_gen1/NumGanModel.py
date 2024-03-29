# nn.Module.eval()：关闭模型梯度计算
# nn.Module.train()：打开模型梯度计算
# torch.no_gard：临时关闭梯度计算，出了代码块模型梯度计算自动恢复
# nn.Module.detach()：返回一个当前模型的副本，这个副本了关闭梯度计算
# nn.Module.parameters().requires_grad=False：模型部分参数关闭梯度计算
# 1、为什么nn.Module.eval()表示模型梯度不会被记录，还需要配合torch.no_gard来使用？
# nn.Module.eval()除了关闭模型梯度计算外，还会关闭Dropout层等模型随机行为
# 如果不考虑这些模型随机行为，两者作用类似
# 2、如果不考虑Dropout层等模型随机行为，把模型所有参数的requires_grad=False跟nn.Module.eval()是相似的

import torch
from torch import nn

from ShowDict import showDict


class NumGanModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义判别器模型
        self.DiscModel = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 200),
            nn.Sigmoid(),
            nn.Linear(200, 1),
            nn.Sigmoid(),
        )
        # 定义生成器模型
        self.GenModel = nn.Sequential(
            nn.Linear(1, 200),
            nn.Sigmoid(),
            nn.Linear(200, 28 * 28),
            nn.Sigmoid(),
        )
        # 将两个模型都移动到GPU执行
        self.DiscModel.to(torch.cuda.current_device())
        self.GenModel.to(torch.cuda.current_device())
        # 定义模型的Loss函数，由于GAN自始至终仅用到判别器的Loss函数，所以只需定义判别器的Loss，无需定义生成器的Loss
        self.DiscLoss = nn.MSELoss()
        # 定义两个模型的优化器
        self.DiscOptimiser = torch.optim.SGD(self.DiscModel.parameters(), lr=0.01)
        self.GenOptimiser = torch.optim.SGD(self.GenModel.parameters(), lr=0.01)
        # 模型训练记录
        self.record = {}

    def TrainModel(self, num_epochs, train_iter):
        for epoch_index in range(num_epochs):
            self.TrainEpoch(train_iter)
        showDict("NumGanModel", self.record, "barch times", ["loss_disc", "loss_gen"])

    def TrainEpoch(self, train_iter):
        self.DiscModel.train()
        self.GenModel.train()
        for imgs, tags in train_iter:
            tags = tags.to(device='cuda', dtype=torch.float)
            imgs = imgs.to(device='cuda', dtype=torch.float)
            # 掺杂生成器数据
            g_tag_seed = torch.rand(tags.size(0), 1).to(device='cuda', dtype=torch.float)
            g_imgs = self.GenModel(g_tag_seed).detach()
            g_imgs = g_imgs.view(g_imgs.size(0), 1, 28, 28)
            g_tags = torch.full((g_tag_seed.size(0),), 0, dtype=torch.float, device='cuda')
            tags_all = torch.cat((tags, g_tags), dim=0)
            imgs_all = torch.cat((imgs, g_imgs), dim=0)
            # 训练判别器
            tags_hat = self.DiscModel(imgs_all).squeeze()
            loss_disc = self.DiscLoss(tags_hat, tags_all)
            self.DiscOptimiser.zero_grad()
            loss_disc.backward()
            self.DiscOptimiser.step()
            # 训练生成器
            g_trian_tag_seed = torch.rand(tags_all.size(0), 1).to(device='cuda', dtype=torch.float)
            g_train_imgs = self.GenModel(g_trian_tag_seed)
            # 使用生成器生成的图片欺骗鉴别器
            # 如果g_train_tags全1说明欺骗成功（奖励）
            # 如果g_train_tags全0说明欺骗失败（惩罚）
            g_train_tags = self.DiscModel(g_train_imgs)
            g_train_tags_std = torch.full((g_trian_tag_seed.size(0), 1), 1, dtype=torch.float, device='cuda')
            # loss_gen越小，说明g_train_tags越接近全1，生成器越成功
            loss_gen = self.DiscLoss(g_train_tags, g_train_tags_std)
            self.GenOptimiser.zero_grad()
            loss_gen.backward()
            self.GenOptimiser.step()
            self.record[len(self.record)] = [loss_disc.item(), loss_gen.item()]
