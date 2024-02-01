import torch
from torch import nn

from ShowDict import showDict


class CNNGAN1(nn.Module):
    def __init__(self, GPU_ENABLE=False):
        super().__init__()
        self.GPU_ENABLE = GPU_ENABLE
        # 定义判别器模型
        self.DiscModel = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=3, padding=1),  # n*1*28*28 --> n*5*28*28
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(5),
            nn.AvgPool2d(kernel_size=2, stride=2),  # n*5*28*28 --> n*5*14*14
            nn.Conv2d(5, 10, kernel_size=5, padding=2),  # n*5*14*14 --> n*10*14*14
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(10),
            nn.AvgPool2d(kernel_size=2, stride=2),  # n*10*14*14 --> n*10*7*7
            nn.Flatten(start_dim=0),
            nn.Linear(490, 100),
            nn.LeakyReLU(0.02),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
        # 定义生成器模型
        self.GenModel = nn.Sequential(nn.Linear(100, 1), nn.Sigmoid())
        # 将两个模型都移动到GPU执行
        if self.GPU_ENABLE:
            self.DiscModel.to(torch.cuda.current_device())
            self.GenModel.to(torch.cuda.current_device())
        # 定义模型的Loss函数，由于GAN自始至终仅用到判别器的Loss函数，所以只需定义判别器的Loss，无需定义生成器的Loss
        # 采用二元交叉熵损失函数
        self.DiscLoss = nn.BCELoss()
        # 定义两个模型的优化器
        self.DiscOptimiser = torch.optim.Adam(self.DiscModel.parameters(), lr=0.0001)
        self.GenOptimiser = torch.optim.Adam(self.GenModel.parameters(), lr=0.0001)
        # 模型训练记录
        self.record = {}

    def DiscForward4Debug(self, X):
        print("start DiscForward4Debug-------------------------------------------------------")
        print("max-->", torch.max(X).item(), "||min-->", torch.min(X).item())
        y = None
        for index in range(len(self.DiscModel)):
            print("Module-->", self.DiscModel[index])
            for name, param in self.DiscModel[index].named_parameters():
                if param.requires_grad:
                    print('print params-->', name, param.data)
            if y is None:
                y = self.DiscModel[index](X)
            else:
                y = self.DiscModel[index](y)
        return y

    def TrainModel(self, num_epochs, train_iter):
        for epoch_index in range(num_epochs):
            self.TrainEpoch(train_iter)
            print("finish epoch index-->", epoch_index, "--of--", num_epochs)
        showDict("NumGanModel", self.record, "barch times", ["loss_disc", "loss_gen"])

    def TrainEpoch(self, train_iter):
        self.DiscModel.train()
        self.GenModel.train()
        for imgs, tags in train_iter:
            if self.GPU_ENABLE:
                tags = tags.to(device='cuda', dtype=torch.float)
                imgs = imgs.to(device='cuda', dtype=torch.float)
            # 掺杂生成器数据
            g_tag_seed = torch.randn(tags.size(0), 100).to(device='cuda', dtype=torch.float)
            g_imgs = self.GenModel(g_tag_seed).detach()
            g_imgs = g_imgs.view(g_imgs.size(0), 1, 28, 28)
            g_tags = torch.full((g_tag_seed.size(0),), 0, dtype=torch.float, device='cuda')
            tags_all = torch.cat((tags, g_tags), dim=0)
            imgs_all = torch.cat((imgs, g_imgs), dim=0)
            # 训练判别器
            loss_disc = self.TrainDisc(imgs_all)
            # 训练生成器
            g_trian_tag_seed = torch.randn(tags_all.size(0), 100).to(device='cuda', dtype=torch.float)
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

    def TrainDisc(self, img_tensor, label_tensor):
        # y_hat = self.DiscModel(img_tensor)
        y_hat = self.DiscForward4Debug(img_tensor)
        loss = self.DiscLoss(label_tensor, y_hat)
        self.DiscOptimiser.zero_grad()
        loss.backward()
        self.DiscOptimiser.step()
        return loss
