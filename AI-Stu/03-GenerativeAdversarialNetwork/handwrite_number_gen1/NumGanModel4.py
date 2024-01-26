import torch
from NumGanModel3 import NumGanModel3


class NumGanModel4(NumGanModel3):

    def TrainEpoch(self, train_iter):
        self.DiscModel.train()
        self.GenModel.train()
        for imgs, tags in train_iter:
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
            tags_hat = self.DiscModel(imgs_all).squeeze()
            loss_disc = self.DiscLoss(tags_hat, tags_all)
            self.DiscOptimiser.zero_grad()
            loss_disc.backward()
            self.DiscOptimiser.step()
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
