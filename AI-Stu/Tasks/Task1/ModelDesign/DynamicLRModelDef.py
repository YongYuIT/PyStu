from ModelDesign import ModelDef as MD
import torch.optim.lr_scheduler as lr_scheduler


class DynamicLRModelDef(MD.ModelDef):
    def __init__(self, batch_size=0, learning_rate=0, num_epochs=0):
        super().__init__(batch_size, learning_rate, num_epochs)
        # 每5个epoch学习率乘以0.5
        self.learning_rate_scheduler = lr_scheduler.StepLR(self.updater, step_size=5, gamma=0.5)

    # 单独的一轮epoch
    def train_epoch(self, train_iter):
        current_lr = self.updater.param_groups[0]['lr']
        print("current learning rate:", current_lr)
        super().train_epoch(train_iter)
        self.learning_rate_scheduler.step()
