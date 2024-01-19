from ModelDesign.LeNetModelDef import LeNetModelDef as CPUModel
import torch


class LeNetGPUModelDef(CPUModel):
    def __init__(self, learning_rate=0., num_epochs=0):
        super().__init__(learning_rate, num_epochs)
        self.to(torch.cuda.current_device())

    # 单独的一轮epoch
    def train_epoch(self, train_iter):
        current_lr = self.optimizer.param_groups[0]['lr']
        print("start epoch current learning rate:", current_lr)
        # 将模型设置为训练模式
        self.train()
        for X, y in train_iter:
            X = X.to('cuda')
            y = y.to('cuda')
            self.train_batch(X, y)
        # 更新学习速率
        self.learning_rate_scheduler.step()

    # 模型评估，返回正确识别的百分比
    def evaluate_model(self, test_iter):
        # 将模型设置为评估模式
        self.eval()
        with torch.no_grad():
            # 初始化统计参数
            totalSamples = 0
            equalSamples = 0
            totalLoss = 0
            for X, y in test_iter:
                X = X.to('cuda')
                y = y.to('cuda')
                equal_num, loss = self.evaluate_batch(X, y)
                totalSamples += y.size(0)
                totalLoss += loss
                equalSamples += equal_num
        return totalLoss / totalSamples, equalSamples / totalSamples
