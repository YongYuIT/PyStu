from torch import nn
import torch


class FCGenModelDef(nn.Module):

    # 模型初始化
    @staticmethod
    def init_W(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    def __init__(self, justifyModel, stop_loss, max_epoch):
        super().__init__()
        self.stop_loss = stop_loss  # 定义一个学习终点
        self.max_epoch = max_epoch
        # 保存超参
        self.justifyModel = justifyModel
        # 判别器模型参数不参与update
        if justifyModel != None:
            for justParams in self.justifyModel.parameters():
                justParams.requires_grad = False
        # 模型各层定义，_modules是一个字典对象
        self._modules['first_full_conn'] = nn.Linear(100, 200)
        self._modules['first_fc_active'] = nn.LeakyReLU(0.02)
        self._modules['normal'] = nn.LayerNorm(200)
        self._modules['second_full_conn'] = nn.Linear(200, 28 * 28)
        self._modules['second_fc_active'] = nn.Sigmoid()
        # 放到GPU运算
        self.to(torch.cuda.current_device())

    # y_hat: n * 1 * 28 * 28
    # 返回n张生成图片的平均相似度的负数即：平均不相似度
    def loss(self, y_hat):
        y_class_max = self.justifyModel.lossForJustify(y_hat)
        # 不相似度均值作为模型损失，不相似度越低越好
        y_class_avg = y_class_max.sum() / y_hat.size(0)
        return y_class_avg

    # X: n * 100
    def forward(self, X):
        X_1 = self._modules['first_full_conn'](X)
        X_1_V = self._modules['first_fc_active'](X_1)
        X_N = self._modules['normal'](X_1_V)
        X_2 = self._modules['second_full_conn'](X_N)
        y = self._modules['second_fc_active'](X_2)
        # 重建图片结构
        y_img = None
        if y.dim() < 2:
            y_img = y.view(1, 28, 28)
        else:
            y_img = y.view(y.size(0), 1, 28, 28)
        ########################测试，需删除
        # print("----X (noise ) max-->", torch.max(X).item(), "||min-->", torch.min(X).item(),
        #       "----y_hat (image ) max-->", torch.max(y_img).item(), "||min-->", torch.min(y_img).item())
        ########################
        return y_img

    # 由于这些参数跟具体运算设备相关，所以需要在运算设备确定之后才能初始化
    def initModel(self):
        # 定义模型初始化方法（即规定学习起点）
        self.apply(FCGenModelDef.init_W)
        # 定义优化器
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.8, momentum=0.9)

    # inputX：1000 * 1000 * 100
    # inputTextX：500 * 1000 * 100
    def train_model(self, inputX, inputTestX):
        dictTrainRecords = {}
        avgLoss = 0
        epoch_index = 0
        while avgLoss < self.stop_loss or epoch_index < self.max_epoch:
            self.train_epoch(inputX)
            avgLoss = self.evaluate_model(inputTestX)
            epoch_index += 1
            print("epoch index-->", epoch_index, "||avgLoss-->", avgLoss)
            dictTrainRecords[epoch_index] = [avgLoss]
        return dictTrainRecords

    # 单独的一轮epoch，inputX：1000*1000*100
    def train_epoch(self, inputX):
        current_lr = self.optimizer.param_groups[0]['lr']
        print("start epoch current learning rate:", current_lr)
        # 将模型设置为训练模式
        self.train()
        for index in range(inputX.size(0)):
            X = inputX[index].to('cuda')
            self.train_batch(X)

    # 单独的一次随机小批量，X：1000*100
    def train_batch(self, X):
        # 前向传播，计算y的预测值，y_hat是生成的图片
        y_hat = self(X)  # 等价于 y_hat = self.forward(X)
        # 这里是平均损失
        avgLoss = self.loss(y_hat)
        # 将梯度矩阵归零，避免梯度积累
        self.optimizer.zero_grad()
        # 后向传播，计算本次随机小批量的梯度
        avgLoss.backward()
        # 根据计算的梯度矩阵，更新模型参数
        self.optimizer.step()

    # inputTextX：500 * 1000 * 100
    def evaluate_model(self, inputTextX):
        # 将模型设置为评估模式
        print("start evaluate_model")
        self.eval()
        with torch.no_grad():
            # 初始化统计参数
            totalLoss = 0
            for index in range(inputTextX.size(0)):
                X = inputTextX[index].to('cuda')
                avgLoss = self.evaluate_batch(X)
                # print("avgLoss-->", avgLoss, '||index-->', index)
                totalLoss += (avgLoss * X.size(0))
        return totalLoss / (inputTextX.size(0) * inputTextX.size(1))

    # 小批量评估
    def evaluate_batch(self, X):
        # 等价于 y_hat = self.forward(X)
        y_hat = self(X)
        # 计算测准样本数
        loss = self.loss(y_hat)
        # print("--loss-->", loss)
        lossValue = loss.item()
        return lossValue

    def saveModel(self, saveName):
        torch.save(self.state_dict(), saveName)

    def loadModel(self, saveName):
        self.load_state_dict(torch.load(saveName))


def test():
    y_class = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    print('y_class size-->', y_class.size(0))
    y_class_max_index = y_class.argmax(axis=1)
    print('y_class_max_index-->', y_class_max_index)
    y_class_max = y_class[torch.arange(y_class.size(0)), y_class_max_index]
    print('y_class_max-->', y_class_max)
    y_class_avg = y_class_max.sum() / 2
    print('y_class_avg-->', y_class_avg)
