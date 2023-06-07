# Numerical Operations
import math
# Reading/Writing Data
import os

# Pytorch
import torch
import torch.nn as nn
# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter
# For Progress Bar
from tqdm import tqdm


def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(reduction='mean')  # 定义你的损失函数，不要修改它

    # 定义优化算法
    # TODO: 查看 https://pytorch.org/docs/stable/optim.html 以获得更多可用算法
    # TODO: L2 正则化（优化器（权重衰减...）或自行实现）
    # torch.optim.SGD有三个入参：
    #   1. 模型参数（也就是θ向量，也就是所有的模型中所有的w和b）
    #   2. 学习速率
    #   3. 动量，一种动态调整学习速率的方法
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)

    writer = SummaryWriter()  # Writer of tensorboard.

    if not os.path.isdir('./models'):
        os.mkdir('./models')  # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train()  # 将模型设置为训练模式
        loss_record = []

        # tqdm 是一个可视化训练进度的软件包
        # 通过tqdm将train_loader分割成每块batch_size大小的batch
        train_pbar = tqdm(train_loader, position=0, leave=True)
        print("batche遍历 start ...... train_pbar size(batche个数):", len(train_pbar), ",type:", type(train_pbar))
        batcheIndex = 0
        for x, y in train_pbar:
            print("batcheIndex:", batcheIndex, "-->x-shape:", x.shape, "-->y-shape:", y.shape)
            optimizer.zero_grad()  # 将梯度初始化为0
            x, y = x.to(device), y.to(device)  # Move your data to device.
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()  # 计算梯度-反向传播
            optimizer.step()  # 参数更新.
            step += 1
            loss_record.append(loss.detach().item())

            # 在 tqdm 进度条上显示当前epoch数和loss（日志前面的红色部分）
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})
            batcheIndex += 1
        print("batche遍历 end ......")
        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()  # 将模型设置为评估模式
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])  # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return
