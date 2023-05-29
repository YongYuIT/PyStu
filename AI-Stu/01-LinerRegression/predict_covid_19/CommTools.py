# Numerical Operations
import numpy as np
# Pytorch
import torch
from torch.utils.data import random_split
# For Progress Bar
from tqdm import tqdm


# 修复随机数生成器种子以实现可重复性
def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 将提供的训练数据拆分为训练集和验证集
def train_valid_split(data_set, valid_ratio, seed):
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size],
                                        generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


def predict(test_loader, model, device):
    model.eval()  # 将模型设置为评估模式
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds


# 选择有用的特征进行回归
def select_feat(train_data, valid_data, test_data, select_all=True):
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [0, 1, 2, 3, 4]  # TODO: 选择合适的特征列

    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid
