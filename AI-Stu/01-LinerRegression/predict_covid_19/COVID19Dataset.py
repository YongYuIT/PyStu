# Pytorch
import torch
from torch.utils.data import Dataset


# x：特征
# y: 结果，样本（每行数据）中的病例数
class COVID19Dataset(Dataset):
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
