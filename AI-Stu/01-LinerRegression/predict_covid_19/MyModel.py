# Pytorch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, input_dim):
        super(MyModel, self).__init__()
        # TODO: 修改模型结构，注意维度
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)  # (B, 1) -> (B)
        return x
