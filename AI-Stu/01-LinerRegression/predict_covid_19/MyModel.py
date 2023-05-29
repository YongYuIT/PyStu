# Pytorch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, input_dim):
        super(MyModel, self).__init__()
        # TODO: 修改模型结构，注意维度
        self.layers = nn.Sequential(
            # r=nn.Linear(x,w,bias=True)  <=>  r=wx+b
            # a=sigmoid(r)  =>  a是神经元，包括五部分：输入x，权重w，偏置b，加权和r，激活函数sigmoid
            # y=b+c^T*a
            nn.Linear(input_dim, 16),  # 神经网络第一层：input_dim个features，16个神经元，使用ReLU作为激活函数；也被称为“输入层”
            nn.ReLU(),
            nn.Linear(16, 8),  # 神经网络第二层：16个features，8个神经元，使用ReLU作为激活函数；也被称为“隐藏层”，隐藏层是输入层和输出层之间的层级，可以是多层
            nn.ReLU(),
            # 在原始代码基础上增加一层
            nn.Linear(8, 4),  # 神经网络第三层：8个features，4个神经元，使用ReLU作为激活函数；隐藏层
            nn.ReLU(),
            nn.Linear(4, 1)  # 第四层输出（没有激活函数就表示不是神经网络中的层）；也被称为“输出层”
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)  # (B, 1) -> (B)
        return x
