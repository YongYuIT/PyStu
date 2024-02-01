import torch
from torch import nn


def FFunc(X):

    Y1 = nn.Conv2d(1, 5, kernel_size=3, padding=1).forward(X)  # n*1*28*28 --> n*5*28*28
    Y2 = nn.LeakyReLU(0.02).forward(Y1)
    Y3 = nn.AvgPool2d(kernel_size=2, stride=2).forward(Y2)  # n*5*28*28 --> n*5*14*14
    Y4 = nn.Conv2d(5, 10, kernel_size=5, padding=2).forward(Y3)  # n*5*14*14 --> n*10*14*14
    Y5 = nn.LeakyReLU(0.02).forward(Y4)
    Y6 = nn.AvgPool2d(kernel_size=2, stride=2).forward(Y5)  # n*10*14*14 --> n*10*7*7
    Y7 = nn.Flatten(start_dim=0).forward(Y6)
    Y8 = nn.Linear(490, 100).forward(Y7)
    Y9 = nn.LeakyReLU(0.02).forward(Y8)
    Y10 = nn.Linear(100, 1).forward(Y9)
    Y11 = nn.Sigmoid().forward(Y10)
    print(Y11)


for index in range(100):
    FFunc(torch.rand(1, 28, 28))
