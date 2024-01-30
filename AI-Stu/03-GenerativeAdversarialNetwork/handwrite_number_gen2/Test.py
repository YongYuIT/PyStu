import torch
from torch import nn


def test1():
    inputTensor = torch.randn(5, 1, 3, 3)
    print('inputTensor---------------------------------------------', inputTensor.shape)
    print(inputTensor)

    finputTensor = nn.Flatten().forward(inputTensor)
    print('finputTensor---------------------------------------------', finputTensor.shape)
    print(finputTensor)

    ufinputTensor = nn.Unflatten(1, (1, 3, 3)).forward(finputTensor)
    print('ufinputTensor---------------------------------------------', ufinputTensor.shape)
    print(ufinputTensor)

def test2():
    inputTensor = torch.randn(5, 5, 3, 3)
    print('inputTensor---------------------------------------------', inputTensor.shape)
    print(inputTensor)

    finputTensor = nn.Flatten().forward(inputTensor)
    print('finputTensor---------------------------------------------', finputTensor.shape)
    print(finputTensor)

    ufinputTensor = nn.Unflatten(1, (5, 3, 3)).forward(finputTensor)
    print('ufinputTensor---------------------------------------------', ufinputTensor.shape)
    print(ufinputTensor)

test2()