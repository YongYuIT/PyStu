import torch
from matplotlib import pyplot as plt
from torch import nn

from NumGanModel5 import NumGanModel5


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


def test3():
    Gan = NumGanModel5()
    Gan.GenModel.eval()
    with torch.no_grad():
        gen_size = 50
        g_seed = torch.randn(gen_size, 100).to(device='cuda', dtype=torch.float)
        gen_img = Gan.GenModel(g_seed)
        # 设置子图的行列数
        num_cols = 10  # 列
        num_rows = int(gen_size / num_cols)  # 行
        # 创建子图并显示图片
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), subplot_kw={'aspect': 'auto'})  # 调整图像大小
        for index in range(gen_size):
            col = index % num_cols
            row = int(index / num_cols)
            img = gen_img[index].view(1, 28, 28).to('cpu')
            npImage = img.permute(1, 2, 0).numpy()
            axes[row, col].imshow(npImage, interpolation='none', cmap='Blues')
        plt.tight_layout()  # 调整子图布局，防止重叠
        plt.show()

test3()