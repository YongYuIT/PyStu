# 将一定比例的生成数据和真实数据混合的数据集
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from NumDataSet import NumDataSet
from NumGen import NumGen
from typing import Callable, Optional


class NumMixDataSet(NumDataSet):
    def __init__(self, noiseDataSet, root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False, ):
        super().__init__(root, train, transform, target_transform, download)
        self.noiseDataSet = noiseDataSet

    def __getitem__(self, index):
        if index < self.noiseDataSet.size(0):
            return self.noiseDataSet[index], 0
        else:
            return super().__getitem__(index)


def testView():
    # 定义数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为Tensor
        transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]范围
    ])
    # 准备生成器伪造的数据
    Gen = NumGen()
    g_data_seed = torch.rand(10000, 1)
    g_data_set = Gen(g_data_seed).detach()
    g_data_set = g_data_set.view(g_data_set.size(0), 1, 28, 28)
    g_data_set = g_data_set.to('cpu')
    # 下载训练集，如果已存在直接加载
    train_dataset = NumMixDataSet(noiseDataSet=g_data_set, root='./data', train=True, download=True,
                                  transform=transform)
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    batch_index = 0
    for batch_X, batch_Y in train_loader:
        print('batch_index --> ', batch_index)
        print('batch_X size --> ', batch_X.size())
        print('batch_Y size --> ', batch_Y.size())
        batch_index += 1
        print('-------------------------------------')
        # 取其中第11个batch中的样本数据看看
        if batch_index == 10:
            view_batch_X = batch_X
            view_batch_Y = batch_Y
            break
    # 设置子图的行列数
    simple_size = 30
    num_cols = 10  # 列
    num_rows = int(simple_size / num_cols)  # 行
    # 创建子图并显示图片
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), subplot_kw={'aspect': 'auto'})  # 调整图像大小
    for index in range(simple_size):
        col = index % num_cols
        row = int(index / num_cols)
        tensor_view = view_batch_X[index]
        print("max-->", torch.max(tensor_view).item())
        print("min-->", torch.min(tensor_view).item())
        npImage = tensor_view.permute(1, 2, 0).numpy()
        axes[row, col].imshow(npImage, cmap='gray')
        axes[row, col].set_title(view_batch_Y[index].item())
    plt.tight_layout()  # 调整子图布局，防止重叠
    plt.show()


testView()
