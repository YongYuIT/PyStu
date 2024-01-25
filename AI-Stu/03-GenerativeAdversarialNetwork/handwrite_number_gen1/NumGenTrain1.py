import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from NumDataSet import NumDataSet
from NumDisc import NumDisc
from NumGen import NumGen
import torchvision.transforms as transforms

# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]范围
])

# 训练轮次，数据集上全部训练一遍为一轮
train_times = 10
Gen = NumGen()
Disc = NumDisc()

for train_index in range(train_times):
    all_dataset = NumDataSet(root='./data', train=True, download=True,
                             transform=transform)
    train_loader = DataLoader(all_dataset, batch_size=10, shuffle=True)
    Disc.trainModel(1, train_loader, gen=Gen)
    print('train index-->', train_index)

Gen.netWork.eval()
with torch.no_grad():
    gen_size = 50
    gen_img = Gen(torch.rand(gen_size, 1))
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
