import torch
from torch.utils.data import DataLoader

from NumDisc import NumDisc
from NumGen import NumGen
from NumMixDataSet import NumMixDataSet
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]范围
])

# 训练轮次，数据集上全部训练一遍为一轮
train_times = 200
Gen = NumGen()
Disc = NumDisc()
for train_index in range(train_times):
    # 准备生成器伪造的数据
    g_data_seed = torch.rand(30000, 1)
    g_data_set = Gen(g_data_seed).detach()
    g_data_set = g_data_set.view(g_data_set.size(0), 1, 28, 28)
    # 掺杂在数据集中去（参入一半生成数据）
    all_dataset = NumMixDataSet(noiseDataSet=g_data_set, root='./data', train=True, download=True,
                                transform=transform)
    train_loader = DataLoader(all_dataset, batch_size=5000, shuffle=True)
    # 使用混合数据训练鉴别器
    Disc.trainModel(1, train_loader)
    # 训练生成器
    g_data_seed = torch.rand(60000, 1)
    Gen.trainModel(Disc, g_data_seed, torch.full((g_data_seed.size(0), 1), 1, dtype=torch.float))
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
        axes[row, col].imshow(npImage, cmap='gray')
    plt.tight_layout()  # 调整子图布局，防止重叠
    plt.show()

# 训练失败，失败原因：每一轮60000个样本使得鉴别器进步太快，鉴别器进化完成后再更新生成器，生成器完全跟不上节奏
# 改进：在每次batch时候，使用改进的生成器，并用改进后的生成器掺杂噪音