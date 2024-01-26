import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from NumDataSet import NumDataSet
import torchvision.transforms as transforms

from NumGanModel4 import NumGanModel4

# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]范围
])

# 加载真实数据集
true_dataset = NumDataSet(root='./data', train=True, download=True, transform=transform)
true_dataloader = DataLoader(true_dataset, batch_size=1, shuffle=True)

# 训练轮次，数据集上全部训练一遍为一轮
Gan = NumGanModel4()
train_times = 2
Gan.TrainModel(train_times, true_dataloader)

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
