import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]范围
])

# 下载训练集，如果已存在直接加载
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

batch_index = 0
total_sample_size = 0

view_batch_X = torch.Tensor()
view_batch_Y = torch.Tensor()

for batch_X, batch_Y in train_loader:
    print('batch_index --> ', batch_index)
    print('batch_X size --> ', batch_X.size())
    print('batch_Y size --> ', batch_Y.size())
    batch_index += 1
    total_sample_size += batch_X.size(0)
    print('-------------------------------------')
    # 取其中第11个batch中的样本数据看看
    if batch_index == 10:
        view_batch_X = batch_X
        view_batch_Y = batch_Y
print('total_sample_size --> ', total_sample_size)

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
