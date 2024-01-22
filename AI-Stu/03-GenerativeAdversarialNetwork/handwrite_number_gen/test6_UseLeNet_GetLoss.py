# 实际使用判别器，看看在真实数据上Loss会是多少，为判别器训练提供终止依据

import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from ModelDesign import FCGenModelDef as FCG_MD
import torch
import matplotlib.pyplot as plt
from ModelDesign import LeNetGPUModelDef as LN_MD

# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]范围
])

# 下载测试集
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

# 创建测试数据加载器
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

view_batch_X = torch.Tensor()
view_batch_Y = torch.Tensor()
batch_index = 0

for batch_X, batch_Y in test_loader:
    print('batch_index --> ', batch_index)
    print('batch_X size --> ', batch_X.size())
    print('batch_Y size --> ', batch_Y.size())
    batch_index += 1
    print('-------------------------------------')
    # 取其中第11个batch中的样本数据看看
    if batch_index == 10:
        view_batch_X = batch_X
        view_batch_Y = batch_Y

# 加载判别器模型
modelLoadLN = LN_MD.LeNetGPUModelDef()
modelLoadLN.loadModel("LeNetGPUModelDef")

# 加载生成器模型
modelLoad = FCG_MD.FCGenModelDef(modelLoadLN)
modelLoad.loadModel("FCGenModelDef")

# 设置子图的行列数
simple_size = view_batch_X.size(0)
num_cols = 10  # 列
num_rows = int(simple_size / num_cols) + 1  # 行
# 创建子图并显示图片
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), subplot_kw={'aspect': 'auto'})  # 调整图像大小

modelLoad.eval()
with torch.no_grad():
    loss = modelLoad.lossTensor(view_batch_X.to('cuda'))
    loss = loss.to('cpu')
    for index in range(view_batch_X.size(0)):
        col = index % num_cols
        row = int(index / num_cols)
        image = view_batch_X[index]
        npImage = image.permute(1, 2, 0).numpy()
        axes[row, col].imshow(npImage, cmap='gray')
        axes[row, col].set_title(round(loss[index].item(), 4))
plt.tight_layout()  # 调整子图布局，防止重叠
plt.show()
