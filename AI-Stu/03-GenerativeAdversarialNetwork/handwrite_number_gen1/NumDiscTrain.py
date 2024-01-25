import torch

from NumDisc import NumDisc
from NumGen import NumGen
from NumMixDataSet import NumMixDataSet
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from ShowDict import showDict

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
# 掺杂在数据集中去
all_dataset = NumMixDataSet(noiseDataSet=g_data_set, root='./data', train=True, download=True,
                            transform=transform)

# 定义训练集和测试集的比例
train_ratio = 0.7  # 假设70%用于训练集
# 根据比例随机划分数据集
train_size = int(train_ratio * len(all_dataset))
test_size = len(all_dataset) - train_size
print("train_size-->", train_size, "||test_size-->", test_size)
train_dataset, test_dataset = random_split(all_dataset, [train_size, test_size])
# 使用 DataLoader 加载训练集和测试集
train_loader = DataLoader(train_dataset, batch_size=5000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=3000, shuffle=False)

disc = NumDisc()
record = disc.trainModel(100, train_loader, test_loader)
showDict("NumDisc", record, 'epochTimes', ['avgLoss'])
