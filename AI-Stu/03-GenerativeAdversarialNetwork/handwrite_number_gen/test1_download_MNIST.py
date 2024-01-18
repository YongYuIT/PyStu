import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]范围
])

# 下载训练集
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 下载测试集
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

# 创建测试数据加载器
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
