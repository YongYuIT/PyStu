import sys

sys.path.append('../../Tasks/Task1')
from Tools import ShowDict as SD

from ModelDesign import LeNetGPUModelDef as MD
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]范围
])

batchSize = 6000

# 下载训练集
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)

# 下载测试集
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

# 创建测试数据加载器
test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)

learningRate = 0.3
numEpochs = 200
model = MD.LeNetGPUModelDef(learningRate, numEpochs)
model.initModel()
dictTrainRecords = model.train_model(train_loader, test_loader)
SD.showDict("LeNetGPUModelDef", dictTrainRecords, "epoch", ["avgLoss", "correct"])

# 保存模型
model.saveModel("LeNetGPUModelDef")

# 加载模型，验证模型
modelLoad = MD.LeNetGPUModelDef()
modelLoad.loadModel("LeNetGPUModelDef")
avgLoss, correct = modelLoad.evaluate_model(test_loader)
print("model check avgLoss-->", avgLoss, "||correct-->", correct)
