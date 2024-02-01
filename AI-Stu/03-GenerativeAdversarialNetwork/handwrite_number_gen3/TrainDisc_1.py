import torch
from torchvision.datasets import MNIST

from CNN_GAN_1 import CNNGAN1
import torchvision.transforms as transforms

from ShowDict import showDict

gan = CNNGAN1(False)

# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor()  # 将图像转换为Tensor
])

# 加载真实数据集
true_dataset = MNIST(root='./data', train=True, download=True, transform=transform)

trainRecord = {}
gan.DiscModel.train()
trainIndex = 0
for img_tensor, label in true_dataset:
    # 真数据训练一把
    lossTrue = gan.TrainDisc(img_tensor.view(1, 1, 28, 28), torch.Tensor([1]))
    # 假数据训练一把
    lossFake = gan.TrainDisc((torch.rand((1, 1, 28, 28))), torch.Tensor([0]))
    trainRecord[len(trainRecord)] = [lossTrue.item(), lossFake.item()]
    trainIndex += 1
    if trainIndex % 1000 == 0:
        print("finish:", trainIndex)
    if trainIndex >= 5:
        break
showDict("Disc Train Loss", trainRecord, "trainIndex", ["lossTrue", "lossFake"])
