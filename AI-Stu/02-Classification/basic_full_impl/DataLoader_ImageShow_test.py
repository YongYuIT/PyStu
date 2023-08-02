import timeit

import torchvision
from torchvision import transforms
from torch.utils import data
import matplotlib.pyplot as plt
import DataLoader as thk_dataLoader
import ImageShow as thk_imageShow

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)

len(mnist_train), len(mnist_test)

print(mnist_train[0][0].shape)

X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
thk_imageShow.show_images(X.reshape(18, 28, 28), 2, 9, titles=thk_dataLoader.get_fashion_mnist_labels(y))
plt.show()

batch_size = 256


# 使用4个进程来读取数据
def get_dataloader_workers():
    return 4


train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())

# 确保子进程不会再次执行主模块的代码
if __name__ == '__main__':
    def testReadSpeed():
        for X, y in train_iter:
            continue


    execution_time = timeit.timeit(testReadSpeed, number=1)
    print("Execution time: ", execution_time)
