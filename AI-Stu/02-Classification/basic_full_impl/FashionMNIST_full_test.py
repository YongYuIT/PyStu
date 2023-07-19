import torchvision
from torchvision import transforms
from torch.utils import data


# 使用4个进程来读取数据
def get_dataloader_workers():
    return 4

# 下载Fashion-MNIST数据集，然后将其加载到内存中
def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


train_iter, test_iter = load_data_fashion_mnist(32, resize=64)

# 确保子进程不会再次执行主模块的代码
if __name__ == '__main__':
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break
