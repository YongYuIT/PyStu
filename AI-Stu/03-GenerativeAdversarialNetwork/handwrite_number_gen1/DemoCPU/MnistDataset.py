import numpy as np
from torchvision.datasets import MNIST
import torch
import matplotlib.pyplot as plt


class MnistDataset(MNIST):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        target = torch.zeros((10))
        target[label] = 1.0
        # image data, normalised from 0-255 to 0-1
        img_array = np.array(img).reshape(-1)
        image_values = torch.FloatTensor(img_array) / 255.0
        return label, image_values, target

    def plot_image(self, index):
        img, label = super().__getitem__(index)
        plt.title("label = " + str(label))
        plt.imshow(img, interpolation='none', cmap='Blues')
        pass

    pass


def test():
    mnist_dataset = MnistDataset(root='./data', train=True, download=True)
    mnist_dataset.plot_image(17)
    plt.show()
