import random

import pandas
import torch
from matplotlib import pyplot as plt
from torch import nn

from DemoCPU.MnistDataset import MnistDataset
from DemoCPU.RandTools import generate_random_image


class Discriminator(nn.Module):

    def __init__(self):
        # initialise parent pytorch class
        super().__init__()

        # define neural network layers
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(0.02),

            nn.LayerNorm(200),

            nn.Linear(200, 1),
            nn.Sigmoid()
        )

        # create loss function
        self.loss_function = nn.BCELoss()

        # create optimiser, simple stochastic gradient descent
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

        # counter and accumulator for progress
        self.counter = 0
        self.progress = []

        pass

    def forward(self, inputs):
        # simply run model
        return self.model(inputs)

    def train(self, inputs, targets):
        # calculate the output of the network
        outputs = self.forward(inputs)

        # calculate loss
        loss = self.loss_function(outputs, targets)

        # increase counter and accumulate error every 10
        self.counter += 1;
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        if (self.counter % 10000 == 0):
            print("counter = ", self.counter)
            pass

        # zero gradients, perform a backward pass, update weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))
        plt.show()
        pass

    pass


def test():
    mnist_dataset = MnistDataset(root='./data', train=True, download=True)
    D = Discriminator()
    for label, image_data_tensor, target_tensor in mnist_dataset:
        # real data
        D.train(image_data_tensor, torch.FloatTensor([1.0]))
        # fake data
        D.train(generate_random_image(784), torch.FloatTensor([0.0]))
        pass
    D.plot_progress()
    for i in range(4):
        image_data_tensor = mnist_dataset[random.randint(0, 60000)][1]
        print(D.forward(image_data_tensor).item())
        pass

    for i in range(4):
        print(D.forward(generate_random_image(784)).item())
        pass


