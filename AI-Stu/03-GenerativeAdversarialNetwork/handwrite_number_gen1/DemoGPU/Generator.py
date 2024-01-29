import pandas
import torch
from matplotlib import pyplot as plt
from torch import nn

from DemoCPU.RandTools import generate_random_seed


class Generator(nn.Module):

    def __init__(self):
        # initialise parent pytorch class
        super().__init__()

        # define neural network layers
        self.model = nn.Sequential(
            nn.Linear(100, 200),
            nn.LeakyReLU(0.02),

            nn.LayerNorm(200),

            nn.Linear(200, 784),
            nn.Sigmoid()
        )

        # 转移到GPU
        self.model.to(torch.cuda.current_device())

        # create optimiser, simple stochastic gradient descent
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

        # counter and accumulator for progress
        self.counter = 0
        self.progress = []

        pass

    def forward(self, inputs):
        inputs = inputs.to(device='cuda')
        # simply run model
        return self.model(inputs)

    def train(self, D, inputs, targets):
        inputs = inputs.to(device='cuda')
        targets = targets.to(device='cuda')

        # calculate the output of the network
        g_output = self.forward(inputs)

        # pass onto Discriminator
        d_output = D.forward(g_output)

        # calculate error
        loss = D.loss_function(d_output, targets)

        # increase counter and accumulate error every 10
        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass

        # zero gradients, perform a backward pass, update weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))
        plt.title("Generator loss")
        plt.show()
        pass

    pass


def test():
    G = Generator()
    output = G.forward(generate_random_seed(100))
    img = output.detach().numpy().reshape(28, 28)
    plt.imshow(img, interpolation='none', cmap='Blues')
    plt.show()
