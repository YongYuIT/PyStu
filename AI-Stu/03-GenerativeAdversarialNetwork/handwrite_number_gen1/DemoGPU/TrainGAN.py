import torch
from matplotlib import pyplot as plt

from DemoCPU.MnistDataset import MnistDataset
from DemoCPU.RandTools import generate_random_seed
from DemoGPU.Discriminator import Discriminator
from DemoGPU.Generator import Generator

D = Discriminator()
G = Generator()

epochs = 4

mnist_dataset = MnistDataset(root='./data', train=True, download=True)

for epoch in range(epochs):
    print("epoch = ", epoch + 1)

    # train Discriminator and Generator

    for label, image_data_tensor, target_tensor in mnist_dataset:
        # train discriminator on true
        D.train(image_data_tensor, torch.FloatTensor([1.0]))

        # train discriminator on false
        # use detach() so gradients in G are not calculated
        D.train(G.forward(generate_random_seed(100)).detach(), torch.FloatTensor([0.0]))

        # train generator
        G.train(D, generate_random_seed(100), torch.FloatTensor([1.0]))

        pass

    pass

D.plot_progress()
G.plot_progress()

# ------------------------------------------------------------------------------------

f, axarr = plt.subplots(2, 3, figsize=(16, 8))
for i in range(2):
    for j in range(3):
        output = G.forward(generate_random_seed(100)).cpu()
        img = output.detach().numpy().reshape(28, 28)
        axarr[i, j].imshow(img, interpolation='none', cmap='Blues')
        pass
    pass
plt.title("gen test")
plt.show()

# ------------------------------------------------------------------------------------


seed1 = generate_random_seed(100)
out1 = G.forward(seed1).cpu()
img1 = out1.detach().numpy().reshape(28, 28)
plt.imshow(img1, interpolation='none', cmap='Blues')
plt.title("gen seed1")
plt.show()
# ------------------------------------------------------------------------------------
seed2 = generate_random_seed(100)
out2 = G.forward(seed2).cpu()
img2 = out2.detach().numpy().reshape(28, 28)
plt.imshow(img2, interpolation='none', cmap='Blues')
plt.title("gen seed2")
plt.show()

# ------------------------------------------------------------------------------------

count = 0

# plot a 3 column, 2 row array of generated images
f, axarr = plt.subplots(3, 4, figsize=(16, 8))
for i in range(3):
    for j in range(4):
        seed = seed1 + (seed2 - seed1) / 11 * count
        output = G.forward(seed).cpu()
        img = output.detach().numpy().reshape(28, 28)
        axarr[i, j].imshow(img, interpolation='none', cmap='Blues')
        count = count + 1
        pass
    pass
plt.title("gen seed1 to seed2")
plt.show()

# ------------------------------------------------------------------------------------

seed3 = seed1 + seed2
out3 = G.forward(seed3).cpu()
img3 = out3.detach().numpy().reshape(28, 28)
plt.imshow(img3, interpolation='none', cmap='Blues')
plt.title("gen seed1 + seed2")
plt.show()

# ------------------------------------------------------------------------------------
seed4 = seed1 - seed2
out4 = G.forward(seed4).cpu()
img4 = out4.detach().numpy().reshape(28, 28)
plt.imshow(img4, interpolation='none', cmap='Blues')
plt.title("gen seed1 - seed2")
plt.show()
# ------------------------------------------------------------------------------------
seed4 = seed1 * seed2
out4 = G.forward(seed4).cpu()
img4 = out4.detach().numpy().reshape(28, 28)
plt.imshow(img4, interpolation='none', cmap='Blues')
plt.title("gen seed1 * seed2")
plt.show()
