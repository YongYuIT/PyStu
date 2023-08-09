import torch
import ChatShow
import Relu as thk_relu
from matplotlib import pyplot as plt

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
ChatShow.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
plt.show()

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = thk_relu.relu(x)
ChatShow.plot(x.detach(), y.detach(), 'x', 'self relu(x)', figsize=(5, 2.5))
plt.show()

