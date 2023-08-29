# 本示例着重体验torch.nn.Module中参数共享
import torch
from torch import nn

# 共享层
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])

print("-----------------------------------------------------------------")

# 下面看看共享层的梯度状况
X = torch.rand(2, 4)
y = net(X)
y_l2 = torch.norm(y)
y_l2.backward()
print("type of weight", type(net[2].weight))
# 梯度会加在一起
print(net[2].weight.grad)
print(net[4].weight.grad)
print("type of weight data", type(net[2].weight.data))
print(net[2].weight.data.grad)
print(net[4].weight.data.grad)
