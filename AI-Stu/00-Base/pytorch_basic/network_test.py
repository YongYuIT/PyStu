import torch
from torch import nn

# 1、借助Sequential和内置块，快速构建深度网络
# 这是一个由三个块顺序连接形成的深度网络
# input(20) --> 线性变换(20->256) --> ReLU激活(256) --> 线性变换(256->10) --> 输出(10)
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
# 随机生成两笔资料，每笔资料20个features
X = torch.rand(2, 20)
print("X org value: ", X)
print("X after net: ", net(X))
print("-----------------------------------------------------------------------------")

# 2、同样是上面的例子，看一下快速微分会有什么表现
X1 = X.clone().detach()
X1.requires_grad_(True)
print("X1 org value: ", X1)
Y = net(X1)
# Y.backward() 张量不允许backward
Y_L2 = torch.norm(Y)
Y_L2.backward()
# 一般而言，求输出（经过某种标量化处理，例如Loss函数）对输入X1的梯度毫无意义，因为在深度模型SGD时候，自变量是w和b，而不是输入
# 当然，输出也可以看作对输入的一顿操作变换，这样在数学上求输出对输入的梯度是可以的，虽然没有啥实际意义
print("X1 grad value: ", X1.grad)
print("params grad value: ", net[0].weight.grad)
print("X1 after net (is Y): ", Y)
print("Y L2: ", Y_L2)
print("-----------------------------------------------------------------------------")
