# 本示例着重体验torch.nn.Module中参数访问（w和b及其梯度）

import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
y = net(X)

# 1、通过state_dict访问模型参数
# 输出第3个块的参数
print("net index 2 state_dict type :", type(net[2].state_dict()))
print("net index 2 state_dict value :", net[2].state_dict())
print("-----------------------------------------------------------------")
print("net index 2 bias type :", type(net[2].bias))
print("net index 2 bias value :", net[2].bias)
print("net index 2 bias data type :", type(net[2].bias.data))
print("net index 2 bias data value :", net[2].bias.data)
print("-----------------------------------------------------------------")
# 2、通过grad查看模型反向传播进度
# 由于此时还没有backwards，所以grad没有
print(net[2].weight.grad == None)
y_l2 = torch.norm(y)
y_l2.backward()
print(net[2].weight.grad == None)
print("-----------------------------------------------------------------")

# 3、通过迭代器访问模型所有参数

# *[]在python里面是解包操作，例如：
array_test = [1, 2, 3]
print(*array_test)


def printNums(a, b, c):
    print("print num a-->", a)
    print("print num b-->", b)
    print("print num c-->", c)


printNums(*array_test)

# 可以通过迭代器访问net中所有模型参数
inter_all = net.named_parameters()
print("type of named_parameters ,", type(inter_all))
print("print all params in net... ...")
for name, params in inter_all:
    print("name of params in net:", name)
    print("params of params in net:", params)

print("print all params in net[0]... ...")
inter_all = net[0].named_parameters()
print("type of named_parameters ,", type(inter_all))
for name, params in inter_all:
    print("name of params in net:", name)
    print("params of params in net:", params)

print("-----------------------------------------------------------------")

# 4、通过下标直接访问模型参数

# 直接通过下标快捷访问模型参数
print("net 2.weight", net.state_dict()['2.weight'].data)
print("net 2.weight[0]", net.state_dict()['2.weight'].data[0])

print("-----------------------------------------------------------------")

# 5、复杂嵌套模型中访问模型参数

X = torch.rand(size=(2, 4))


def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block-{i}', block1())
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))

y = rgnet(X)

# 查看这个复杂嵌套的模型内部结构

print("rgnet struct :", rgnet)

# 通过下标快捷访问模型参数
print("0-1-0 :", rgnet[0][0][0].bias.data)
# 另一种等价的下标访问
print("0-1-0 :", rgnet.state_dict()['0.block-0.0.bias'].data)
