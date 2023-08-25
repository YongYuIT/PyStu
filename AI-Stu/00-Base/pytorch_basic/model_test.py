import torch
from torch import nn
from torch.nn import functional as F

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


# 3、自定义块
# 在pytorch中，nn.Sequential、nn.Linear、nn.ReLU都继承自nn.Module
# nn.Module是一切块的基类，所以自定义块也需要继承nn.Module
# 自定义块时需要注意，任何自定义块都需要做到一下5点：
# a. 将输入数据作为其前向传播函数的参数
# b. 通过前向传播函数来生成输出。请注意，输出的形状可能与输入的形状不同。例如，我们示例1模型中的第一个全连接的层接收一个20维的输入，但是返回一个维度为256的输出
# c. 计算其输出关于输入的梯度，可通过其反向传播函数进行访问。通常这是自动发生的
# d. 存储和访问前向传播计算所需的参数
# e. 根据需要初始化模型参数

class MyModule(nn.Module):
    def __init__(self):
        # 调用父类Module的构造函数来执行必要的初始化
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params
        super().__init__()
        # 隐藏层
        self.myHidden = nn.Linear(20, 256)
        # 激活层
        self.myActive = nn.ReLU()
        # 输出层
        self.myOut = nn.Linear(256, 10)

        # 定义模型的前向传播，即如何根据输入X返回所需的模型输出

    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义
        # 实际上，MyModule相当于一个 Linear(20, 256) --> ReLU --> Linear(256, 10) 的线性组合
        return self.myOut(self.myActive(self.myHidden(X)))


X3 = X.clone().detach()
net1 = MyModule()
print("X3 org value: ", X3)
# 可以看到net1虽然与net网络架构是一样的，但是由于参数初始化不同（存在随机性），所以对相同的输入还是有着不同的输出
print("X3 after net: ", net1(X3))
# 打印net和net1第一个Linear的权重就知道了
print("net first Linear wight: ", net[0].weight.data[0])
print("net1 first Linear wight: ", net1.myHidden.weight.data[0])


# 下面对net和net1的使用相同的参数初始化，得到的输出就相同了
# 定义参数初始化方法，将线性模型Linear的权重全部设置为1，偏置全部设置为0
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


# 指定net和net1的初始化方法
print("change by init_constant...")
net.apply(init_constant)
net1.apply(init_constant)
print("net first Linear wight: ", net[0].weight.data[0])
print("net1 first Linear wight: ", net1.myHidden.weight.data[0])
X4 = X.clone().detach()
print("X4 org value: ", X4)
print("X4 after net: ", net(X4))
X5 = X.clone().detach()
print("X5 org value: ", X5)
print("X5 after net: ", net(X5))

print("-----------------------------------------------------------------------------")


# 4、自定义顺序块
# nn.Sequential是一种内置的顺序块，如果想自定义实现类似的功能需要实现如下两点
# a. 实现追加功能：一种将块逐个追加到列表中的函数
# b. 顺序执行的传播函数：一种前向传播函数，用于将输入按追加块的顺序传递给块组成的“链条”

class MySequential(nn.Module):
    def __init__(self):
        super().__init__()

    # 追加函数
    def build(self, subModel):
        indexStr = str(len(self._modules))
        # 这里，subModel是Module子类的一个实例
        # 我们把它保存在'Module'类的成员变量_modules中（MySequential也是Module类，因为MySequential继承了Module）
        # _module的类型是OrderedDict
        self._modules[indexStr] = subModel
        return self

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X


net2 = MySequential().build(nn.Linear(20, 256)).build(nn.ReLU()).build(nn.Linear(256, 10))
X6 = X.clone().detach()
print("X6 org value: ", X6)
print("X6 after net: ", net2(X6))
net2.apply(init_constant)
print("change net2 by init_constant...")
X7 = X.clone().detach()
print("X7 org value: ", X7)
print("X7 after net: ", net2(X7))

print("-----------------------------------------------------------------------------")


# 5、在前向传播函数中执行代码
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


X8 = X.clone().detach()
net3 = FixedHiddenMLP()
print("X8 org value: ", X8)
print("X8 after net: ", net3(X8))


# 混合搭配各种组合块
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


X9 = X.clone().detach()
net4 = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())

print("X9 org value: ", X9)
print("X9 after net: ", net4(X9))
