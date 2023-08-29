# 本示例着重体验torch.nn.Module中参数初始化


from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

# 1、内置初始化方法

print("init_normal... ...")


def init_normal(m):
    if type(m) == nn.Linear:
        # 内置初始化方法 normal_
        nn.init.normal_(m.weight, mean=0, std=0.01)
        # 内置初始化方法 zeros_
        nn.init.zeros_(m.bias)


net.apply(init_normal)
print("net[0].weight values :", net[0].weight.data[0])
print("net[0].bias values :", net[0].bias.data[0])

net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))

print("init_constant... ...")


def init_constant(m):
    if type(m) == nn.Linear:
        # 内置初始化方法 constant_
        nn.init.constant_(m.weight, 1)
        # 内置初始化方法 zeros_
        nn.init.zeros_(m.bias)


net.apply(init_constant)
print("net[0].weight values :", net[0].weight.data[0])
print("net[0].bias values :", net[0].bias.data[0])

print("-----------------------------------------------------------------")

# 2、分别对不同块使用不同初始化方法

net = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1))


def init_xavier(m):
    if type(m) == nn.Linear:
        # 内置初始化方法 xavier_uniform_
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        # 内置初始化方法 constant_
        nn.init.constant_(m.weight, 42)


# 不同块分别初始化，不同于上面第1点用net.apply让所有块都用同一个初始化方法
net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)

print("-----------------------------------------------------------------")


# 3、自定义初始化

def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5


net.apply(my_init)
print(net[0].weight[:2])

# 直接初始化
print("init by index... ...")
net[0].weight.data[0][0] = 100
net[0].weight.data[1][:] += 1
print(net[0].weight[:2])
