import torch

# 新建一个[0., 1., 2., 3.]T的一维张量（即向量）
x = torch.arange(4.0)
print(x)

# 用x.grad储存梯度数据
# 任何关于x的函数y=Ax，y对x的梯度矩阵等于A的转置
# 梯度矩阵中的任何一列的形状必须与x相同
x.requires_grad_(True)
print(x.grad)

# y = 2*||x||^2 = 2*(xT*x) = 2*(x1^2+x2^2+...+xn^2)
y = 2 * torch.dot(x, x)
print(y)

# y关于x的梯度矩阵为: 4x
y.backward()
# 在x=[0., 1., 2., 3.]T处，梯度是[ 0.,  4.,  8., 12.]
print(x.grad)

# 用梯度计算式4x校验backward正确性
print(x.grad==4*x)
