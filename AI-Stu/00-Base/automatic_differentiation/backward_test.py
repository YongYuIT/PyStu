import torch

# 新建一个[0., 1., 2., 3.]T的一维张量（即向量）
x = torch.arange(4.0)
print("x org value: ", x)

# 用x.grad储存梯度数据
# 任何关于x的函数y=Ax，y对x的梯度矩阵等于A的转置
# 梯度矩阵中的任何一列的形状必须与x相同
# 如果y是标量的话，梯度矩阵需要跟x形状相同（试想一下，如果梯度矩阵与x的形状不同，x如何沿梯度方向移动以达到梯度下降？）
# 事实上，pytorch里面只允许对标量执行backward，通过标量的反向传播计算对张量x的梯度
x.requires_grad_(True)
print("x.grad: ", x.grad)

# y = 2*||x||^2 = 2*(xT*x) = 2*(x1^2+x2^2+...+xn^2)
y = 2 * torch.dot(x, x)
print("y= 2*||x||^2: ", y)
print("x.grad before y backward: ", x.grad)

# y关于x的梯度矩阵为: 4x
y.backward()
# 在x=[0., 1., 2., 3.]T处，梯度是[ 0.,  4.,  8., 12.]
print("x.grad after y backward: ", x.grad)

# 用梯度计算式4x校验backward正确性
print("x.grad==4*x ? : ", x.grad == 4 * x)

# 如果尝试对非标量进行backward求梯度会报错：grad can be implicitly created only for scalar outputs
x1 = torch.tensor([[1.], [2.], [3.], [4.]])
x2 = x1.clone().detach().T
print("x1 org value: ", x1)
print("x1 shape: ", x1.shape)
print("x2 org value: ", x2)
print("x2 shape: ", x2.shape)
x1.requires_grad_(True)
y1 = torch.matmul(x1, x2)
print("y1 value: ", y1)
# 下面的语句报错，grad can be implicitly created only for scalar outputs
# y1.backward()
# 下面将y1张量变成y2标量（y2是y1的L2范数）,再对y2反向传播就不会报错了
y2 = torch.norm(y1)
print("y2 value: ", y2)
y2.backward()
print("x1.grad: ", x1.grad)
