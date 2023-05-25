import torch
from torch.utils.data import DataLoader

from MyDataset import MyDataset

file = ""
dataset = MyDataset(file)
dataloader = DataLoader(dataset, batch_size=5, shuffle=False)

print("def tensor---------------------------------------------")

x = torch.tensor([[1, -1], [-1, 1]])
print(x)

x = torch.zeros([2, 2])
print(x)

x = torch.ones([1, 2, 5])
print(x)

print("add sub 乘---------------------------------------------")

x = torch.tensor([[2., -3.], [-4., 2.]])
y = torch.tensor([[2., -4.], [-2., 6.]])

z = x + y
print(z)

z = x - y
print(z)

z = x @ y
print("x乘以y=")
print(z)

print("pow sum and mean---------------------------------------------")

z = x.pow(2)
print(z)

z = x.sum()
print(z)

print("y.mean")
y = torch.tensor([1, 2, 3, 4, 5])
z = y.float().mean()
print(z)

z = x.mean(dim=0)
print(z)

x = torch.randint(1, 10, [4, 2, 3, 5])  # 元素从1~10，四维，长度分别是4，2，3，5
print(x)
print(x.shape)
print(x.float().mean(dim=1))

print("zero and transpose---------------------------------------------")

x = torch.zeros([2, 3])
print(x)
print(x.shape)

x = x.transpose(0, 1)
print(x.shape)

print("squeeze and unsqueeze---------------------------------------------")

x = torch.zeros([4, 2, 3, 5])
y = x.squeeze(dim=0)
print(y)
print(y.shape)

x = torch.zeros([1, 2, 3, 5])  # only length=1 works
y = x.squeeze(dim=0)
print(y)
print(y.shape)

x = torch.zeros([4, 2, 3, 5])
y = x.unsqueeze(dim=0)
print(y)
print(y.shape)


##### cat #####