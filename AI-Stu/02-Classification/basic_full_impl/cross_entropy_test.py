import torch

y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# 选取y_hat中第0和第1个元素，
# 选取第0个元素的第0和子元素，即0.1
# 选取第一个元素的第2个子元素，即0.5
print(y_hat[[0, 1], y])
# tensor([0.1000, 0.5000])




y = torch.tensor([0, 2, 4])
y_hat = torch.tensor(
    [[0.1, 0.2, 0.3, 0.4, 0.5], [1.1, 1.2, 1.3, 1.4, 1.5], [2.1, 2.2, 2.3, 2.4, 2.5], [3.1, 3.2, 3.3, 3.4, 3.5]])
print(y_hat[[0, 2, 3], y])
# tensor([0.1000, 2.3000, 3.5000])

y = torch.tensor([0, 2, 4, 1])
print(range(len(y_hat)))
print(y_hat[range(len(y_hat)), y])
# tensor([0.1000, 1.3000, 2.5000, 3.2000])

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])
print(cross_entropy(y_hat, y))