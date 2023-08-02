import torch

y = torch.tensor([[0, 1, 2, 3, 4], [0.4, 0.3, 0.2, 0.1, 0.0]])
print(y.shape)
print(y.shape[1])

print(y.argmax(axis=1))
# tensor([4, 0])
# 因为[0, 1, 2, 3, 4]中最大值是4，其索引是4，所以第一个值是4
# 因为[0.4, 0.3, 0.2, 0.1, 0.0]中最大值是0.4，其索引是0，所以第二个参数是0

print(y.dtype)
print(y.type(y.dtype))

print("----------------------------------------------------------------")
y=torch.tensor([0,0,1])
y_hat=torch.tensor([[0.2,0.3,0.5],[0.1,0.9,0],[0.1,0.01,0.99]])
y_hat = y_hat.argmax(axis=1)
print(y_hat)
print(y_hat.type(y.dtype))
cmp = y_hat.type(y.dtype) == y
print(cmp)
print(cmp.type(y.dtype).sum())
print(float(cmp.type(y.dtype).sum()))

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

print(accuracy(y_hat, y))