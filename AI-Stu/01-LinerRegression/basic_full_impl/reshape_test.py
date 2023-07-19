import torch

y = torch.tensor([1, 2, 3, 4])
y_reshaped = y.reshape((-1, 1))

print(y)
print(y.shape)
print(y_reshaped)
print(y_reshaped.shape)