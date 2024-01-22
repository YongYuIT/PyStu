import torch

inputX = torch.normal(0, 1, (3, 3, 3))
print(inputX)
print("max-->", torch.max(inputX).item(), "||min-->", torch.min(inputX).item())

inputX1 = torch.randint(1, 100, (3, 3, 3))
print('inputX1-->', inputX1)
