import torch

inputX = torch.torch.normal(0, 1, (3, 3, 3))
print(inputX)
print("max-->", torch.max(inputX).item(), "||min-->", torch.min(inputX).item())
