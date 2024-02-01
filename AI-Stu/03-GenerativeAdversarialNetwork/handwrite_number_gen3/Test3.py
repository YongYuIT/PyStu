import torch

test1 = torch.Tensor([1])
print('test1-->', test1.shape, "-->", test1)

for index in range(100):
    test2 = torch.rand((1, 28, 28))
    # print('test2-->', test2.shape, "-->", test2)
    print("test2 max-->", torch.max(test2).item(), " test2 min-->", torch.min(test2).item())
