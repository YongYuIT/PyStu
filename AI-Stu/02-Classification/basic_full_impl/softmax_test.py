import torch

# X是2*3的张量
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(X)
# 对X第一个维度进行求和压缩，keepdim=True表示压缩后仍然保持原张量的维度
Y = X.sum(0)
print(Y)
Y = X.sum(0, keepdim=True)
print(Y)
# 对X第二个维度进行求和压缩
Y = X.sum(1, keepdim=True)
print(Y)

def softmax(X):
    X_exp = torch.exp(X)
    print("torch.exp(X)")
    print(X_exp)
    partition = X_exp.sum(1, keepdim=True)
    print("X_exp.sum")
    print(partition)
    return X_exp / partition

print("check softmax--------------------")
x=torch.tensor([[1,2,3],[4,5,6]])
print(x)
y=softmax(x)
print(y)