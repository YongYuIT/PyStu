import numpy as np
import torch

# 解析解为：
# y=w1x1+w2x2+w3x3+...+w200x200
# 其中w1~w200是随机生成
# 拟合的目的就是要尽可能拟合随机生成的这组w1~w200


# 构造随机的一组wight，这组wight就是拟合的目标
wight = np.random.normal(size=(200, 1))
print("top 10 of wight --> ", wight[:10])
print("wight shape --> ", wight.shape)
wightT = wight.reshape(1, -1)
print("wightT shape --> ", wightT.shape)

# 构造样本，2000个，每个样本200个features，第i个样本xi=[xi1,xi2,...,xi200]T
samples = np.random.normal(size=(200, 2000))
print("samples shape --> ", samples.shape)

# 构造lable y=wTx
labels = np.dot(wightT, samples)
print("lable shape --> ", labels.shape)
# 加上随机噪音形成最终标签
labels += np.random.normal(scale=0.1, size=labels.shape)

# NumPy ndarray转换为tensor
samples = torch.tensor(samples, dtype=torch.float32)
samples = samples.t()
print("samples to load shape->", samples.shape)
labels = torch.tensor(labels, dtype=torch.float32)
labels = labels.t()
print("labels to load shape->", labels.shape)