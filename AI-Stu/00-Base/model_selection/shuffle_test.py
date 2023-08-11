import numpy as np

# 生成符合正态分布（高斯分布）的随机数
features = np.random.normal(size=(10, 1))
print("features-->", features)
# 对数组进行原地随机排列
np.random.shuffle(features)
print("features shuffle-->", features)
