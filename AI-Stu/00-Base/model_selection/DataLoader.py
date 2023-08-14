# fix G:\UE_DEV\dev_tools_running\anaconda3\envs\test_py-3-9-17-fix-mxnet\lib\site-packages\mxnet\numpy\utils.py:37
from mxnet import gluon
import numpy as np
import math
import torch


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a Gluon data iterator.

    Defined in :numref:`sec_linear_concise`"""
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)


# 加载训练/测试集
def load_data(batch_size):
    max_degree = 20  # 多项式的最大阶数
    n_train, n_test = 100, 100  # 训练和测试数据集大小
    true_w = np.zeros(max_degree)  # 分配大量的空间
    print("true_w-->", true_w)
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

    features = np.random.normal(size=(n_train + n_test, 1))
    print("features shape-->", features.shape)
    np.random.shuffle(features)
    powers = np.arange(max_degree).reshape(1, -1)
    print("powers shape-->", powers.shape)
    # 将features里面的每个元素拿出来，做0~19的指数运算
    # features=[f1,f2,f3,...,f200]T
    # f1   power--> [f1^0  ,f1^1  ,f1^2  ,...,f1^19]
    # f2   power--> [f2^0  ,f2^1  ,f2^2  ,...,f2^19]
    # ...
    # f200 power--> [f200^0,f200^1,f200^2,...,f200^19]
    # fxi=fx^i，其中x属于1到200，i属于0到19
    poly_features = np.power(features, powers)
    print("poly_features shape-->", poly_features.shape)
    for i in range(max_degree):
        # fxi=fx^i/i!
        poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!
    # labels的维度:(n_train+n_test,)
    # lx=(fx^0/0!)*5 + (fx^1/1!)*1.2 + (fx^2/2!)*(-3.4) + (fx^3/3!)*5.6
    # 其中fx是f1~f200，符合正态分布（打乱）
    labels = np.dot(poly_features, true_w)
    # 加上正态分布的噪声
    # lx = (fx^0/0!)*5 + (fx^1/1!)*1.2 + (fx^2/2!)*(-3.4) + (fx^3/3!)*5.6 + 噪声
    # 其中fx是f1~f200，符合正态分布（打乱），表示200个样本
    # lx是l1~l200，表示200个样本对应的标签
    labels += np.random.normal(scale=0.1, size=labels.shape)

    # NumPy ndarray转换为tensor
    true_w, features, poly_features, labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]

    train_features = poly_features[:n_train, :4]
    test_features = poly_features[n_train:, :4]
    train_labels = labels[:n_train]
    test_labels = labels[n_train:]

    train_iter = load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    test_iter = load_array((test_features, test_labels.reshape(-1, 1)), batch_size, is_train=False)

    return train_iter, test_iter
