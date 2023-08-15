import math

import numpy as np
import torch
from torch.utils import data


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_linear_concise`"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# 加载训练/测试集
def load_data(totalSamples, trainSamplesRate, num_features, batch_size):
    trainSamplesNum = math.ceil(totalSamples * trainSamplesRate)

    # 构造随机的一组wight，这组wight就是拟合的目标
    wight = np.random.uniform(0.009, 0.011, size=(num_features, 1))
    print("top 10 of wight --> ", wight[:10])
    print("wight shape --> ", wight.shape)
    wightT = wight.reshape(1, -1)
    print("wightT shape --> ", wightT.shape)
    print("gen wight：", wightT)

    # 构造样本，totalSamples个，每个样本200个features，第i个样本xi=[xi1,xi2,...,xi200]T
    samples = np.random.normal(size=(num_features, totalSamples))
    print("samples shape --> ", samples.shape)

    # 构造lable y=wTx
    labels = np.dot(wightT, samples)
    print("lable shape --> ", labels.shape)
    # 加上随机噪音形成最终标签
    labels += np.random.normal(scale=0.01, size=labels.shape)

    # NumPy ndarray转换为tensor
    samples = torch.tensor(samples, dtype=torch.float32)
    samples = samples.t()
    print("samples to load shape->", samples.shape)
    labels = torch.tensor(labels, dtype=torch.float32)
    labels = labels.t()
    print("labels to load shape->", labels.shape)

    train_samples = samples[:trainSamplesNum]
    test_samples = samples[trainSamplesNum:]
    train_labels = labels[:trainSamplesNum]
    test_labels = labels[trainSamplesNum:]

    train_iter = load_array((train_samples, train_labels), batch_size)
    test_iter = load_array((test_samples, test_labels), batch_size, is_train=False)

    return train_iter, test_iter
