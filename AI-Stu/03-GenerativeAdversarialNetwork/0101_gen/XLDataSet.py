# 0101序列数据集
import random
from torch.utils.data import Dataset
import numpy as np

import torch


class XLDataSet(Dataset):

    # totalNum：数据集包含的样本总量
    # noiseRate：样本中噪声数据比例（不符合0101格式的样本数据比例）
    def __init__(self, _totalNum, noiseRate):
        self.totalNum = _totalNum
        self.noiseSampleNum = int(self.totalNum * noiseRate)
        trueSampleNum = self.totalNum - self.noiseSampleNum
        trueSamples = []
        for index in range(trueSampleNum):
            trueSamples.append(torch.FloatTensor([
                random.uniform(-0.1, 0.1),
                random.uniform(0.9, 1.1),
                random.uniform(-0.1, 0.1),
                random.uniform(0.9, 1.1),
                random.uniform(-0.1, 0.1),
                random.uniform(0.9, 1.1),
                random.uniform(-0.1, 0.1),
                random.uniform(0.9, 1.1)
            ]))
        noiseSamples = []
        for index in range(self.noiseSampleNum):
            isOkSample = True
            while isOkSample:
                genNoiseSample = torch.FloatTensor([
                    random.uniform(-2, 2),
                    random.uniform(-2, 2),
                    random.uniform(-2, 2),
                    random.uniform(-2, 2),
                    random.uniform(-2, 2),
                    random.uniform(-2, 2),
                    random.uniform(-2, 2),
                    random.uniform(-2, 2)
                ])
                isOkSample = self.checkSample(genNoiseSample)
                if not isOkSample:
                    noiseSamples.append(genNoiseSample)

        self.ids = np.arange(0, self.totalNum)
        np.random.shuffle(self.ids)
        self.samples = trueSamples + noiseSamples

    def __getitem__(self, index):
        id = self.ids[index]
        if id < self.totalNum - self.noiseSampleNum:
            return self.samples[id], float(1)
        else:
            return self.samples[id], float(0)

    def __len__(self):
        return self.totalNum

    # 验证样本是否符合规格
    def checkSample(self, sample):
        for index in range(sample.size(0)):
            if sample[index] <= 1.1 and sample[index] >= 0.9:
                sample[index] = 1
            if sample[index] <= 0.1 and sample[index] >= -0.1:
                sample[index] = 0
        std = torch.FloatTensor([0., 1., 0., 1., 0., 1., 0., 1.])
        return torch.equal(std, sample)


def test1():
    fake_data = torch.randn(4)
    print('fake_data-->', fake_data)

    real_data = torch.FloatTensor([0, 1, 0, 1])
    print('real_data-->', real_data)

    real_data_1 = torch.FloatTensor([
        random.uniform(-0.2, 0.2),
        random.uniform(0.8, 1.2),
        random.uniform(-0.2, 0.2),
        random.uniform(0.8, 1.2),
    ])
    print('real_data_1-->', real_data_1)

    for index in range(20):
        # uniform：一定范围内随机均匀分布
        print('uniform-->', index, '-->', random.uniform(0.1, 0.2))


# test1()


def test_check(X):
    for index in range(X.size(0)):
        if X[index] <= 1.1 and X[index] >= 0.9:
            X[index] = 1
        if X[index] <= 0.1 and X[index] >= -0.1:
            X[index] = 0

    std = torch.FloatTensor([0., 1., 0., 1., 0., 1., 0., 1.])
    return torch.equal(std, X)


def test2():
    testTrue = torch.FloatTensor([
        random.uniform(-0.1, 0.1),
        random.uniform(0.9, 1.1),
        random.uniform(-0.1, 0.1),
        random.uniform(0.9, 1.1),
        random.uniform(-0.1, 0.1),
        random.uniform(0.9, 1.1),
        random.uniform(-0.1, 0.1),
        random.uniform(0.9, 1.1)
    ])
    print("check testTrue-->", test_check(testTrue))
    testFalse = torch.FloatTensor([
        random.uniform(-2, 2),
        random.uniform(-2, 2),
        random.uniform(-2, 2),
        random.uniform(-2, 2),
        random.uniform(-2, 2),
        random.uniform(-2, 2),
        random.uniform(-2, 2),
        random.uniform(-2, 2)
    ])
    print("check testFalse-->", test_check(testFalse))
