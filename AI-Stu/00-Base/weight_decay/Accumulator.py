import torch


# 在n个变量上累加
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    # zip(self.data,args)：见test程序，将生成一个二元组集合
    # for a,b in zip(self.data,args)：遍历这个二元组集合，每次都赋值给a,b
    # a + float(b)：将遍历到的二元组元素相加
    # self.data = []：遍历相加得到的数组赋值给self.data
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 计算在指定数据集上模型的精度
def evaluate_loss(model, data_iter):  # @save
    """评估给定数据集上模型的损失"""
    metric = Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        out = model.net(X)
        y = y.reshape(out.shape)
        l = model.loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]
