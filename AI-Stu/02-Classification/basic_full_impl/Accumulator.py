import torch
import Accuracy as thk_accuracy

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
def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        # 将模型设置为评估模式
        net.eval()
    # 正确预测数、预测总数
    # 得到[0.0,0.0]
    metric = Accumulator(2)
    with torch.no_grad():
        # 假设X,y在data_iter中有两次迭代（update）
        # 每次X中的样本个数都是10（即y的长度是10）
        # 第一次迭代，预测准确的个数是8
        # 第二次迭代，预测准确的个数是9
        # 那么会执行：
        # metric.add(8,10)
        # metric.add(9,10)
        # metric中dara为：[17.0, 20.0]
        for X, y in data_iter:
            # thk_accuracy.accuracy 计算模型net在输入X样本的情况下预测正确的数量。例如X有10个样本，如果其中9个预测正确，那么thk_accuracy.accuracy(net(X), y)输出9
            # numel：输出张量中元素总数
            metric.add(thk_accuracy.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]