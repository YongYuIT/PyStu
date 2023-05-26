import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net = nn.Sequential(
            # 9个feathers，32个神经元。这9个feathers分别是：
            # year-年份，age-年龄，sex-性别，maritl-婚否，race-种族，education-教育程度，jobclass-工作类型，heath-健康状况，heath_ins-是否有健康保险
            nn.Linear(9, 32),
            nn.Sigmoid(),
            # 预测的目标变量是一个：wage-工资
            nn.Linear(32, 1)
        )

    def forward(self,x):
        return self.net(x)
