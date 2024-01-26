# 最初的GAN模型

见[NumGanModel.py](NumGanModel.py)

鉴别器：

~~~
# 网络模型
nn.Flatten(),
nn.Linear(28 * 28, 200),
nn.Sigmoid(),
nn.Linear(200, 1),
nn.Sigmoid(),

# 优化器
self.DiscOptimiser = torch.optim.SGD(self.DiscModel.parameters(), lr=0.01)
~~~

生成器：

~~~
# 网络模型
nn.Linear(1, 200),
nn.Sigmoid(),
nn.Linear(200, 28 * 28),
nn.Sigmoid()

# 优化器
self.GenOptimiser = torch.optim.SGD(self.GenModel.parameters(), lr=0.01)
~~~

损失函数：

~~~
self.DiscLoss = nn.MSELoss()
~~~

设计要点：鉴别器和生成器网络模型反向同构，放置一个比另一个快

训练：

见[NumGanModelTrain.py](NumGanModelTrain.py)

一共6万轮，每轮用一个生成样本和一个真实样本训练鉴别器；用两个随机种子训练鉴别器

结果：

![NumGanModel.png](ReadMe%2FNumGanModel.png)

![NumGanModelResult.png](ReadMe%2FNumGanModelResult.png)

可以看到
* 有效果，输出的已不再是毫无规律的随机图片
* 训练不平衡，鉴别器进化速度远快于生成器，生成效果不明显

# 改进模型-1

改进网络模型，改进优化器，见[NumGanModel1.py](NumGanModel1.py)

鉴别器：

~~~
# 网络模型
nn.Flatten(),
nn.Linear(28 * 28, 200),
nn.LeakyReLU(0.02), # 此处将Sigmoid换成LeakyReLU
nn.Linear(200, 1),
nn.Sigmoid(),

# 优化器，从SGD换成Adam，学习速率调整为0.0001
self.DiscOptimiser = torch.optim.Adam(self.DiscModel.parameters(), lr=0.0001)
~~~

生成器：

~~~
# 网络模型
nn.Linear(1, 200),
nn.LeakyReLU(0.02), # 此处将Sigmoid换成LeakyReLU
nn.Linear(200, 28 * 28),
nn.Sigmoid(),

# 优化器，从SGD换成Adam，学习速率调整为0.0001
self.GenOptimiser = torch.optim.Adam(self.GenModel.parameters(), lr=0.0001)
~~~

SGD：new_parameters = old_parameters - learing_rate * gradient

其中 learing_rate 是一个固定的数字，需要手动调整

Adam：参数更新公式跟SGD类似，不同的是learing_rate是自适应调整的，并且gradient也考虑了历史梯度的加权平均

Adam考虑历史梯度的意义在于给梯度一个“惯性”，使得梯度变化更加平滑，这样有利于模型冲出平坦区域，避免在平坦区域停滞不前。

总结：

* SGD适合小模型和小数据集
* Adam适用于模型和数据集


Sigmoid：容易导致梯度消失（输入较大/较小处，梯度为0），但是在二分类问题的输出层比较适合（因为其将输出映射到了0~1范围内）
LeakyReLU：避免了梯度消失，但是会将输出映射到负数空间（概率是没有负数的，所以不适合放到二分类问题的输出层）

训练方式不变，结果：

![NumGanModel1.png](ReadMe%2FNumGanModel1.png)

![NumGanModel1Result.png](ReadMe%2FNumGanModel1Result.png)

* 有效果，输出的是数字
* 模式崩溃，生成器找到了一个较优之后就不再进化

# 改进模型-2

损失函数采用二元交叉熵损失函数，更大程度上奖励和惩罚

~~~
self.DiscLoss = nn.BCELoss()
~~~

生成器和鉴别器新增标准化层

训练方式不变，结果：

![NumGanModel2.png](ReadMe%2FNumGanModel2.png)

![NumGanModel2Result.png](ReadMe%2FNumGanModel2Result.png)

模式崩溃依然存在，结构变得更加清晰

