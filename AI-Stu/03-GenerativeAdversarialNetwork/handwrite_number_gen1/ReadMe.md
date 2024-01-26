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

设计要点：鉴别器和生成器网络模型反向同构，放置一个比另一个快

训练：

见[NumGanModelTrain.py](NumGanModelTrain.py)

一共6万轮，每轮用一个生成样本和一个真实样本训练鉴别器；用两个随机种子训练鉴别器

结果：

![NumGanModel.png](ReadMe%2FNumGanModel.png)

![NumGanModelResult.png](ReadMe%2FNumGanModelResult.png)

可以看到
* 有效果，输出的已不再是毫无规律的随机图片
* 模式崩溃，鉴别器进化速度远快于生成器，生成效果不明显

# 改进模型

改进网络模型，改进优化器，见[NumGanModel1.py](NumGanModel1.py)

鉴别器：

~~~
# 网络模型
nn.Flatten(),
nn.Linear(28 * 28, 200),
nn.LeakyReLU(0.02), # 此处将Sigmoid换成LeakyReLU
nn.Linear(200, 1),
nn.Sigmoid(),

# 优化器，从SGD换成Adam
self.DiscOptimiser = torch.optim.Adam(self.DiscModel.parameters(), lr=0.01)
~~~

生成器：

~~~
# 网络模型
nn.Linear(1, 200),
nn.LeakyReLU(0.02), # 此处将Sigmoid换成LeakyReLU
nn.Linear(200, 28 * 28),
nn.Sigmoid(),

# 优化器，从SGD换成Adam
self.GenOptimiser = torch.optim.Adam(self.GenModel.parameters(), lr=0.01)
~~~

训练方式不变，结果：

![NumGanModel1.png](ReadMe%2FNumGanModel1.png)

![NumGanModel1Result.png](ReadMe%2FNumGanModel1Result.png)

* 有效果，输出的是数字
* 模式崩溃，生成器找到了一个较优之后就不再进化