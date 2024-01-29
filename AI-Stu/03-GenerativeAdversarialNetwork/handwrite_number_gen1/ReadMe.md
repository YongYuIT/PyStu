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

对于单个样本
BCELoss(y_hat,y)=-[y*log(y_hat)+(1-y)*log(1-y_hat)]
多个样本的话，将上面的BCELoss取平均

一般二分问题中，一般y不是0就是1，y_hat在0到1之间

当y等于0时：loss= -log(1-y_hat) = log1/(1-y_hat)
当y_hat在[0,1)区间时，loss单调递增，y_hat等于0时loss最小为0

当y等于1时：loss = -log(y_hat) = log1/(y_hat)
当y_hat在[0,1)区间时，loss单调递减，y_hat等于1时loss最小为0

总之，y_hat越靠近y，越接近0；否则就会很大

~~~
self.DiscLoss = nn.BCELoss()
~~~

生成器和鉴别器新增标准化层

训练方式不变，结果：

![NumGanModel2.png](ReadMe%2FNumGanModel2.png)

![NumGanModel2Result.png](ReadMe%2FNumGanModel2Result.png)

模式崩溃依然存在，结构变得更加清晰

# 改进模型-3

将随机数种子从单个值扩展到100个值即：
* 改进前，一个float值对应一个图片
* 改进后，100个float值对应一个图片

训练方式不变，结果：

![NumGanModel3.png](ReadMe%2FNumGanModel3.png)

![NumGanModel3Result.png](ReadMe%2FNumGanModel3Result.png)

模式崩溃依然存在

# 改进模型-4

改-3中，随机数种子100个值从rand改进到randn

* randn：均值为0，标准差为1的正态分布随机
* rand：[0,1]区间上平均分布随机

训练加倍到20次，结果：

![NumGanModel4.png](ReadMe%2FNumGanModel4.png)

![NumGanModel4Result.png](ReadMe%2FNumGanModel4Result.png)

# 改进训练

在改-4中，加大训练次数后，发现图片有效信息快消失了。通过对比Demo发现是输入数据归一化的问题。

MNIST数据集本身就已经将图片数据归一化到[0,1]，无需更多的归一化处理，前面没有意识到这个问题，教训惨痛

以后每次训练之前都需要再三确认输入数据范围是否符合预期

本次改进通改-4，继续使用[NumGanModel4.py](NumGanModel4.py)模型，训练采用[NumGanModelTrain3.py](NumGanModelTrain3.py)

修正归一化的问题，训练4轮，结果：

![NumGanModel4-1.png](ReadMe%2FNumGanModel4-1.png)

![NumGanModel4-1Result.png](ReadMe%2FNumGanModel4-1Result.png)

完美解决模式崩溃问题

加大训练次数至10次