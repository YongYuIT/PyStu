# 原始模型

* 模型：[NumGanModel.py](NumGanModel.py)
* 训练：[NumGanModelTrain.py](NumGanModelTrain.py)，6万次
* 表现：![NumGanModel.png](ReadMe%2FNumGanModel.png)![NumGanModelResult.png](ReadMe%2FNumGanModelResult.png)

# 优化1

* 模型：[NumGanModel1.py](NumGanModel1.py)，替换激活层；替换优化器
* 训练：[NumGanModelTrain.py](NumGanModelTrain.py)，24万次
* 表现：![NumGanModel1.png](ReadMe%2FNumGanModel1.png)![NumGanModel1Result.png](ReadMe%2FNumGanModel1Result.png)

# 优化2

* 模型：[NumGanModel2.py](NumGanModel2.py)，新增标准化层；替换损失函数
* 训练：[NumGanModelTrain.py](NumGanModelTrain.py)，48万次
* 表现：![NumGanModel2.png](ReadMe%2FNumGanModel2.png)![NumGanModel2Result.png](ReadMe%2FNumGanModel2Result.png)

增加训练次数到60万次

* 表现：![NumGanModel2-1.png](ReadMe%2FNumGanModel2-1.png)![NumGanModel2-1Result.png](ReadMe%2FNumGanModel2-1Result.png)

# 优化3

* 模型：[NumGanModel3.py](NumGanModel3.py)，扩展生成器随机种子（从单个值扩展到100个值）
* 训练：跳过

# 优化4

* 模型：[NumGanModel4.py](NumGanModel4.py)，进一步扩展生成器随机种子（rand替换成randn）
* 训练：[NumGanModelTrain3.py](NumGanModelTrain3.py)，60万次
* 表现：![NumGanModel4.png](ReadMe%2FNumGanModel4.png)![NumGanModel4Result.png](ReadMe%2FNumGanModel4Result.png)

增加训练次数到120万次
