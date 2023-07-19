import matplotlib.pyplot as plt


# 创建示例数据
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# 绘制散点图
# 最后一个参数，1，只用于指定绘制的数据点的大小。数值越大描点越大，反之越小
plt.scatter(x, y,1)

# 添加标题和坐标轴标签
plt.title("TestTitleByThinking")
plt.xlabel("X-test")
plt.ylabel("Y-test")

# 显示图表
plt.show()