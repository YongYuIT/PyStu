import matplotlib.pyplot as plt


def showDict(title, x_name, y_name, dict_points):
    keys = list(dict_points.keys())
    values = list(dict_points.values())
    # 绘制折线图
    plt.plot(keys, values)
    # 设置图表标题和坐标轴标签
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    # 显示网格线
    plt.grid(True)
    # 显示图表
    plt.show()
