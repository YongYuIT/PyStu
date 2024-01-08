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


def test():
    dictPoints = {
        0: 0.1940726577437859,
        1: 0.19502868068833656,
        2: 0.19502868068833656,
        3: 0.1940726577437859,
        4: 0.19502868068833656,
        5: 0.19216061185468447,
        6: 0.1940726577437859,
        7: 0.19216061185468447,
        8: 0.19216061185468447,
        9: 0.1940726577437859,
        10: 0.1940726577437859,
        11: 0.19502868068833656,
        12: 0.19502868068833656,
        13: 0.19502868068833656,
        14: 0.19502868068833656,
        15: 0.19502868068833656,
        16: 0.19502868068833656,
        17: 0.19502868068833656,
        18: 0.1940726577437859,
        19: 0.1940726577437859,
    }
    showDict("Start Status", "epoch", "test", dictPoints)
