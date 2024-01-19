import matplotlib.pyplot as plt


def showDict(title, dict_points, x_name, y_names):
    plt.title(title)
    plt.xlabel(x_name)
    max_length = max(len(values) for values in dict_points.values())
    for index in range(max_length):
        subDict = {key: values[index] for key, values in dict_points.items()}
        name = y_names[index]
        keys = list(subDict.keys())
        values = list(subDict.values())
        # 绘制折线图
        plt.plot(keys, values, label=name)
    # 显示网格线
    plt.grid(True)
    # 添加图例
    plt.legend()
    # 显示图表
    plt.show()


def test():
    points = {
        1: [1, 2, 3],
        2: [4, 5, 6],
        3: [7, 8, 9],
        4: [10, 11, 12]
    }
    showDict("Start Status", points, "x", ["y1", "y2", "y3"])
