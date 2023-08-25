import numpy as np
import matplotlib.pyplot as plt

def printDemo():
    # 创建网格点
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    x, y = np.meshgrid(x, y)

    # 计算对应的 z 值
    z = x ** 2 + 2 * y ** 2 + 3

    # 创建 3D 图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制二次曲面
    ax.plot_surface(x, y, z, cmap='viridis')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置图形标题
    ax.set_title('quadric surface: z = x^2 + 2y^2 + 3')