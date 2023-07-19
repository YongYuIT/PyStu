import matplotlib.pyplot as plt

def set_figsize(figsize=(10, 6)):
    """设置matplotlib的图表尺寸"""
    plt.rcParams['figure.figsize'] = figsize

x = [1, 2, 3, 4, 5]
y = [10, 8, 6, 4, 2]

plt.plot(x, y)
plt.show()