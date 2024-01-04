from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np


def ImageToSquare(image):
    width, height = image.size
    diff = abs(width - height)
    padding = (0, 0, diff, 0) if width <= height else (0, 0, 0, diff)
    squared_image = ImageOps.expand(image, border=padding, fill='black')
    return squared_image


def test():
    # 打开图像
    image = Image.open('../pic/dog/0a2b7c5f-dd76-434b-996d-3b2494bd50a2.png')
    # 将PIL图像转换为NumPy数组
    image_array = np.array(image)
    # 使用matplotlib.pyplot显示图像
    plt.imshow(image_array)
    plt.show()

    # 获取图像的宽度和高度
    width, height = image.size
    # 计算要添加的黑色填充的大小
    diff = abs(width - height)
    padding = (0, 0, diff, 0) if width <= height else (0, 0, 0, diff)
    # 使用PIL的pad函数填充黑色
    squared_image = ImageOps.expand(image, border=padding, fill='black')
    image_array = np.array(squared_image)
    plt.imshow(image_array)
    plt.show()
