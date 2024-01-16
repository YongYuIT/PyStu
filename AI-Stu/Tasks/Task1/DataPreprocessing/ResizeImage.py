from PIL import Image
from DataPreprocessing import ImageToSquare as I2S
import matplotlib.pyplot as plt
import numpy as np


def ResizeImage(image, tagSize):
    target_size = (tagSize, tagSize)
    return image.resize(target_size, Image.LANCZOS)


def test():
    image = Image.open('../pic/dog/0a2b7c5f-dd76-434b-996d-3b2494bd50a2.png')
    simage = I2S.ImageToSquare(image)
    # 定义目标尺寸（宽度，高度）
    target_size = (100, 100)
    resized_image = simage.resize(target_size, Image.ANTIALIAS)  # 使用抗锯齿方法调整大小
    image_array = np.array(resized_image)
    plt.imshow(image_array)
    plt.show()
