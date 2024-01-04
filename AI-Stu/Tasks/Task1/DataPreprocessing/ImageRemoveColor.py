from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def ImageRemoveColor(image):
    return image.convert('L')


def test():
    image = Image.open('../pic/dog/0a2b7c5f-dd76-434b-996d-3b2494bd50a2.png')
    # 将图像转换为黑白（灰度）图像
    bw_image = image.convert('L')
    image_array = np.array(bw_image)
    plt.imshow(image_array)
    plt.show()
