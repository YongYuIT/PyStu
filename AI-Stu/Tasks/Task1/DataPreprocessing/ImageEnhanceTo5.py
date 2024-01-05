from PIL import Image, ImageEnhance

from DataPreprocessing import ImageLeft2Right as IL2R
from DataPreprocessing import ImageRemoveColor as IRC
from DataPreprocessing import ImageToSquare as I2S
from DataPreprocessing import ResizeImage as RI
import matplotlib.pyplot as plt


def ImageEnhanceTo5(image):
    enhancer = ImageEnhance.Brightness(image)
    brightened_image_1_1 = enhancer.enhance(1.1)
    brightened_image_1_2 = enhancer.enhance(1.2)
    brightened_image_1_3 = enhancer.enhance(1.3)
    brightened_image_1_4 = enhancer.enhance(1.4)
    return [image, brightened_image_1_1, brightened_image_1_2, brightened_image_1_3, brightened_image_1_4]


def test():
    image = Image.open('../pic/dog/0a2b7c5f-dd76-434b-996d-3b2494bd50a2.png')
    simage = I2S.ImageToSquare(image)
    rimage = RI.ResizeImage(simage, 100)
    cimage = IRC.ImageRemoveColor(rimage)
    lrimage = IL2R.ImageLeft2Right(cimage)

    # 设置子图的行列数
    num_rows = len(lrimage)
    num_cols = 5
    # 创建子图并显示图片
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 4))  # 调整图像大小

    for index in range(len(lrimage)):
        subImage = lrimage[index]
        # 创建一个亮度调整对象
        enhancer = ImageEnhance.Brightness(subImage)
        # 调整亮度（1.0为原始亮度，小于1.0降低亮度，大于1.0增加亮度）
        brightened_image_1_1 = enhancer.enhance(1.1)
        brightened_image_1_2 = enhancer.enhance(1.2)
        brightened_image_1_3 = enhancer.enhance(1.3)
        brightened_image_1_4 = enhancer.enhance(1.4)
        axes[index, 0].imshow(subImage)
        axes[index, 1].imshow(brightened_image_1_1)
        axes[index, 2].imshow(brightened_image_1_2)
        axes[index, 3].imshow(brightened_image_1_3)
        axes[index, 4].imshow(brightened_image_1_4)

    plt.tight_layout()  # 调整子图布局，防止重叠
    plt.show()