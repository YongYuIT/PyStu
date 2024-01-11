import sys

sys.path.append('../../Task1')
from DataPreprocessing import ImageLeft2Right as IL2R
from DataPreprocessing import ImageToSquare as I2S
from DataPreprocessing import ResizeImage as RI
from Data import ImageEnhanceToMany as IE
from PIL import Image
import matplotlib.pyplot as plt


def DataPreprocessing(picPath):
    image = Image.open(picPath)
    simage = I2S.ImageToSquare(image)
    rimage = RI.ResizeImage(simage, 128)
    lrimage = IL2R.ImageLeft2Right(rimage)
    eimage = []
    for index in range(len(lrimage)):
        subImage = lrimage[index]
        eimage.extend(IE.ImageEnhanceToMany(subImage, 0.5, 1.5))
    return eimage


def test():
    eimage = DataPreprocessing('../Task1/pic/dog/0a2b7c5f-dd76-434b-996d-3b2494bd50a2.png')
    # 设置子图的行列数
    num_cols = 5  # 列
    num_rows = int(len(eimage) / num_cols)  # 行
    # 创建子图并显示图片
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 4))  # 调整图像大小
    for index in range(len(eimage)):
        col = index % num_cols
        row = int(index / num_cols)
        axes[row, col].imshow(eimage[index])

    plt.tight_layout()  # 调整子图布局，防止重叠
    plt.show()
