from PIL import Image
import ImageToSquare as I2S
import ResizeImage as RI
import ImageRemoveColor as IRC
import matplotlib.pyplot as plt


def ImageLeft2Right(image):
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return [image, flipped_image]


def test():
    image = Image.open('../pic/dog/0a2b7c5f-dd76-434b-996d-3b2494bd50a2.png')
    simage = I2S.ImageToSquare(image)
    rimage = RI.ResizeImage(simage, 100)
    cimage = IRC.ImageRemoveColor(rimage)

    # 左右翻转图像
    flipped_image = cimage.transpose(Image.FLIP_LEFT_RIGHT)

    # 设置子图的行列数
    num_rows = 1
    num_cols = 2
    # 创建子图并显示图片
    # figsize指定整个图像（包含若干个子图）的大小，单位是英寸
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 4))  # 调整图像大小
    axes[0].imshow(cimage)
    axes[1].imshow(flipped_image)
    plt.tight_layout()  # 调整子图布局，防止重叠
    plt.show()
