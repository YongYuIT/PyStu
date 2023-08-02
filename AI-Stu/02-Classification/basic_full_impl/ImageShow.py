import torch
import matplotlib.pyplot as plt

# 绘制图像列表
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  # @save
    figsize = (num_cols * scale, num_rows * scale)
    # subplots：在一张大图中展示若干个小图。num_rows=2 num_cols=9 表示大图将展示两行，每行9个小图
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    # 由于上面subplots返回的是一个2*9的子图集合，flatten用于将这个集合转成一维数组
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes