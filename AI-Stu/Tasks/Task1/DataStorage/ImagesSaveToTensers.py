import os
import torchvision.transforms as transforms
from DataPreprocessing import MainPreprocessing as DP
import torch
import matplotlib.pyplot as plt

# 定义转换操作
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
])

allPicDictName = 'pic_dict'


def ImagesSaveToTensers(picRootPath):
    allPicDict = {}
    for type in os.listdir(picRootPath):
        print("try dir-->", type)
        type_path = os.path.join(picRootPath, type)
        if os.path.isdir(type_path):
            print("handle dir-->", type)
            for picFile in os.listdir(type_path):
                print("try file-->", picFile)
                pic_file_path = os.path.join(type_path, picFile)
                if os.path.isfile(pic_file_path) and pic_file_path.endswith(".png"):
                    print("handle file-->", pic_file_path)
                    exImages = DP.DataPreprocessing(pic_file_path)
                    for index in range(len(exImages)):
                        exImageName = type + "_" + picFile.split(".")[0] + "_" + str(index)
                        # 特别注意，用这个方法转换张量形状是(channels, height, width)
                        # 其中channels是通道数，通常是3，分别表示RGB
                        # 由于这里图片是灰度图片，大小是100*100，所以张量形状是1*100*100
                        # 由于我们只需要黑白通道上的数据，所以此处需要取一下下标
                        exImageTensor = transform(exImages[index])[0]
                        allPicDict[exImageName] = exImageTensor
    if os.path.exists(allPicDictName):
        os.remove(allPicDictName)
    torch.save(allPicDict, allPicDictName)


def test():
    picRootPath = "../pic/"
    # 列出文件夹中的文件和子文件夹
    for item in os.listdir(picRootPath):
        item_path = os.path.join(picRootPath, item)
        if os.path.isdir(item_path):
            print("handle subpath:", item)

    picRootPath = "../tmp_pic/"
    ImagesSaveToTensers(picRootPath)


def testRead():
    test()
    allPicDict = torch.load(allPicDictName)
    imageNameList = list(allPicDict.keys())

    # 设置子图的行列数
    num_cols = 5  # 列
    num_rows = int(len(imageNameList) / num_cols)  # 行
    # 创建子图并显示图片
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 4))  # 调整图像大小

    for index in range(len(imageNameList)):
        col = index % num_cols
        row = int(index / num_cols)
        axes[row, col].imshow(allPicDict[imageNameList[index]].numpy())
        title = imageNameList[index]
        print("get title-->", title)
        shotTitle = ""
        if len(title) > 10:
            shotTitle = title[:5] + "..." + title[len(title) - 5:]
        axes[row, col].set_title(shotTitle)

    plt.tight_layout()  # 调整子图布局，防止重叠
    plt.show()
