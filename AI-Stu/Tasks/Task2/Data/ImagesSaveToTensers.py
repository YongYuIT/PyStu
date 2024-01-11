import os
import torchvision.transforms as transforms
from Data import MainPreprocessing as DP
import torch
import matplotlib.pyplot as plt
import numpy as np

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
                        # 与Task1不同的是，这里需要将三个通道数据都加载进去
                        exImageTensor = transform(exImages[index])
                        print("exImageTensor channel-->", len(exImageTensor))
                        if len(exImageTensor) != 3:
                            print("this is not RBG pic!!!!!")
                        else:
                            allPicDict[exImageName] = exImageTensor
    if os.path.exists(allPicDictName):
        os.remove(allPicDictName)
    torch.save(allPicDict, allPicDictName)


def test():
    picRootPath = "../tmp_pic/"
    ImagesSaveToTensers(picRootPath)


def testRead():
    test()
    allPicDict = torch.load(allPicDictName)
    imageNameList = list(allPicDict.keys())

    # 设置子图的行列数
    num_cols = 20  # 列
    num_rows = int(len(imageNameList) / num_cols)  # 行
    # 创建子图并显示图片
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), subplot_kw={'aspect': 'auto'})  # 调整图像大小

    for index in range(len(imageNameList)):
        col = index % num_cols
        row = int(index / num_cols)
        imageData = allPicDict[imageNameList[index]].numpy()
        # 调整形状，将 (3, 128, 128) 转换为 (128, 128, 3)
        imageDataTrans = np.transpose(imageData, (1, 2, 0))
        axes[row, col].imshow(imageDataTrans)
        title = imageNameList[index]
        print("get title-->", title)
        shotTitle = ""
        if len(title) > 10:
            shotTitle = title[:5] + "..." + title[len(title) - 5:]
        axes[row, col].set_title(shotTitle)

    plt.show()
