import sys

sys.path.append('../Task1')
from DataPreprocessing import ImageToSquare as I2S
from DataPreprocessing import ResizeImage as RI
from DataStdRead import ImgClassDataSet as ICDS

from Data import ImagesSaveToTensers as ISTT
from ModelDesign import LeNetModelDef as MD
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch

modelLoad = MD.LeNetModelDef()
modelLoad.loadModel("LeNetModelDef_bak0111")
types = list(ICDS.labelDict.keys())

rootPath = "../Task1/pic_check"
fileList = os.listdir(rootPath)
fileCount = len(fileList)
# 设置子图的行列数
num_cols = 10  # 列
num_rows = int(fileCount / num_cols)  # 行
# 创建子图并显示图片
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), subplot_kw={'aspect': 'auto'})  # 调整图像大小

index = 0
modelLoad.net.eval()  # 将模型设置为评估模式
with torch.no_grad():
    for picFile in fileList:
        picFilePath = os.path.join(rootPath, picFile)
        if os.path.isfile(picFilePath) and picFilePath.endswith(".png"):
            image = Image.open(picFilePath)
            simage = I2S.ImageToSquare(image)
            rimage = RI.ResizeImage(simage, 128)
            x = ISTT.transform(rimage).view(1, 3, 128, 128)
            y_hat = modelLoad.net(x)
            type = types[y_hat.argmax(axis=1)]
            print("file-->", picFilePath, "||type-->", type)
            col = index % num_cols
            row = int(index / num_cols)
            index = index + 1
            axes[row, col].imshow(rimage)
            axes[row, col].set_title(type)
plt.tight_layout()  # 调整子图布局，防止重叠
plt.show()
