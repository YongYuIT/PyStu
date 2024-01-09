from DataPreprocessing import ImageRemoveColor as IRC
from DataPreprocessing import ImageToSquare as I2S
from DataPreprocessing import ResizeImage as RI
from DataStorage import ImagesSaveToTensers as ISTT
from ModelDesign import LessLevelModelDef as MD
from DataStdRead import ImgClassDataSet as ICDS
from PIL import Image
import matplotlib.pyplot as plt
import os

modelLoad = MD.LessLevelModelDef()
modelLoad.loadModel("LessLevelModelDef-bak0109")
types = list(ICDS.labelDict.keys())

rootPath = "pic_check"
fileList = os.listdir(rootPath)
fileCount = len(fileList)
# 设置子图的行列数
num_cols = 10  # 列
num_rows = int(fileCount / num_cols)  # 行
# 创建子图并显示图片
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), subplot_kw={'aspect': 'auto'})  # 调整图像大小

index = 0
for picFile in fileList:
    picFilePath = os.path.join(rootPath, picFile)
    if os.path.isfile(picFilePath) and picFilePath.endswith(".png"):
        image = Image.open(picFilePath)
        simage = I2S.ImageToSquare(image)
        rimage = RI.ResizeImage(simage, 100)
        cimage = IRC.ImageRemoveColor(rimage)
        x = ISTT.transform(cimage)[0].view(1, -1)
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
