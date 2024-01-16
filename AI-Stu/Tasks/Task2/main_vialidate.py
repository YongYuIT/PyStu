import sys

sys.path.append('../Task1')
from DataPreprocessing import ImageToSquare as I2S
from DataPreprocessing import ResizeImage as RI
from DataStdRead import ImgClassDataSet as ICDS

from Data import ImagesSaveToTensers as ISTT
from ModelDesign import LeNetGPUModelDef as MD
from PIL import Image
import os
import torch
import matplotlib.pyplot as plt

# 加载模型-----------------------------------------------------------------------------------------
modelLoad = MD.LeNetGPUModelDef()
modelLoad.loadModel("LeNetGPUModelDef-check-bak0115")
types = list(ICDS.labelDict.keys())

# 读取验证数据
rootPath = "pic_vad"
subPaths = os.listdir(rootPath)
imgTensorList = []
yTypeList = []
for subPath in subPaths:
    fullSubPath = os.path.join(rootPath, subPath)
    for imgFileName in os.listdir(fullSubPath):
        fullImgFilePath = os.path.join(fullSubPath, imgFileName)
        if os.path.isfile(fullImgFilePath) and fullImgFilePath.endswith(".png"):
            image = Image.open(fullImgFilePath)
            simage = I2S.ImageToSquare(image)
            rimage = RI.ResizeImage(simage, 128)
            imageTensor = ISTT.transform(rimage)
            if imageTensor.size(0) != 3:
                continue
            imgTensorList.append(imageTensor)
            yTypeList.append(subPath)

# 验证数据-----------------------------------------------------------------------------------------
modelLoad.net.eval()  # 将模型设置为评估模式
with torch.no_grad():
    X = torch.stack(imgTensorList, dim=0)
    X = X.to('cuda')
    y_hat = modelLoad.net(X)

# 统计验证结果-----------------------------------------------------------------------------------------
errorDict = {}
for imgIndex in range(y_hat.size(0)):
    y_hat_item = y_hat[imgIndex]
    max_value_index = torch.argmax(y_hat_item).item()
    type = types[max_value_index]
    if type != yTypeList[imgIndex]:
        errorDict[imgIndex] = type
print("error rate:", str(len(errorDict) / y_hat.size(0)))
# 将错误的结果显示出来-----------------------------------------------------------------------------------------
# 设置子图的行列数
num_cols = 10  # 列
num_rows = int(len(errorDict) / num_cols) + 1  # 行
# 创建子图并显示图片
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), subplot_kw={'aspect': 'auto'})  # 调整图像大小

index = 0
for imgIndex in errorDict.keys():
    imageTensor = imgTensorList[imgIndex]
    errorType = errorDict[imgIndex]
    trueType = yTypeList[imgIndex]
    col = index % num_cols
    row = int(index / num_cols)
    npImage = imageTensor.permute(1, 2, 0).numpy()
    axes[row, col].imshow(npImage)
    axes[row, col].set_title(errorType + "->" + trueType)
    index = index + 1

plt.tight_layout()  # 调整子图布局，防止重叠
plt.show()
