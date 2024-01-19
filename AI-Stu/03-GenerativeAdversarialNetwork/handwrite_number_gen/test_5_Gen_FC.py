import torch

from ModelDesign import FCGenModelDef as FCG_MD
from ModelDesign import LeNetGPUModelDef as LN_MD
import sys

sys.path.append('../../Tasks/Task1')
from Tools import ShowDict as SD

learningRate = 0.00001 # 模型数据必须从较小设置起，否则容易梯度爆炸出现nan
numEpochs = 100

modelLoad = LN_MD.LeNetGPUModelDef()
modelLoad.loadModel("LeNetGPUModelDef")

model = FCG_MD.FCGenModelDef(modelLoad, learningRate, numEpochs)
model.initModel()

inputX = torch.torch.normal(0, 1, (1000, 1000, 100))
inputTestX = torch.normal(0, 1, (500, 1000, 100))

dictTrainRecords = model.train_model(inputX, inputTestX)

SD.showDict("FCGenModelDef", dictTrainRecords, "epoch", "test")
