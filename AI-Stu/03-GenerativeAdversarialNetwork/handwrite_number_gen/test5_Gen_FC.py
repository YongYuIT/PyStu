import torch

from ModelDesign import FCGenModelDef as FCG_MD
from ModelDesign import LeNetGPUModelDef as LN_MD
import sys

sys.path.append('../../Tasks/Task1')
from Tools import ShowDict as SD

modelLoad = LN_MD.LeNetGPUModelDef()
modelLoad.loadModel("LeNetGPUModelDef")

model = FCG_MD.FCGenModelDef(modelLoad, -30, 10000)
model.initModel()

# inputX = torch.normal(0, 1, (1000, 1000, 100))
inputX = torch.rand(1000, 1000, 100)
# inputTestX = torch.normal(0, 1, (500, 1000, 100))
inputTestX = torch.rand(300, 1000, 100)

dictTrainRecords = model.train_model(inputX, inputTestX)

# 保存模型
model.saveModel("FCGenModelDef")

SD.showDict("FCGenModelDef", dictTrainRecords, "epoch", "test")
