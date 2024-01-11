import sys

sys.path.append('../Task1')
from DataStdRead import ImgClassDataSet as ICDS
from DataStdRead import ImgClassDataLoader as ICDL
from Tools import ShowDict as SD

from Data import ImagesSaveToTensers as ISTT
from ModelDesign import LeNetModelDef as MD
import os

# 读取图片，将图片存储为张量字典
if not os.path.exists(ISTT.allPicDictName):
    picRootPath = "../Task1/pic/"
    ISTT.ImagesSaveToTensers(picRootPath)
# 由张量字典创建数据集对象，用于模型读取数据
dataset = ICDS.ImgClassDataSet(ISTT.allPicDictName)
# 读取数据集，划分训练、测试集
batchSize = 100
train_data, test_data = ICDL.getDataLoader(dataset, batchSize)
# 定义和训练模型
learningRate = 0.8
numEpochs = 60
model = MD.LeNetModelDef(batchSize, learningRate, numEpochs)
dictTrainRecords = model.train(train_data, test_data)
SD.showDict("LeNetModelDef", "epoch", "test", dictTrainRecords)
# 保存模型
model.saveModel("LeNetModelDef")

# 加载模型，验证模型
modelLoad = MD.LeNetModelDef()
modelLoad.loadModel("LeNetModelDef")
correct_rate, loss = modelLoad.evaluate(test_data)
print("model check-->correct rate-->", correct_rate, "||loss-->", loss)
