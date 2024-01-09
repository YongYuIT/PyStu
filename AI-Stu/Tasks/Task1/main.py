from DataStorage import ImagesSaveToTensers as ISTT
from DataStdRead import ImgClassDataSet as ICDS
from DataStdRead import ImgClassDataLoader as ICDL
from ModelDesign import LessLevelModelDef as MD
from Tools import ShowDict as SD
import os

# 读取图片，将图片存储为张量字典
if not os.path.exists(ISTT.allPicDictName):
    picRootPath = "pic/"
    ISTT.ImagesSaveToTensers(picRootPath)
# 由张量字典创建数据集对象，用于模型读取数据
dataset = ICDS.ImgClassDataSet(ISTT.allPicDictName)
# 读取数据集，划分训练、测试集
batchSize = 100
train_data, test_data = ICDL.getDataLoader(dataset, batchSize)
# 定义和训练模型
# learningRate=0.1意味着模型参数会以当前梯度的一个十分之一的比例进行更新
learningRate = 0.5
numEpochs = 5
model = MD.LessLevelModelDef(batchSize, learningRate, numEpochs)
dictTrainRecords = model.train(train_data, test_data)
SD.showDict("LessLevelModelDef", "epoch", "test", dictTrainRecords)
# 保存模型
model.saveModel("LessLevelModelDef")

# 加载模型，验证模型
modelLoad = MD.LessLevelModelDef()
modelLoad.loadModel("LessLevelModelDef")
correct_rate, loss = modelLoad.evaluate(test_data)
print("model check-->correct rate-->", correct_rate, "||loss-->", loss)
