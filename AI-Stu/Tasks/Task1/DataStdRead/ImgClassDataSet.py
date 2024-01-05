import torch
from torch.utils.data import Dataset, DataLoader
from DataStorage import ImagesSaveToTensers as ISTT
import matplotlib.pyplot as plt


# ImgClassDataSet 继承自 Dataset
class ImgClassDataSet(Dataset):

    def __init__(self, dictPath):
        allPicDict = torch.load(dictPath)
        self.keys = list(allPicDict.keys())  # 字典的键作为索引
        self.values = list(allPicDict.values())  # 字典的值作为数据

    def __len__(self):
        return len(self.keys)  # 返回数据集的长度

    def __getitem__(self, index):
        key = self.keys[index]  # 获取对应索引的键
        value = self.values[index]  # 获取对应索引的值
        type = key.split("_")[0]
        return type, value  # 返回键值对作为数据集的元素


def test():
    testDicr = {"a": 1, "b": 2}
    print(list(testDicr.keys())[0])
    print(len(testDicr))
    print("---------------------------------------")
    # 创建数据集对象
    dataset = ImgClassDataSet(ISTT.allPicDictName)
    # 创建 DataLoader
    batch_size = 5
    shuffle = True  # 是否在每个 epoch 中打乱数据
    num_workers = 0  # 多少个子进程用于数据加载
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # 遍历数据集
    for key, value in data_loader:
        print("key-->", key, "----value-->", value.size())
    print("---------------------------------------")
    # 检验对应关系
    for batch in data_loader:
        batch_keys = batch[0]
        batch_values = batch[1]
        # 设置子图的行列数
        num_cols = len(batch_keys)  # 列
        num_rows = 1  # 行
        # 创建子图并显示图片
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 4))  # 调整图像大小
        for index in range(num_cols):
            axes[index].imshow(batch_values[index].numpy())
            axes[index].set_title(batch_keys[index])
        plt.tight_layout()  # 调整子图布局，防止重叠
        plt.show()


test()
