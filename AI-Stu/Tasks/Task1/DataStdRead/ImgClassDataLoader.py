from torch.utils.data import DataLoader, random_split
from DataStdRead import ImgClassDataSet as ICDS
from DataStorage import ImagesSaveToTensers as ISTT
import matplotlib.pyplot as plt


# 每个batch计算完成之后都会进行update（正向传播，计算loss，反向传播，更新参数）
# 一个epoch会包含 总样本数/batchSize 个batch计算过程，每个epoch结束时模型会经历了整个数据集的训练
# 为了提高模型的泛化能力，一个模型训练通常会包含若干个epoch
def getDataLoader(dataset, batchSize):
    # 定义训练集和测试集的比例
    train_ratio = 0.8  # 假设70%用于训练集
    # 根据比例随机划分数据集
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    print("train_size-->", train_size, "||test_size-->", test_size)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 使用 DataLoader 加载训练集和测试集
    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)

    return train_loader, test_loader


def test():
    # 创建数据集对象
    dataset = ICDS.ImgClassDataSet(ISTT.allPicDictName)
    train_loader = getDataLoader(dataset, 5)[0]
    print("---------------------------------------")
    # 检验对应关系
    for batch in train_loader:
        batch_keys = batch[0]
        batch_values = batch[1]
        # 设置子图的行列数
        num_cols = len(batch_keys)  # 列
        num_rows = 1  # 行
        # 创建子图并显示图片
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 4))  # 调整图像大小
        if num_cols > 1:
            for index in range(num_cols):
                axes[index].imshow(batch_values[index].numpy())
                axes[index].set_title(batch_keys[index])
        else:
            axes.imshow(batch_values[0].numpy())
            axes.set_title(batch_keys[0])
        plt.tight_layout()  # 调整子图布局，防止重叠
        plt.show()
