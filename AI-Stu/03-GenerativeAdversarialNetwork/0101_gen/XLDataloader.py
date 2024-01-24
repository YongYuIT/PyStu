from torch.utils.data import DataLoader

from XLDataSet import XLDataSet


def test():
    train_data_set = XLDataSet(100, 0.3)
    train_loader = DataLoader(train_data_set, batch_size=10)
    for batch_X, batch_y in train_loader:
        print(batch_X.shape)
        print(batch_y)
        check = []
        for index in range(batch_y.size(0)):
            check.append(train_data_set.checkSample(batch_X[index]))
        print(check)


test()
