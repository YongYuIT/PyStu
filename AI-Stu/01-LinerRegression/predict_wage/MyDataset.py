from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, flie):
        # 从文件中加载数据到内存
        self.data = ""

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
