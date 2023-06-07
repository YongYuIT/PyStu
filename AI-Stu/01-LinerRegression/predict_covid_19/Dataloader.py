# Reading/Writing Data
import pandas as pd
# Pytorch
from torch.utils.data import DataLoader

from COVID19Dataset import COVID19Dataset
from CommTools import same_seed, train_valid_split, select_feat
from Config import config

same_seed(config['seed'])

# 训练数据大小：2699 x 118 (id + 37 states + 16 features x 5 days)
# 测试数据大小：1078 x 117 (without last day's positive rate)
train_data, test_data = pd.read_csv('../covid.train.csv').values, pd.read_csv('../covid.test.csv').values
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

# Print out the data size.
print(f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
test_data size: {test_data.shape}""")

# Select features
x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])

# Print out the number of features.
print(f'number of features: {x_train.shape[1]}')

train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), \
    COVID19Dataset(x_valid, y_valid), \
    COVID19Dataset(x_test)

# Pytorch data loader loads pytorch dataset into batches.
# shuffle：洗牌，对数据的顺序进行打乱
# pin_memory：是否将 Tensor 数据加载到固定的内存位置。在使用 GPU 进行训练时，启用 pin_memory=True 可以帮助减少数据从主机内存到 GPU 内存的复制时间，并提高数据加载的吞吐量
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
