import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,  # 随机数种子
    'select_all': True,  # 是否启用所有feature
    'valid_ratio': 0.2,  # 评估数据占比，评估数据条数=训练数据条数*评估数据占比
    'n_epochs': 3000,  # epochs的个数
    'batch_size': 256,
    'learning_rate': 1e-5,
    'early_stop': 400,  # 如果模型在这么多连续的 epoch 中没有改进，停止训练
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}
