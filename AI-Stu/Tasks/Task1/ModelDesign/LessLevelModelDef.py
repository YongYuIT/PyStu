from ModelDesign import DynamicLRModelDef as MD
from torch import nn


class LessLevelModelDef(MD.DynamicLRModelDef):
    def __init__(self, batch_size, learning_rate, num_epochs):
        super().__init__(batch_size, learning_rate, num_epochs)
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.Linear(10000, 5000),
                                 nn.ReLU(),
                                 nn.Linear(5000, 2500),
                                 nn.ReLU(),
                                 nn.Linear(2500, 1250),
                                 nn.ReLU(),
                                 nn.Linear(1250, 625),
                                 nn.ReLU(),
                                 nn.Linear(625, 5)
                                 )
