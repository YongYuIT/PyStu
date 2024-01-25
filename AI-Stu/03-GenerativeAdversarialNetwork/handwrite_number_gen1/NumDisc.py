# 判别器

import torch.nn as nn
import torch.optim


class NumDisc(nn.Module):
    def __init__(self):
        super().__init__()
        self.netWork = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 200),
            nn.Sigmoid(),
            nn.Linear(200, 1),
            nn.Sigmoid(),
        )
        self.netWork.to(torch.cuda.current_device())
        self.lossFunc = nn.MSELoss()
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self, X):
        return self.netWork.forward(X)

    def trainModel(self, num_epochs, train_iter, test_iter=None):
        dictTrainRecords = {}
        for epoch_index in range(num_epochs):
            self.trainEpoch(train_iter)
            if test_iter is not None:
                avgLoss = self.evaluate(test_iter)
                print("epoch index-->", epoch_index, "||avgLoss-->", avgLoss)
                dictTrainRecords[epoch_index] = [avgLoss]
        return dictTrainRecords

    def trainEpoch(self, train_iter):
        self.netWork.train()
        for X_batch, y_batch in train_iter:
            y_batch = y_batch.to(device='cuda', dtype=torch.float)
            X_batch = X_batch.to(device='cuda', dtype=torch.float)
            y_hat_batch = self(X_batch).squeeze()
            loss_batch = self.lossFunc(y_hat_batch, y_batch)
            self.optimiser.zero_grad()
            loss_batch.backward()
            self.optimiser.step()

    def evaluate(self, test_iter):
        self.netWork.eval()
        with torch.no_grad():
            totalLoss = 0.
            totalSamples = 0
            for X_batch, y_batch in test_iter:
                y_batch = y_batch.to(device='cuda', dtype=torch.float)
                X_batch = X_batch.to(device='cuda', dtype=torch.float)
                y_hat_batch = self(X_batch).squeeze()
                loss_batch = self.lossFunc(y_hat_batch, y_batch)
                totalSamples += y_batch.size(0)
                totalLoss += torch.sum(loss_batch).item()
        return totalLoss / totalSamples
