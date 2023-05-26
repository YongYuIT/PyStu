import torch
from torch.utils.data import DataLoader

from MyDataset import MyDataset
from MyModel import MyModel
import torch.nn as nn

file = ""
dataset = MyDataset(file)
# 每100个资料为一个分片
tr_set = DataLoader(dataset, batch_size=100, shuffle=True)
model = MyModel().to("cpu")
criterion = nn.MSELoss()
optimzer = torch.optim.SGD(model.parameters(), 0.1)  # 学习速率：0.1
# Training -----------------------------
for epoch in range(n_epochs):
    model.train()
    for x, y in tr_set:
        optimzer.zero_grad()
        x, y = x.to("cpu"), y = y.to("cpu")
        perd = model(x)
        loss = criterion(perd, y)
        loss.backward()
        optimzer.step()

# Validation -----------------------------

model.eval()
total_loss = 0
for x.y in dv_set:
    x, y = x.to("cpu"), y.to("cpu")
    with torch.no_grad():
        perd = model(x)
        loss = criterion(perd, y)
    total_loss += loss.cpu().item * len(x)
    avg_loss = total_loss / len(dv_set.dataset)

# Testing -----------------------------

model.eval()
perds = []
for x in tt_set:
    x = x.to("cpu")
    with torch.no_grad():
        pred = model(x)
        pred.append(pred.cpu())