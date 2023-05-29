from Config import config, device
from Dataloader import x_train, train_loader, valid_loader
from MyModel import MyModel
from Trainer import trainer

model = MyModel(input_dim=x_train.shape[1]).to(device)  # put your model and data on the same computation device.
trainer(train_loader, valid_loader, model, config, device)
