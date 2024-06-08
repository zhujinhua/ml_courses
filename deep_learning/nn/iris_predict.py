import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
_mean = np.mean(X, axis=0)
_std = np.mean(X, axis=0) + 1e-9
X = (X - _mean) / _std
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)


class HouseDataset(Dataset):
    # define a dataset of pricing
    def __init__(self, X, y):
        # get the parameters, define the fields
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        x = self.X[item]
        y = self.y[item]
        x = torch.tensor(data=x, dtype=torch.float32)
        y = torch.tensor(data=y, dtype=torch.long)
        return x, y


house_dataset = HouseDataset(X_train, y_train)
print(house_dataset[0])
house_train_dataloader = DataLoader(dataset=house_dataset, batch_size=8, shuffle=True)
for x, y in house_train_dataloader:
    print(x)
    print(y)
test_dataset = HouseDataset(X_test, y_test)
print(test_dataset[0])
house_test_dataloader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=True)

model = nn.Linear(in_features=4, out_features=1)

model_seq = nn.Sequential(
    nn.Linear(in_features=4, out_features=1)
)


class Model(nn.Module):
    # get the hyperparameters, define the process layer, matrix transform
    def __init__(self, n_features, n_classes):
        # must do
        super().__init__()
        self.linear1 = nn.Linear(in_features=n_features, out_features=3)

    def forward(self, x):
        x = self.linear1(x)
        return x


model_define = Model(n_features=4, n_classes=3)
epochs = 200
learning_rate = 1e-3  # gradient explosion, need preprocess data, then lower the lr
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_define.parameters(), lr=learning_rate)


def get_acc(dataloader):
    model_define.eval()  # define the model the evaluate module(latchNorm, layerNorm, Dropout)????
    losses = []
    for x, y in dataloader:
        y_pred = model_define(x)
        y_pred = y_pred.argmax(-1)
        loss = (y == y_pred).to(dtype=torch.float32).mean()
        losses.append(loss.item())
    final_loss = round(number=sum(losses) / len(losses), ndigits=5)
    return final_loss


train_accs = []
test_accs = []
for epoch in range(epochs):
    model_define.train()
    for x, y in house_train_dataloader:
        y_pred = model_define(x)
        loss = loss_fun(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_acc = get_acc(house_train_dataloader)
    test_acc = get_acc(house_test_dataloader)  # test not need train
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    print('Epoch: %s: train acc: %s, test acc: %s' % (epoch, train_acc, test_acc))


plt.plot(train_accs, label=f'train acc lr {learning_rate}', c='blue')
plt.plot(test_accs, label=f'test acc lr {learning_rate}', c='red')
plt.title('Iris Accuracy')
plt.xlabel('epoch')
plt.ylabel('Acc')
plt.legend()
plt.show()
