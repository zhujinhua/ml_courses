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
_std = np.std(X, axis=0) + 1e-9
X = (X - _mean) / _std
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)


class IrisDataset(Dataset):
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


iris_dataset = IrisDataset(X_train, y_train)
iris_train_dataloader = DataLoader(dataset=iris_dataset, batch_size=8, shuffle=True)
test_dataset = IrisDataset(X_test, y_test)
iris_test_dataloader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=True)


class Model(nn.Module):
    # get the hyperparameters, define the process layer, matrix transform
    def __init__(self, n_features, n_classes):
        # must do
        super().__init__()
        self.linear1 = nn.Linear(n_features, 100)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(100, n_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


customized_model = Model(n_features=4, n_classes=3)
epochs = 300
learning_rate = 1e-2  # gradient explosion, need preprocess data, then lower the lr
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=customized_model.parameters(), lr=learning_rate)


def get_acc(dataloader):
    customized_model.eval()  # define the model the evaluate module(latchNorm, layerNorm, Dropout, batch normalization)????
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            y_pred = customized_model(x)
            _, predicted = torch.max(y_pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total


train_accs = []
test_accs = []
for epoch in range(epochs):
    customized_model.train()
    for x, y in iris_train_dataloader:
        y_pred = customized_model(x)
        loss = loss_fun(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_acc = get_acc(iris_train_dataloader)
    test_acc = get_acc(iris_test_dataloader)  # test not need train
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
