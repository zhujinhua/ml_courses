import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

housing_file = '../../dataset/BostonHousing.csv'
housing_data = pd.read_csv(housing_file)
X = housing_data.iloc[:, :-1]
y = housing_data.iloc[:, -1]
_mean = np.mean(X, axis=0)
_std = np.std(X, axis=0) + 1e-9
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
        x = self.X.iloc[item]
        y = self.y.iloc[item]
        x = torch.tensor(data=x, dtype=torch.float32)
        y = torch.tensor(data=[y], dtype=torch.float32)
        return x, y


house_dataset = HouseDataset(X_train, y_train)
house_train_dataloader = DataLoader(dataset=house_dataset, batch_size=16, shuffle=True)
test_dataset = HouseDataset(X_test, y_test)
house_test_dataloader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

# 1
model = nn.Linear(in_features=13, out_features=1)
# 2
model_seq = nn.Sequential(
    nn.Linear(in_features=13, out_features=1)
)


class Model(nn.Module):
    # get the hyperparameters, define the process layer, matrix transform
    def __init__(self, n_features):
        # must do
        super().__init__()
        self.linear1 = nn.Linear(in_features=n_features, out_features=8)
        self.linear2 = nn.Linear(in_features=8, out_features=1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)  # ???? break the linear space, activation fun
        x = self.linear2(x)
        return x


customized_model = Model(n_features=13)
epochs = 500
learning_rate = 1e-3  # gradient explosion, need preprocess data, then lower the lr
loss_fun = nn.MSELoss()
optimizer = torch.optim.SGD(params=customized_model.parameters(), lr=learning_rate)


def get_loss(dataloader):
    customized_model.eval()  # define the model the evaluate module(latchNorm, layerNorm, Dropout)????
    losses = []
    with torch.no_grad():
        for x, y in dataloader:
            y_pred = customized_model(x)
            loss = loss_fun(y_pred, y)
            losses.append(loss.item())
    final_loss = round(sum(losses) / len(losses), 5)
    return final_loss


train_losses = []
test_losses = []
for epoch in range(epochs):
    customized_model.train()
    for x, y in house_train_dataloader:
        y_pred = customized_model(x)
        loss = loss_fun(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = get_loss(house_train_dataloader)
    test_loss = get_loss(house_test_dataloader)  # test not need train
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print('Epoch: %s: train loss: %s, test loss: %s' % (epoch, train_loss, test_loss))

plt.plot(train_losses, label=f'train loss lr {learning_rate}', c='blue')
plt.plot(test_losses, label=f'test loss lr {learning_rate}', c='red')
plt.title('Housing Prices losses')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
