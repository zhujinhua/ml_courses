import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

housing_file = '../ensemble/BostonHousing.csv'
housing_data = pd.read_csv(housing_file)
X = housing_data.iloc[:, :-1]
y = housing_data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


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
print(house_dataset[0])
house_train_dataloader = DataLoader(dataset=house_dataset, batch_size=32, shuffle=True)
for x, y in house_train_dataloader:
    print(x)
    print(y)
test_dataset = HouseDataset(X_test, y_test)
print(test_dataset[0])
house_test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)
