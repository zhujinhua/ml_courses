"""
Author: jhzhu
Date: 2024/7/10
Description: 
"""
import jieba
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from deep_learning.practice.SimpleRNN import SimpleRNN


def get_vocabulary_list(msgs):
    vocab_set = set()
    for i in msgs:
        vocab_set |= set(i)
        # vocab_set |= set(i) - stopwords_set
    return sorted(list(vocab_set))


# every row is a message vector
def messages_2_vectors(vocab_list, msgs):
    msgs_len = len(msgs)
    shape = (msgs_len, len(vocab_list))
    matrix = np.zeros(shape)

    for i in range(msgs_len):
        for word in msgs[i]:
            if word in vocab_list:
                matrix[i, vocab_list.index(word)] = 1
    return matrix


class TakeAwayDataset(Dataset):
    def __init__(self, X, y):
        if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
            self.X = X.reset_index(drop=True)
            self.y = y.reset_index(drop=True)
        else:
            raise ValueError("X must be a pandas DataFrame and y must be a pandas Series")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        if item >= len(self.X):
            raise IndexError(f"Item index {item} is out of bounds")
        x = self.X.iloc[item]
        y = self.y.iloc[item]
        x = torch.tensor(data=x, dtype=torch.long)
        y = torch.tensor(data=[y], dtype=torch.long)
        return x, y


emotion_file = '../../dataset/中文外卖评论数据集.csv'
take_away_df = pd.read_csv(emotion_file)
take_away_df['words'] = take_away_df['review'].apply(lambda x: jieba.lcut(x.replace(' ', ''), cut_all=False))
vocabulary_set = get_vocabulary_list(take_away_df['words'])
take_away_vecs = messages_2_vectors(vocabulary_set, take_away_df['words'])
X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(take_away_vecs), take_away_df.loc[:, 'label'], test_size=0.3,
                                                    random_state=42, shuffle=True)

take_away_dataset = TakeAwayDataset(X_train, y_train)
take_away_train_dataloader = DataLoader(dataset=take_away_dataset, batch_size=16, shuffle=True)
test_dataset = TakeAwayDataset(X_test, y_test)
take_away_test_dataloader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

customized_model = SimpleRNN(dict_len=5000, embedding_dim=256, n_classes=2)
epochs = 100
learning_rate = 1e-3  # gradient explosion, need preprocess data, then lower the lr
loss_fun = nn.CrossEntropyLoss()
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
    for x, y in take_away_train_dataloader:
        y_pred = customized_model(x)
        loss = loss_fun(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = get_loss(take_away_train_dataloader)
    test_loss = get_loss(take_away_test_dataloader)  # test not need train
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print('Epoch: %s: train loss: %s, test loss: %s' % (epoch, train_loss, test_loss))

plt.plot(train_losses, label=f'train loss lr {learning_rate}', c='blue')
plt.plot(test_losses, label=f'test loss lr {learning_rate}', c='red')
plt.title('Take-away data losses')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
