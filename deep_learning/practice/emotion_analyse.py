"""
Author: jhzhu
Date: 2024/7/10
Description: 
"""
import jieba
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader

from deep_learning.practice.SimpleRNN import SimpleRNN


def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        stopwords = set(f.read().splitlines())
    return stopwords


def get_vocabulary_list(msgs, stopwords_set):
    vocab_set = set()
    for i in msgs:
        # vocab_set |= set(i)
        vocab_set |= set(i) - stopwords_set
    vocab_set.add('<UNK>')
    vocab_set.add('<PAD>')
    return sorted(list(vocab_set))


# every row is a message vector
def messages_2_vectors(vocab_list, msgs, max_len=300):
    pad_index = vocab_list.index('<PAD>')
    unk_index = vocab_list.index('<UNK>')
    msgs_marix = []
    for msg in msgs:
        msg_vector = [vocab_list.index(m) if m in vocab_list else unk_index for m in msg]
        if len(msg) < max_len:
            msg_vector.extend([pad_index] * (max_len - len(msg)))
        else:
            msg_vector = msg_vector[:max_len]

        msgs_marix.append(msg_vector)
    return msgs_marix


class TakeAwayDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        if item >= len(self.X):
            raise IndexError(f"Item index {item} is out of bounds")
        x = self.X.iloc[item]
        y = self.y.iloc[item]
        x = torch.tensor(data=x, dtype=torch.long)
        y = torch.tensor(data=y, dtype=torch.long)
        return x, y


emotion_file = '../../dataset/中文外卖评论数据集.csv'
stopwords = load_stopwords('../../dataset/cn_stopwords.txt')
take_away_df = pd.read_csv(emotion_file)
take_away_df['words'] = take_away_df['review'].apply(lambda x: jieba.lcut(x.replace(' ', ''), cut_all=False))
vocabulary_set = get_vocabulary_list(take_away_df['words'], stopwords)
max_msg_len = 20 # take_away_df['words'].str.len().max()
dic_len = len(vocabulary_set)
take_away_vecs = messages_2_vectors(vocabulary_set, take_away_df['words'], max_msg_len)
X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(take_away_vecs), take_away_df.loc[:, 'label'], test_size=0.3,
                                                    random_state=42, shuffle=True)

take_away_dataset = TakeAwayDataset(X_train, y_train)
take_away_train_dataloader = DataLoader(dataset=take_away_dataset, batch_size=16, shuffle=True)
test_dataset = TakeAwayDataset(X_test, y_test)
take_away_test_dataloader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

customized_model = SimpleRNN(dict_len=dic_len)
epochs = 50
learning_rate = 1e-3  # gradient explosion, need preprocess data, then lower the lr
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=customized_model.parameters(), lr=learning_rate)


def get_acc(dataloader):
    customized_model.eval()  # define the model the evaluate module(latchNorm, layerNorm, Dropout, batch normalization)????
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            y_pred = customized_model(x)
            predicted = y_pred.argmax(dim=-1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total


train_accs = []
test_accs = []
for epoch in range(epochs):
    customized_model.train()
    for x, y in take_away_train_dataloader:
        y_pred = customized_model(x)
        loss = loss_fun(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_acc = get_acc(take_away_train_dataloader)
    test_acc = get_acc(take_away_test_dataloader)  # test not need train
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    print('Epoch: %s: train acc: %s, test loss: %s' % (epoch, train_acc, test_acc))

torch.save(obj=customized_model.state_dict(), f="./take_away.pt")

plt.plot(train_accs, label=f'train acc lr {learning_rate}', c='blue')
plt.plot(test_accs, label=f'test acc lr {learning_rate}', c='red')
plt.title('Take-away data losses')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()
plt.show()

# test on other review

X_test_review = ['口感不错', '口感不错，但送得太慢了', '不太好吃', '味道一般']
X_test_seg = [jieba.lcut(x, cut_all=False) for x in X_test_review]
X_test_vectors = messages_2_vectors(vocabulary_set, X_test_seg, max_msg_len)
model = SimpleRNN(dict_len=dic_len)
model.load_state_dict(state_dict=torch.load(f=f"./take_away.pt"))
model.eval()
result_dict = {0: '差评', 1: '好评'}
for test_vector in X_test_vectors:
    test_tensor = torch.tensor(data=test_vector, dtype=torch.long).unsqueeze(0)
    y_pred = model(test_tensor).argmax(dim=-1)
    print(y_pred.item())
    print(result_dict[y_pred.item()])
