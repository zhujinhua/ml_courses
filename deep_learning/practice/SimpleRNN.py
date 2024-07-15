"""
Author: jhzhu
Date: 2024/7/10
Description: 
"""
from torch import nn
import torch


class SimpleRNN(nn.Module):
    def __init__(self, dict_len=5000, embedding_dim=256, n_classes=2, pad_index=-1):
        super().__init__()
        # 嵌入：词向量
        self.embed = nn.Embedding(num_embeddings=dict_len,
                                  embedding_dim=embedding_dim,
                                  padding_idx=pad_index)
        # 循环神经网络提取特征
        self.rnn = nn.RNN(input_size=embedding_dim,
                          hidden_size=embedding_dim,
                          batch_first=True)
        # 转换输出
        self.fc = nn.Linear(in_features=embedding_dim,
                             out_features=n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # [seq_len, batch_size] --> [seq_len, batch_size, embedding_dim]
        x = self.embed(x)
        # out: [seq_len, batch_size, embedding_dim]
        # hn: [1, batch_size, embedding_dim]
        out, hn = self.rnn(x)
        # [1, batch_size, embedding_dim] --> [batch_size, embedding_dim]
        out = self.fc(out[:, -1, :])
        # [batch_size, embedding_dim] --> [batch_size, n_classes]
        out = self.sigmoid(out)
        return out
