"""
Author: jhzhu
Date: 2024/7/10
Description: 
"""
from torch import nn
import torch


class SimpleRNN(nn.Module):
    def __init__(self, dict_len=5000, embedding_dim=256, n_classes=2):
        super().__init__()
        # 嵌入：词向量
        self.embed = nn.Embedding(num_embeddings=dict_len,
                                  embedding_dim=embedding_dim)
        # 循环神经网络提取特征
        self.rnn = nn.RNN(input_size=embedding_dim,
                          hidden_size=embedding_dim)
        # 转换输出
        self.out = nn.Linear(in_features=embedding_dim,
                             out_features=n_classes)

    def forward(self, x):
        # [seq_len, batch_size] --> [seq_len, batch_size, embedding_dim]
        x = self.embed(x)
        # out: [seq_len, batch_size, embedding_dim]
        # hn: [1, batch_size, embedding_dim]
        out, hn = self.rnn(x)
        # [1, batch_size, embedding_dim] --> [batch_size, embedding_dim]
        x = torch.squeeze(input=hn, dim=0)
        # [batch_size, embedding_dim] --> [batch_size, n_classes]
        x = self.out(x)
        return x
