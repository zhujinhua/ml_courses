"""
    3，搭建一个可用于垃圾短信分类的循环神经网络模型（短信最长70个字，可借助 PyTorch）；
"""
import torch
from torch import nn


class Model(nn.Module):

    def __init__(self, 
                 dict_len=5000, 
                 embedding_dim=512):
        """
            假定：
                - 字典长度：5000
                - 特征长度：512
        """
        super(Model, self).__init__()
        self.embed = nn.Embedding(num_embeddings=dict_len,
                                  embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, 
                          hidden_size=embedding_dim,
                          num_layers=1,
                          batch_first=False)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=embedding_dim, out_features=2)

    def forward(self, x):
        x = self.embed(x)
        out, hn = self.gru(x)
        x = torch.relu(hn)
        x = torch.squeeze(input=x, dim=0)
        x = self.linear(x)
        return x