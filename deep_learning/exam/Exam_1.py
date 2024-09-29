"""
Author: jhzhu
Date: 2024/9/19
Description: 
"""
import os

import jieba
from torch import nn
import torch
import glob
from torchvision import transforms


class HouseModel(nn.Module):
    """
        Boston house model: apply two fully connected layers
    """
    def __init__(self, n_features):
        # firstly invoke parent init method
        super(HouseModel, self).__init__()
        self.linear1 = nn.Linear(in_features=n_features, out_features=8)
        self.linear2 = nn.Linear(in_features=8, out_features=1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


class ConvBlock(nn.Module):
    def __int__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__int__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class HandWrittenModel(nn.Module):
    def __int__(self):
        super(HandWrittenModel, self).__int__()
        self.block1 = ConvBlock(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.block2 = ConvBlock(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.linear1 = nn.Linear(in_features=64 * 7 * 7, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = self.block1(x)
        x = self.mp1(x)
        x = self.block2(x)
        x = self.mp2(x)

        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)

        return x


class SpamModel(nn.Module):
    def __int__(self, embedding_dict=5000, embedding_dim=512):
        super(SpamModel, self).__int__()
        self.embed = nn.Embedding(num_embeddings=embedding_dict, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=1, batch_first=False)
        self.linear = nn.Linear(in_features=embedding_dim, out_features=2)

    def forward(self, x):
        x = self.embed(x)
        output, hidden = self.gru(x)
        x = torch.relu(hidden)
        x = torch.squeeze(input=x, dim=0)
        x = self.linear(x)

        return x


def build_dict(corpus_dir='./corpus'):
    all_words = set()
    for file in glob.glob(os.path.join(corpus_dir, '*.txt')):
        with open(file, 'r') as file_reader:
            line = file_reader.read()
            words = jieba.lcut(line)
            all_words = all_words.union(set(words))
    word_2_index = {word: index for index, word in enumerate(sorted(all_words))}
    index_2_word = {index: word for word, index in word_2_index.items()}
    return word_2_index, index_2_word


def get_data_fusion():
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.1,
                               contrast=0.1,
                               saturation=0.1,
                               hue=0.2),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=30)
    ])


if __name__ == "__main__":
    # print(dir(nn))
    # house_model = HouseModel(n_features=13)
    # hand_written_model = HandWrittenModel()
    # word_2_index, index_2_word = build_dict(corpus_dir='./corpus')
    data_fusion = get_data_fusion()
