"""
Author: jhzhu
Date: 2024/9/25
Description: 
"""
import torch
from torch import nn
import jieba
import os
import glob
from torchvision import transforms


class ConvBlock(nn.Module):
    """
    Define a common conv block, including a conv layer, a batch normal layer, a relu layer
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class HandWrittenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ConvBlock(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.block2 = ConvBlock(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = self.block1(x)
        x = self.mp1(x)
        x = self.block2(x)
        x = self.mp2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class SpamModel(nn.Module):

    def __init__(self, dict_size, embedding_dim):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=dict_size, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=2)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features=embedding_dim, out_features=2)

    def forward(self, x):
        x = self.embed(x)
        output, hn = self.gru(x)
        x = self.dropout(hn[-1])
        x = self.fc(x)
        return x


def build_dict(data_path):
    words_set = set()
    for file_path in glob.glob(os.path.join(data_path, '*.txt')):
        with open(file_path, 'r') as file_reader:
            file_info = file_reader.read()
            words_set = words_set.union(set(jieba.lcut(file_info)))
    words_2_index_dict = {word: index for index, word in enumerate(words_set)}
    index_2_words_dict = {index: word for word, index in words_2_index_dict.items()}
    return words_2_index_dict, index_2_words_dict


def get_data_transforms():
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.2),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30)
    ])


if __name__ == "__main__":
    # words_2_index_dict, index_2_words_dict = build_dict('./corpus')
    help(transforms.ColorJitter)
    transforms = get_data_transforms()
