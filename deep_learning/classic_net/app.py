"""
Author: jhzhu
Date: 2024/6/22
Description: 
"""
import os
import time

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from deep_learning.classic_net.ResNet50 import ResNet50
from deep_learning.classic_net.VGG16 import VGG16
from deep_learning.classic_net.HandWrittenNet import HandWrittenNet
from deep_learning.utils.LeNet import LeNet


class GestureDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.X[idx]
        img_label = int(self.y[idx])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(img_label, dtype=torch.long)
        return image, label

    def __len__(self):
        return len(self.X)


def get_gesture_transform(sequence):
    # transform
    return transforms.Compose([
        transforms.Resize(sequence),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量并归一化到 [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化到 [-1, 1]
    ])


def get_data_path_label(data_path):
    dataset_dict = dict()
    for label in os.listdir(data_path):
        dir_path = os.path.join(data_path, label)
        for file_path in os.listdir(dir_path):
            dataset_dict[os.path.join(dir_path, file_path)] = label.split('G')[1]
    return dataset_dict


data_root = '../../dataset/gestures'
train_path = os.path.join(data_root, 'train')
test_path = os.path.join(data_root, 'test')

train_dataset_dict = get_data_path_label(train_path)
test_dataset_dict = get_data_path_label(test_path)


def get_dataloader(resize):
    train_dataset = GestureDataset(X=list(train_dataset_dict.keys()), y=list(train_dataset_dict.values()),
                                   transform=get_gesture_transform(resize))
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=4)

    test_dataset = GestureDataset(X=list(test_dataset_dict.keys()), y=list(test_dataset_dict.values()),
                                  transform=get_gesture_transform(resize))
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=4)
    return train_dataloader, test_dataloader


def get_acc(data_loader, model):
    accs = []
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            y_pred = model(X)
            y_pred = y_pred.argmax(dim=-1)
            acc = (y_pred == y).to(torch.float32).mean().item()
            accs.append(acc)
    final_acc = round(number=sum(accs) / len(accs), ndigits=5)
    return final_acc


def train(resize, model, name):
    train_accs = []
    test_accs = []
    cur_test_acc = 0
    train_dataloader, test_dataloader = get_dataloader(resize)
    train_acc = get_acc(data_loader=train_dataloader, model=model)
    test_acc = get_acc(data_loader=test_dataloader, model=model)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"训练之前：train_acc: {train_acc},test_acc: {test_acc}")

    for epoch in range(epochs):
        model.train()
        start_train = time.time()
        for X, y in train_dataloader:
            y_pred = model(X)
            loss = loss_fun(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        stop_train = time.time()
        train_acc = get_acc(data_loader=train_dataloader, model=model)
        test_acc = get_acc(data_loader=test_dataloader, model=model)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        if cur_test_acc < test_acc:
            cur_test_acc = test_acc
            torch.save(obj=model.state_dict(), f=f"{name}_best.pt")
        torch.save(obj=model.state_dict(), f=f"{name}_last.pt")

        print(f"""当前是第 {epoch + 1} 轮：
                --> train_acc: {train_acc},
                --> test_acc: {test_acc},
                --> elapsed_time: {round(number=stop_train - start_train, ndigits=3)}秒""")
    return train_accs, test_accs


def plot_training_result(resize, model, name):
    train_accs, test_accs = train(resize=resize, model=model, name=name)
    plt.plot(train_accs, label="train_acc")
    plt.plot(test_accs, label="train_acc")
    plt.legend()
    plt.grid()
    plt.xlabel(xlabel='epoch')
    plt.ylabel(ylabel="acc")
    plt.title(label="LeNet Training Process")
    plt.show()


def infer(resize, img_path, model):
    img = Image.open(fp=img_path)
    img = get_gesture_transform(resize)(img)
    if img.dim() == 3:
        img = img.unsqueeze(0)
    # 4，预处理
    # img = img.resize((32, 32))
    # img = np.array(img)
    # img = img / 255
    # img = (img - 0.5) / 0.5
    #
    # # 5, 转张量
    # img = torch.tensor(data=img, dtype=torch.float32)
    # img = img.permute(dims=(2, 0, 1))
    #
    # # 7, 新增一个批量维度
    # img = img.unsqueeze(dim=0)  # ???
    model.eval()
    with torch.no_grad():
        y_pred = model(img)
        return y_pred.argmax(dim=-1).item()


if __name__ == "__main__":
    classic_net_dict = {
        # 'lenet': (LeNet(), (32, 32)),
        # 'vgg16': (VGG16(), (224, 224)),
        # 'resnet50': (ResNet50(), (224, 224)),
        'handwrittenNet': ((HandWrittenNet()), (224, 224)),
    }
    for network, model_object in classic_net_dict.items():
        model = model_object[0]
        epochs = 20
        lr = 1e-3
        loss_fun = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
        plot_training_result(model_object[1], model=model, name=network)
        m1 = model_object[0]
        # load from pth file
        m1.load_state_dict(state_dict=torch.load(f=f"{network}_best.pt"),
                           strict=False)
        label = infer(resize=model_object[1], img_path="../../dataset/gestures/test/G5/IMG_1204.JPG",
                      model=m1)
        print(label)
