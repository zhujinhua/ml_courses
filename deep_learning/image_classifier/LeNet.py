import torch
from torch import nn
from image import ConvBlock


class LeNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=6,
                               kernel_size=5,
                               stride=1,
                               padding=0)
        self.feature_extraction = nn.Sequential(
            ConvBlock(in_channels=1, out_channels=6, kernel_size=5, strid=1, padding=0),
            #sub sampling
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            ConvBlock(in_channels=6, out_channels=16, kernel_size=5, strid=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(400, 120)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.linear3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        print(x.shape)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        # x = x.review(x.size(0), -1)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


model = LeNet()
X = torch.randn(2, 1, 32, 32)
print(model(X))
