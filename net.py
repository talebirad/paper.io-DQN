import torch.nn as nn
import torch.nn.functional as F

import densenet, resnet
#DQN = densenet.densenet121
DQN = resnet.resnet34

class myDQN(nn.Module):

    def __init__(self, pt = False):
        super(DQN, self).__init__()
        self.pt = pt
        self.conv = nn.Sequential(
            nn.Conv2d(7, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 5 * 5, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 8192),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(8192, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 3),
        )

    def forward(self, x):
        if self.pt:
            print(x.shape)
        x = self.conv(x)
        if self.pt:
            print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


