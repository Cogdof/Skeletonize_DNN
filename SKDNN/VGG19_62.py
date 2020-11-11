import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn


label = 0

class Net(nn.Module):


    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            # 3 224 128
            nn.Conv2d(3, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 64 112 64
            nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 128 56 32
            nn.Conv2d(128, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 256 28 16
            nn.Conv2d(256, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 512 14 8
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )
        # 512 7 4

        self.avg_pool = nn.AvgPool2d(7)
        # 512 1 1

        # self.classifier = nn.Linear(512, label)

        self.fc1 = nn.Linear(512 * 2 * 2, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 512)

        self.classifier = nn.Linear(512, label)

    def forward(self, x):
        # print(x.size())
        features = self.conv(x)
        # print(features.size())
        x = self.avg_pool(features)
        x = x.view(features.size(0), -1)
        vector = x
        x = self.classifier(x)
        result = x

        # x = self.softmax(x)
        # result = self.softmax(x)
        # vector = self.last_vector

        return x, features, result, vector
