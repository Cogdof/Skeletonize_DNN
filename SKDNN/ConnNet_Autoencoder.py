import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

label = 0

class Net_Autencoder(nn.Module):
    def __init__(self):
        super(Net_Autencoder, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)
        self.fc3 = nn.Linear(10, 256)
        self.fc4 = nn.Linear(256, 100)
        self.classifier = nn.Linear(100, label)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.classifier(x)
        return x
