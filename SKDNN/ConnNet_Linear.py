
import torch.nn as nn
import torch.nn.functional as F

label = 0


# Connected Network 1.1 , simple layers, edit some linear parameter size
class Net_convol(nn.Module):
    def __init__(self):
        super(Net_convol, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 100)
        self.classifier = nn.Linear(100, label)

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.classifier(x)
        return x

# Connected Network 1.2 , simple layers, mode layer

class Net_convol2(nn.Module):
    def __init__(self):
        super(Net_convol2, self).__init__()
        self.fc1 = nn.Linear(512, 480)
        self.fc2 = nn.Linear(480, 240)
        self.fc3 = nn.Linear(240, 120)
        self.fc4 = nn.Linear(120, 100)
        self.fc5 = nn.Linear(100, 60)
        self.classifier = nn.Linear(60, label)

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        x = self.classifier(x)
        return x