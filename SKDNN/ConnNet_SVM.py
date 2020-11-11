
import torch.nn as nn
import torch.nn.functional as F

label = 0


# Connected Network  , SVM
class Net_SVM(nn.Module):

    def __init__(self):
        super(Net_SVM, self).__init__()
        self.linear = nn.Linear(512, label)

    def forward(self, x):

        x = self.linear(x)

        return x

