import torch
import torch.nn as nn
# import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(6*6*64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        y = self.net1(x)
        y = torch.flatten(y, 1)
        y = nn.functional.relu(self.fc1(y))
        y = self.fc2(y)
        return y
