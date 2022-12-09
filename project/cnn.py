import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_classes=30):
        super(Model, self).__init__()
        # input = 224 x 224 x 3
        self.conv1 = nn.Conv2d(3, 10, 5, bias=False)          # 220 x 220 x 10
        self.pool = nn.MaxPool2d(2, 2)                       # 110 x 110 x 10
        self.conv2 = nn.Conv2d(10, 20, 5, bias=False)         # 106 x 106 x 20
        self.conv2_bn = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 32, 3, bias=False)         #  51 x  51 x 32
        self.conv3_bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(25 * 25 * 32, 120)                #        30
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 30)

    def forward(self, x):
        # run your data through the layers
        x = self.pool(F.dropout2d(F.relu(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x