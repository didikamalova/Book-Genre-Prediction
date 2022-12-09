import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_classes=30):
        super(Model, self).__init__()
        # input = 224 x 224 x 3
        self.conv1 = nn.Conv2d(3, 10, 5)          # 220 x 220 x 10
        self.pool1 = nn.MaxPool2d(2, 2)           # 110 x 110 x 10
        self.conv2 = nn.Conv2d(10, 20, 5)         # 106 x 106 x 20
        self.pool2 = nn.MaxPool2d(2, 2)           #  53 x  53 x 20
        self.conv3 = nn.Conv2d(20, 32, 3)         #  51 x  51 x 32
        self.pool3 = nn.MaxPool2d(2, 2)           #  25 x  25 x 32
        self.fc1 = nn.Linear(25 * 25 * 32, 30)    #        30

    def forward(self, x):
        # run your data through the layers
        x = self.pool1(F.dropout(F.relu(self.conv1(x))))
        x = self.pool2(F.dropout(F.relu(self.conv2(x))))
        x = self.pool3(F.dropout(F.relu(self.conv3(x))))
        x = x.view(-1, 25 * 25 * 32)
        x = self.fc1(x).squeeze()
        return x