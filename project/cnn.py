import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
        self.fc1 = nn.Linear(25 * 25 * 32, num_classes)                #        30

    def forward(self, x):
        # run your data through the layers
        x = self.pool(F.dropout2d(F.relu(self.conv1(x))), 0.3)
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.dropout2d(F.relu(self.conv3_bn(self.conv3(x)))))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()