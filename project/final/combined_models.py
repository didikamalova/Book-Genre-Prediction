import torch
import torch.nn as nn
import torch.nn.functional as F


class OneParam(nn.Module):
    def __init__(self):
        super(OneParam, self).__init__()
        self.alpha = nn.Parameter(data=torch.Tensor(1), requires_grad=True)

    def forward(self, x1, x2):
        return x1 + x2 * self.alpha


class ThirtyParam(nn.Module):
    def __init__(self):
        super(ThirtyParam, self).__init__()
        self.alpha = nn.Parameter(data=torch.Tensor(30), requires_grad=True)

    def forward(self, x1, x2):
        return x1 + x2 * self.alpha

