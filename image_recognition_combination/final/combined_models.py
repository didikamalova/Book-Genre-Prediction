import torch
import torch.nn as nn


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


class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.linear = nn.Linear(60, 30)

    def forward(self, x1, x2):
        return self.linear(torch.hstack([x1, x2]).to(dtype=torch.float32)).squeeze()
