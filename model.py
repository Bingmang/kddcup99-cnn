from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_dim, n_class):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(True),
            nn.Conv2d(6, 16, 3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(144, 512),
            nn.Linear(512, 256),
            nn.Linear(256, n_class)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
