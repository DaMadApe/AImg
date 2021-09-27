import torch
from torch import nn


class TimbiricheNet(nn.Module):
    def __init__(self):
        super(TimbiricheNet, self).__init__()
        self.nn_part1 = nn.Sequential([
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.Dropout()])

    def train(examples):
        pass

    def predict(self, board):
        pass

    def save(folder, filename):
        pass

    def load(folder, filename):
        pass