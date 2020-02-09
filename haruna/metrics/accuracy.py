from __future__ import division

import torch

from .metric import Metric


class Accuracy(Metric):

    def __init__(self):
        self.correct = 0
        self.count = 0

    @torch.no_grad()
    def update(self, output: torch.Tensor, target: torch.Tensor):
        self.correct += output.argmax(dim=1).eq(target).sum().item()
        self.count += target.numel()

    @property
    def value(self):
        return self.correct / self.count

    def __str__(self):
        return '{:.2f}%'.format(self.value * 100)
