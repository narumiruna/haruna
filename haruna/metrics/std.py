import math

import torch

from .metric import Metric


class StandardDeviation(Metric):

    def __init__(self):
        self.sum = 0
        self.squared_sum = 0
        self.count = 0

    @property
    def value(self):
        return self.std

    @property
    def mean(self):
        return self.sum / self.count

    @property
    def var(self):
        return (self.squared_sum / self.count) - self.mean**2

    @property
    def std(self):
        return math.sqrt(self.var)

    @torch.no_grad()
    def update(self, tensor: torch.Tensor):
        self.sum += tensor.sum().item()
        self.squared_sum += tensor.pow(2).sum().item()
        self.count += tensor.numel()
