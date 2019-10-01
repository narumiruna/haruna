import torch

from .metric import Metric


class R2Score(Metric):

    def __init__(self):
        self.count = 0
        self.y_sum = 0
        self.y_sq_sum = 0
        self.res_sq_sum = 0

    def __str__(self):
        return '{:.4f}'.format(self.value)

    @property
    def value(self):
        if self.count == 0:
            return float('-inf')
        else:
            return 1 - self.res_sq_sum / (self.y_sq_sum - self.y_sum**2 / self.count)

    @torch.no_grad()
    def update(self, prediction, target):
        self.count += target.numel()
        self.y_sum += target.sum().item()
        self.y_sq_sum += target.pow(2).sum().item()
        self.res_sq_sum += (prediction - target).pow(2).sum().item()
