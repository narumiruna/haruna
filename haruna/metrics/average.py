from __future__ import division

from .metric import Metric


class Average(Metric):

    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, value, number=1):
        self.sum += value * number
        self.count += number

    @property
    def value(self):
        if self.count == 0:
            return float('inf')
        else:
            return self.sum / self.count

    def __str__(self):
        return '{:.4f}'.format(self.value)


class ExponentialMovingAverage(Metric):

    def __init__(self, alpha: float = 0.5, bias_correct: bool = True):
        assert 0 < alpha < 1.0
        self.average = 0
        self.alpha = alpha
        self.bias_correct = bias_correct
        self.step = 0

    def update(self, value):
        self.step += 1

        self.average = self.alpha * self.average + (1 - self.alpha) * value

        if self.bias_correct:
            bias_correction = 1 - self.alpha**self.step
            self.average /= bias_correction

    @property
    def value(self):
        if self.step == 0:
            return float('inf')
        else:
            return self.average

    def __str__(self):
        return '{:.4f}'.format(self.value)
