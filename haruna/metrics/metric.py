from abc import ABCMeta
from abc import abstractproperty


def _to_value(x):
    if isinstance(x, Metric):
        x = x.value
    return x


class Metric(metaclass=ABCMeta):

    @abstractproperty
    def value(self):
        raise NotImplementedError

    def __gt__(self, other):
        return self.value > _to_value(other)

    def __lt__(self, other):
        return self.value < _to_value(other)

    def __le__(self, other):
        return self.value <= _to_value(other)

    def __ge__(self, other):
        return self.value >= _to_value(other)
