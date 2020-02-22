from abc import ABCMeta
from abc import abstractmethod
from collections import OrderedDict

from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class AbstractTrainer(metaclass=ABCMeta):

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


class Trainer(AbstractTrainer):

    def __init__(self):
        self.modules = OrderedDict()
        self.status = OrderedDict()

    def __setattr__(self, name, value):
        modules = self.__dict__.get('modules')
        status = self.__dict__.get('status')

        if isinstance(value, (Module, Optimizer, _LRScheduler)):
            if modules is None:
                raise AttributeError("cannot assign module, optimizer or lr_scheduler before Trainer.__init__() call")
            modules[name] = value
        elif status is not None and name in status:
            status[name] = value
        else:
            super(Trainer, self).__setattr__(name, value)

    def __getattr__(self, name):
        if 'modules' in self.__dict__:
            modules = self.__dict__['modules']
            if name in modules:
                return modules[name]

        if 'status' in self.__dict__:
            status = self.__dict__['status']
            if name in status:
                return status[name]

        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    def register_status(self, name, value):
        status = self.__dict__.get('status')
        if status is None:
            raise AttributeError("cannot register status before Trainer.__init__() call")
        else:
            status[name] = value

    def load_state_dict(self, state_dict: dict):
        modules = state_dict['modules']
        for k, v in modules.items():
            self.modules[k].load_state_dict(v)

        status = state_dict['status']
        for k, v in status.items():
            self.status[k] = v

    def state_dict(self):
        return {'modules': {k: v.state_dict() for k, v in self.modules.items()}, 'status': self.status}
