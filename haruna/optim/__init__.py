import mlconfig

from .rmsprop import TFRMSprop

mlconfig.torch.register_torch_optimizers()
mlconfig.torch.register_torch_schedulers()
