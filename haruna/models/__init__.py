import mlconfig
from torchvision import models

from .efficientnet import EfficientNet
from .lenet import LeNet
from .unet import UNet

mlconfig.register(models.segmentation.deeplabv3_resnet50)
