import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image

# https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py


class Compose(T.Compose):

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class ToTensor(T.ToTensor):

    def __call__(self, img, target):
        img = F.to_tensor(img)
        target = torch.as_tensor(np.asarray(target), dtype=torch.int64)
        return img, target


class Normalize(T.Normalize):

    def __call__(self, tensor, target):
        return F.normalize(tensor, self.mean, self.std, self.inplace), target


class RandomHorizontalFlip(T.RandomHorizontalFlip):

    def __call__(self, img, target):
        if random.random() < self.p:
            return F.hflip(img), F.hflip(target)
        return img, target


class RandomVerticalFlip(T.RandomVerticalFlip):

    def __call__(self, img, target):
        if random.random() < self.p:
            return F.vflip(img), F.vflip(target)
        return img, target


class RandomRotation(T.RandomRotation):

    def __call__(self, img, target):
        angle = self.get_params(self.degrees)
        img = F.rotate(img, angle, self.resample, self.expand, self.center)
        target = F.rotate(target, angle, Image.NEAREST, self.expand, self.center)
        return img, target


class RandomResizeCrop(T.RandomResizedCrop):

    def __call__(self, img, target):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        target = F.resized_crop(target, i, j, h, w, self.size, Image.NEAREST)
        return img, target


class Resize(T.Resize):

    def __call__(self, img, target):
        img = F.resize(img, self.size, interpolation=self.interpolation)
        target = F.resize(target, self.size, interpolation=Image.NEAREST)
        return img, target


class ColorJitter(T.ColorJitter):

    def __call__(self, img, target):
        return super(ColorJitter, self).__call__(img), target
