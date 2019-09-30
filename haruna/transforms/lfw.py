import random

import mlconfig
import torchvision.transforms as torchtrans
import torchvision.transforms.functional as torchfunc


class Compose(torchtrans.Compose):

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(torchtrans.ToTensor):

    def __call__(self, image, target):
        return torchfunc.to_tensor(image), torchfunc.to_tensor(target).argmax(dim=0)


class Normalize(torchtrans.Normalize):

    def __call__(self, tensor, target):
        return torchfunc.normalize(tensor, self.mean, self.std, self.inplace), target


class RandomHorizontalFlip(torchtrans.RandomHorizontalFlip):

    def __call__(self, img, target):
        if random.random() < self.p:
            return torchfunc.hflip(img), torchfunc.hflip(target)
        return img, target


class RandomVerticalFlip(torchtrans.RandomVerticalFlip):

    def __call__(self, img, target):
        if random.random() < self.p:
            return torchfunc.vflip(img), torchfunc.vflip(target)
        return img, target


class RandomRotation(torchtrans.RandomRotation):

    def __call__(self, img, target):
        angle = self.get_params(self.degrees)

        return (
            torchfunc.rotate(img, angle, self.resample, self.expand, self.center),
            torchfunc.rotate(target, angle, self.resample, self.expand, self.center),
        )


class RandomResizeCrop(torchtrans.RandomResizedCrop):

    def __call__(self, img, target):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return (
            torchfunc.resized_crop(img, i, j, h, w, self.size, self.interpolation),
            torchfunc.resized_crop(target, i, j, h, w, self.size, self.interpolation),
        )


@mlconfig.register
class LFWTransform(object):

    def __init__(self, degrees=15, mean=0.0, std=1.0):
        self.training = True

        self.train_trasform = Compose([
            RandomRotation(degrees),
            RandomVerticalFlip(),
            ToTensor(),
            Normalize(mean, std),
        ])

        self.eval_transform = Compose([
            ToTensor(),
            Normalize(mean, std),
        ])

    def __call__(self, img, target):
        if self.training:
            return self.train_trasform(img, target)
        return self.eval_transform(img, target)

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        return self.train(False)
