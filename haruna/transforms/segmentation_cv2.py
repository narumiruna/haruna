import random

import cv2
import torchvision.transforms as T


class Compose(T.Compose):

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):

    def __call__(self, img, target):
        if random.random() < self.p:
            return cv2.flip(img, 1), cv2.flip(target, 1)
        return img, target


class RandomVerticalFlip(T.RandomVerticalFlip):

    def __call__(self, img, target):
        if random.random() < self.p:
            return cv2.flip(img, 0), cv2.flip(target, 0)
        return img, target
