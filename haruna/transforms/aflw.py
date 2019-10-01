import mlconfig
import torchvision.transforms as T
import torchvision.transforms.functional as F


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class ToPercentCoords(object):

    def __call__(self, img, target):
        w, h = img.size

        target[:, 0] /= w
        target[:, 1] /= h

        return img, target


class FlattenTarget(object):

    def __call__(self, img, target):
        return img, target.flatten()


class ToTensor(object):

    def __call__(self, img, target):

        return F.to_tensor(img), target


class Resize(T.Resize):

    def __call__(self, img, target):
        return F.resize(img, self.size, self.interpolation), target


@mlconfig.register
class AFLW2000Transform(object):

    def __init__(self, size=224):
        self.training = True

        self.train_trasform = Compose([
            ToPercentCoords(),
            Resize(size),
            ToTensor(),
            FlattenTarget(),
        ])

        self.eval_transform = self.train_trasform

    def __call__(self, img, target):
        if self.training:
            return self.train_trasform(img, target)
        return self.eval_transform(img, target)

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        return self.train(False)
