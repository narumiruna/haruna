import mlconfig

from .segmentation import Compose, Normalize, RandomRotation, RandomVerticalFlip, ToTensor


@mlconfig.register
class LFWTransform(object):

    def __init__(self, degrees=15):
        self.training = True

        normalize = Normalize(mean=(0.433922, 0.374727, 0.332006), std=(0.297252, 0.272523, 0.266515))

        self.train_trasform = Compose([
            RandomRotation(degrees),
            RandomVerticalFlip(),
            ToTensor(),
            normalize,
        ])

        self.eval_transform = Compose([
            ToTensor(),
            normalize,
        ])

    def __call__(self, img, target):
        if self.training:
            return self.train_trasform(img, target)
        return self.eval_transform(img, target)

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        return self.train(False)
