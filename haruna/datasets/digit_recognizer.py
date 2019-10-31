import os

import mlconfig
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from .utils import get_train_valid_split_sampler


class DigitRecognizer(Dataset):

    def __init__(self, root='data/digit-recognizer', transform=None):
        self.root = root
        self.transform = transform
        self.df = pd.read_csv(os.path.join(root, 'train.csv'))

    def __getitem__(self, index):
        tensor = torch.tensor(self.df.iloc[index][1:], dtype=torch.uint8).view(28, 28)
        img = to_pil_image(tensor)
        label = int(self.df.iloc[index][0])

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return self.df.shape[0]


@mlconfig.register
class DigitRecognizerLoader(DataLoader):

    def __init__(self, root, train=True, size=32, batch_size=32, valid_ratio=0.1, **kwargs):
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])
        dataset = DigitRecognizer(root, transform=transform)
        sampler = get_train_valid_split_sampler(dataset, valid_ratio, train)
        super(DigitRecognizerLoader, self).__init__(dataset,
                                                    batch_size=batch_size,
                                                    sampler=sampler,
                                                    shuffle=False,
                                                    **kwargs)
