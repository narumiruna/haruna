import os
from glob import glob

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader
from torchvision.datasets.utils import download_file_from_google_drive, extract_archive
from torchvision.transforms.functional import to_pil_image

from .utils import get_train_valid_split_sampler


class Severstal(Dataset):
    files = [('1xtzycQnyGTvSJ0SQfCfUw9AyqinTdF3E', 'train_images.zip'),
             ('1TudKIxyZ-bEzO1lMwU6XzVXL4wIaH_4O', 'test_images.zip'),
             ('1rAdo2k2b4Xp9P2KyhihDWkR75Iwg7nNT', 'train.csv.zip'),
             ('1jvO3srUzjLuDCJ-deNqICCQRpcR4XYWB', 'train_targets.zip')]

    def __init__(self, root, transform=None, download=False):
        self.root = root
        self.transform = transform

        if download:
            self.download()

        self.samples = self.prepare()

    def prepare(self):
        samples = []

        paths = sorted(glob(os.path.join(self.root, 'train_targets', '*.npy')))
        for path in paths:
            sample = os.path.basename(path).replace('.npy', '')
            samples.append(sample)

        return samples

    def download(self):
        for file_id, filename in self.files:
            download_file_from_google_drive(file_id, self.root, filename)

            path = os.path.join(self.root, os.path.splitext(filename)[0])
            if not os.path.exists(path):
                extract_archive(os.path.join(self.root, filename), to_path=path)

    def __getitem__(self, index):
        sample = self.samples[index]

        img_path = os.path.join(self.root, 'train_images', '{}.jpg'.format(sample))
        img = pil_loader(img_path)

        target_path = os.path.join(self.root, 'train_targets', '{}.npy'.format(sample))
        target = to_pil_image(np.load(target_path).astype(np.int32))

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.samples)


class SeverstalLoader(DataLoader):

    def __init__(self, root, train=True, transform=None, valid_ratio=0.1, download=False, **kwargs):
        transform.train(mode=train)
        dataset = Severstal(root, transform=transform, download=download)
        sampler = get_train_valid_split_sampler(dataset, valid_ratio, train)
        super(SeverstalLoader, self).__init__(dataset, sampler=sampler, **kwargs)
