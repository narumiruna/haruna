import os

import torch
import torchfile
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader
from torchvision.datasets.utils import download_file_from_google_drive, extract_archive
from tqdm import tqdm

from ..transforms.aflw import ToTensor
from ..utils import find_ext
from .utils import get_train_valid_split_sampler


class LS3DW(Dataset):
    r"""https://www.adrianbulat.com/face-alignment"""

    file_id = '1ImWSqTon63QgqdsWiRudCxkYypkfQl1U'
    filename = 'LS3D-W.tar.gz'

    def __init__(self, root, transform=None, download=True):
        self.root = root
        self.transform = transform

        if download:
            self.download()

        self.samples = self.prepare()

    def __getitem__(self, index):
        name, ext = self.samples[index]
        img = pil_loader(name + ext)
        assert img.size == (780, 580), name  # TODO: crop image
        target = torch.from_numpy(torchfile.load(name + '.t7'))

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.samples)

    def prepare(self):
        samples = []

        paths = find_ext(os.path.join(self.root, 'LS3D-W'), ext='.t7')
        for path in tqdm(paths):
            name = os.path.splitext(path)[0]

            if os.path.exists(name + '.jpg'):
                sample = (name, '.jpg')
            elif os.path.exists(name + '.png'):
                sample = (name, '.png')
            else:
                raise FileNotFoundError

            samples.append(sample)
        return samples

    def download(self):
        if not os.path.exists(os.path.join(self.root, 'LS3D-W')):
            download_file_from_google_drive(self.file_id, self.root, filename=self.filename)
            extract_archive(os.path.join(self.root, self.filename))


class LS3DWLoader(DataLoader):

    def __init__(self, root, train=True, download=True, valid_ratio=0.05, **kwargs):
        transform = ToTensor()
        dataset = LS3DW(root, transform=transform, download=download)
        sampler = get_train_valid_split_sampler(dataset, valid_ratio, train)
        super(LS3DWLoader, self).__init__(dataset, sampler=sampler, **kwargs)
