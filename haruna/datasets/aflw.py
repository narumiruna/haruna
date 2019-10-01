import os
from glob import glob

import mlconfig
import scipy.io
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader
from torchvision.datasets.utils import download_and_extract_archive

from .utils import get_train_valid_split_sampler


def _load_target(mat_path):
    data = scipy.io.loadmat(mat_path)
    pt3d_68 = data['pt3d_68'].T
    landmarks = torch.from_numpy(pt3d_68[:, :2]).float()
    return landmarks


def _crop_by_landmarks(img, target, scale=1.1):
    min_x = target[:, 0].min().item()
    min_y = target[:, 1].min().item()
    max_x = target[:, 0].max().item()
    max_y = target[:, 1].max().item()

    w = max_x - min_x
    h = max_y - min_y

    c_x = (min_x + max_x) // 2
    c_y = (min_y + max_y) // 2
    size = int(max(w, h) * scale)

    box = (c_x - size // 2, c_y - size // 2, c_x + size // 2, c_y + size // 2)
    img = img.crop(box)

    target[:, 0] -= box[0]
    target[:, 1] -= box[1]

    return img, target


@mlconfig.register
class AFLW2000(Dataset):
    url = 'http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip'

    def __init__(self, root, transform=None, download=True):
        self.root = root
        self.transform = transform

        self.data_dir = os.path.join(self.root, 'AFLW2000')
        if download:
            self.download()

        self.samples = sorted(glob(os.path.join(self.data_dir, '*.mat')))

    def __getitem__(self, index):
        mat_path = self.samples[index]
        img_path = os.path.splitext(mat_path)[0] + '.jpg'

        img = pil_loader(img_path)
        target = _load_target(mat_path)

        img, target = _crop_by_landmarks(img, target)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.samples)

    def download(self):
        if not os.path.exists(self.data_dir):
            download_and_extract_archive(self.url, self.root)


@mlconfig.register
class AFLW2000Loader(DataLoader):

    def __init__(self, root, train=True, transform=None, valid_ratio=0.1, download=True, **kwargs):
        transform.train(mode=train)
        dataset = AFLW2000(root, transform=transform, download=download)
        sampler = get_train_valid_split_sampler(dataset, valid_ratio, train)
        super(AFLW2000Loader, self).__init__(dataset, sampler=sampler, **kwargs)
