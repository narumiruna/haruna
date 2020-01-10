import os
from glob import glob


import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.utils import (download_file_from_google_drive, extract_archive)
from torchvision.utils import save_image

from ..transforms import segmentation as TS
from .utils import get_train_valid_split_sampler

LABEL_LIST = [
    'cloth', 'neck', 'neck_l', 'ear_r', 'hat', 'hair', 'l_lip', 'u_lip', 'mouth', 'r_ear', 'l_ear', 'r_brow', 'l_brow',
    'r_eye', 'l_eye', 'eye_g', 'nose', 'skin'
]

COLOR_LIST = [[0, 0, 0], [0, 204, 0], [255, 153, 51], [0, 51, 0], [0, 204, 204], [255, 51, 153], [0, 0, 204],
              [0, 0, 153], [255, 255, 0], [102, 204, 0], [255, 0, 0], [102, 51, 0], [255, 204, 204], [0, 255, 255],
              [204, 0, 204], [51, 51, 255], [204, 204, 0], [76, 153, 0], [204, 0, 0]]


def load_image(f, mode='RGB'):
    with open(f, 'rb') as fp:
        img = Image.open(fp)
        return img.convert(mode)


class CelebAMaskHQ(Dataset):

    def __init__(self, root, transform=None, download=False, label_list=None):
        self.root = root
        self.transform = transform
        self.label_list = label_list or LABEL_LIST
        self.num_classes = len(self.label_list) + 1

        self.dataset_dir = os.path.join(self.root, 'CelebAMask-HQ')
        self.image_dir = os.path.join(self.dataset_dir, 'CelebA-HQ-img')
        self.mask_dir = os.path.join(self.dataset_dir, 'CelebAMask-HQ-mask-anno')

        if download:
            self.download()

        self.samples = glob(os.path.join(self.image_dir, '*.jpg'))

    def download(self):
        file_id = '1badu11NqxGf6qM3PTTooQDJvQbejgbTv'
        filename = 'CelebAMask-HQ.zip'

        if not os.path.exists(self.dataset_dir):
            f = os.path.join(self.root, filename)
            if not os.path.exists(f):
                download_file_from_google_drive(file_id, self.root, filename)

            extract_archive(f)

    def __getitem__(self, index):
        img_path = self.samples[index]
        img = load_image(img_path)

        target = self._load_target(img_path)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.samples)

    def _load_target(self, img_path):
        r"""Load target from mask annotations
        Note that the results of torch.argmax and numpy.argmax are different:
            torch.tensor([0, 1, 1]).argmax()
            >>> tensor(2)
            np.array([0, 1, 1]).argmax()
            >>> 1
        https://github.com/switchablenorms/CelebAMask-HQ/tree/master/face_parsing
        """
        img_index = int(os.path.splitext(os.path.basename(img_path))[0])
        folder_num = img_index // 2000

        target = np.zeros((self.num_classes, 512, 512), dtype=np.uint8)
        for i, label in enumerate(self.label_list, start=1):
            label_path = os.path.join(self.mask_dir, str(folder_num), '{:05d}_{}.png'.format(img_index, label))

            if os.path.exists(label_path):
                target[i, :, :] = np.array(load_image(label_path, mode='L'), dtype=np.uint8)

        target[0, :, :] = 255 - target[1:, :, :].max(axis=0)  # background

        target = target.argmax(axis=0).astype(np.uint8)
        return Image.fromarray(target)


class CelebAMaskHQTransform(object):

    def __init__(self, training=True, size=256, degrees=15):
        self.training = training

        self.augmentation = TS.Compose([
            TS.RandomVerticalFlip(),
            TS.RandomHorizontalFlip(),
            TS.RandomRotation(degrees),
        ])

        self.transform = TS.Compose([
            TS.Resize(size),
            TS.ToTensor(),
            TS.Normalize(mean=(0.5170, 0.4166, 0.3633), std=(0.1324, 0.1234, 0.1226)),
        ])

    def __call__(self, img, target):
        if self.training:
            img, target = self.augmentation(img, target)
        return self.transform(img, target)



class CelebAMaskHQLoader(DataLoader):

    def __init__(self, root, train=True, download=False, label_list=None, image_size=64, valid_ratio=0.1, **kwargs):
        transform = CelebAMaskHQTransform(train, size=image_size)
        dataset = CelebAMaskHQ(root, transform=transform, download=download, label_list=label_list)

        sampler = None
        if valid_ratio > 0.0:
            sampler = get_train_valid_split_sampler(dataset, valid_ratio, train)

        super(CelebAMaskHQLoader, self).__init__(dataset, sampler=sampler, shuffle=(sampler is None), **kwargs)


def save_target_image(target, filename, color_list=None, **kwargs):
    """Draw batch mask annotation and save it

    Example:
        from torchface.datasets.celebamask import CelebAMaskHQLoader, save_target_image

        dataloader = CelebAMaskHQLoader('data', batch_size=4, image_size=512)
        _, y = next(iter(dataloader))
        save_target_image(y, 'img.jpg')

    Arguments:
        target (Tensor): tensor with size (batch_size, height, width) to visualize
        color_list (list): a list of colors
    """
    color_list = color_list or COLOR_LIST

    def _colorize(target: torch.Tensor):
        height, width = target.shape
        tensor = torch.zeros(3, height, width)

        for color_index, color in enumerate(COLOR_LIST):
            color_tensor = torch.tensor(color, dtype=torch.float).unsqueeze(1)
            tensor[:, target.eq(color_index)] = color_tensor
        return tensor

    tensors = []
    for i in range(target.size(0)):
        tensors.append(_colorize(target[i]))

    img = torch.stack(tensors, dim=0).div(255.0)
    save_image(img, filename, **kwargs)
