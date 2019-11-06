import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_file_from_google_drive, extract_archive

LABEL_LIST = [
    'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair',
    'hat', 'ear_r', 'neck_l', 'neck', 'cloth'
]

COLOR_LIST = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
              [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
              [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

NUM_LABELS = 19


class CelebAMaskHQ(Dataset):

    def __init__(self, root, transform=None, download=False):
        self.root = root
        self.transform = transform

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
            f = os.path.join(self.root, self.FILENAME)
            if not os.path.exists(f):
                download_file_from_google_drive(file_id, self.root, filename)

            extract_archive(f)

    def __getitem__(self, index):
        img_path = self.samples[index]
        img = cv2.imread(img_path)

        target = self._load_target(img_path)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.samples)

    def _load_target(self, img_path):
        # https://github.com/switchablenorms/CelebAMask-HQ/tree/master/face_parsing
        img_index = int(Path(img_path).stem)
        folder_num = img_index // 2000

        target = np.zeros((512, 512, NUM_LABELS), dtype=np.uint8)
        for i, label in enumerate(LABEL_LIST, start=1):
            label_path = os.path.join(self.mask_dir, str(folder_num), '{:05d}_{}.png'.format(img_index, label))

            if os.path.exists(label_path):
                label = cv2.imread(label_path)
                target[:, :, i] = label[:, :, 0]

        target[:, :, 0] = 255 - target[:, :, 1:].max(axis=2)  # background
        target = target.argmax(axis=2).astype(np.uint8)
        return target
