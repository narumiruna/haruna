import os
from glob import glob
from pathlib import Path

import mlconfig
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms.functional import to_pil_image, to_tensor

from .utils import get_train_valid_split_sampler


def _load_target(path):
    pic = pil_loader(path)
    tensor = to_tensor(pic)
    tensor = tensor[[2, 0, 1], :, :]  # background first
    tensor = tensor.argmax(dim=0).int()
    return to_pil_image(tensor)


class LFW(Dataset):
    urls = [
        'http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz',
        'http://vis-www.cs.umass.edu/lfw/part_labels/parts_lfw_funneled_gt_images.tgz'
    ]

    image_dir = 'lfw_funneled'
    target_dir = 'parts_lfw_funneled_gt_images'

    def __init__(self, root, transform=None, download=False):
        self.root = root
        self.transform = transform

        if download:
            self.download()

        self.samples = self._prepare()

    def _prepare(self):
        samples = []

        target_paths = glob(os.path.join(self.root, self.target_dir, '*.ppm'))

        for target_path in target_paths:
            image_path = self._get_image_path(target_path)
            sample = (image_path, target_path)
            samples.append(sample)

        return samples

    def _get_image_path(self, target_path):
        name, index = Path(target_path).stem.rsplit('_', maxsplit=1)
        image_path = Path(self.root) / self.image_dir / name / '{}_{}.jpg'.format(name, index)
        return image_path

    def __getitem__(self, index):
        image_path, target_path = self.samples[index]
        image = pil_loader(image_path)
        target = _load_target(target_path)

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def __len__(self):
        return len(self.samples)

    def download(self):
        for url in self.urls:
            download_and_extract_archive(url,
                                         download_root=self.root,
                                         filename=Path(url).with_suffix('.tar').name,
                                         extract_root=self.root)


@mlconfig.register
class LFWLoader(DataLoader):

    def __init__(self, root, train=True, transform=None, download=False, valid_ratio=0.1, **kwargs):
        transform.train(mode=train)
        dataset = LFW(root, transform=transform, download=download)
        sampler = get_train_valid_split_sampler(dataset, valid_ratio, train)
        super(LFWLoader, self).__init__(dataset, sampler=sampler, **kwargs)
