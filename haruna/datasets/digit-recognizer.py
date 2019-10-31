import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image


class DigitRecognizer(Dataset):

    def __init__(self, root='data/digit-recognizer', train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        if train:
            self.df = pd.read_csv(os.path.join(root, 'train.csv'))
        else:
            self.df = pd.read_csv(os.path.join(root, 'test.csv'))

    def __getitem__(self, index):
        tensor = torch.tensor(self.df.iloc[index][1:], dtype=torch.uint8).view(28, 28)
        img = to_pil_image(tensor)
        label = int(self.df.iloc[index][0])

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return self.df.shape[0]


class DigitRecognizerLoader(DataLoader):

    def __init__(self, root, train, size=32, shuffle=True, **kwargs):
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])
        dataset = DigitRecognizer(root, train, transform=transform)
        super(DigitRecognizerLoader, self).__init__(dataset, shuffle=shuffle, **kwargs)


def main():
    from torchvision.utils import save_image

    loader = DigitRecognizerLoader(root='data/digit-recognizer', train=True, batch_size=32)

    x, y = next(iter(loader))

    print(x.size())
    print(y.size())
    save_image(x, 'test.jpg')


if __name__ == '__main__':
    main()
