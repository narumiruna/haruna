import random

from torch.utils.data import SubsetRandomSampler


def get_train_valid_split_sampler(dataset, valid_ratio, train):
    num_samples = len(dataset)
    num_valid_samples = int(num_samples * valid_ratio)
    indices = list(range(num_samples))

    random.seed(0)
    random.shuffle(indices)

    train_indices = indices[:-num_valid_samples]
    valid_indices = indices[-num_valid_samples:]
    assert len(train_indices) == num_samples - num_valid_samples
    assert len(valid_indices) == num_valid_samples

    if train:
        sampler = SubsetRandomSampler(train_indices)
    else:
        sampler = SubsetRandomSampler(valid_indices)

    return sampler
