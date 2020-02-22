import json
import os
import random
from collections.abc import Sequence

import numpy as np
import torch
import yaml
from torch import distributed


def load_yaml(f):
    with open(f, 'r') as fp:
        return yaml.safe_load(fp)


def save_yaml(data, f, **kwargs):
    with open(f, 'w') as fp:
        yaml.safe_dump(data, fp, **kwargs)


def load_json(f):
    with open(f, 'r') as fp:
        return json.load(fp)


def save_json(data, f, **kwargs):
    with open(f, 'w') as fp:
        json.dump(data, fp, **kwargs)


def distributed_is_initialized():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False


def to_sequence(x):
    if not isinstance(x, Sequence):
        x = (x,)
    return x


def is_extension(f, ext):
    ext = to_sequence(ext)
    # os.path.splitext(f)[1] is faster than Path(f).suffix
    return os.path.splitext(f)[1] in ext


def list_all(top):
    for root, _, files in os.walk(top):
        for f in files:
            yield os.path.join(root, f)


def list_all_ext(top, ext):
    for f in list_all(top):
        if is_extension(f, ext):
            yield f


def manual_seed(seed=0):
    """https://pytorch.org/docs/stable/notes/randomness.html"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
