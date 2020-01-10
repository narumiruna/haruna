import json
import os
from collections.abc import Sequence

import numpy as np
import torch
from torch import distributed

import yaml


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


def isextension(f, ext):
    if not isinstance(ext, Sequence):
        ext = (ext,)

    return os.path.splitext(f)[1] in ext


def find_ext(top, ext):
    paths = []

    for root, _, files in os.walk(top):
        for f in files:
            if not isextension(f, ext):
                continue

            path = os.path.join(root, f)
            paths.append(path)

    return paths


def manual_seed(seed=0):
    """https://pytorch.org/docs/stable/notes/randomness.html"""
    torch.manual_seed(seed)
    np.random.seed(seed)
