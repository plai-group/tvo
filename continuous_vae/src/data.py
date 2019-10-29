# Adapted from
# https://github.com/tensorflow/models/tree/master/research/rebar and
# https://github.com/duvenaud/relax/blob/master/datasets.py

import logging
import numpy as np
import os
import scipy.io
import urllib.request
import torch.utils.data
from torch.utils.data import Dataset


def get_data_loader(np_array, batch_size, args):
    """Args:
        np_array: shape [num_data, data_dim]
        batch_size: int
        device: torch.device object

    Returns: torch.utils.data.DataLoader object
    """

    if args.device == torch.device('cpu'):
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {}

    dataset = torch.tensor(np_array, dtype=torch.float, device=args.device)

    dataset = STOCHASTIC_MNIST(dataset)

    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, **kwargs)


class STOCHASTIC_MNIST(Dataset):
    def __init__(self, image):
        super(STOCHASTIC_MNIST).__init__()
        self.image = image

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, idx):
        return torch.bernoulli(self.image[idx, :])
