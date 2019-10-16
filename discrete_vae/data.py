# Adapted from
# https://github.com/tensorflow/models/tree/master/research/rebar and
# https://github.com/duvenaud/relax/blob/master/datasets.py

import logging
import numpy as np
import os
import scipy.io
import urllib.request
import torch.utils.data

BINARIZED_MNIST_URL_PREFIX = \
    'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist'
BINARIZED_MNIST_DIR = \
    '/Users/tuananhle/Documents/research/datasets/binarized-mnist'
BINARIZED_MNIST_DIR_CC = \
    '/home/tuananh/projects/def-fwood/tuananh/datasets/binarized-mnist'


def download_binarized_mnist(dir=BINARIZED_MNIST_DIR,
                             url_prefix=BINARIZED_MNIST_URL_PREFIX,
                             splits=['train', 'valid', 'test']):
    """Downloads the binarized MNIST dataset and saves to .npy files.

    Args:
        dir: directory where to save dataset
        url_prefix: prefix of url where to download from
        splits: list of url suffixes; subset of train, valid, test
    """
    for split in splits:
        filename = 'binarized_mnist_{}.amat'.format(split)
        url = '{}/binarized_mnist_{}.amat'.format(url_prefix, split)
        path = os.path.join(dir, filename)
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)
            logging.info('Downloaded {} to {}'.format(url, path))

        npy_filename = 'binarized_mnist_{}.npy'.format(split)
        npy_path = os.path.join(dir, npy_filename)
        if not os.path.exists(npy_path):
            with open(path, 'rb') as f:
                np.save(npy_path,
                        np.array([list(map(int, line.split()))
                                  for line in f.readlines()], dtype='uint8'))
                logging.info('Saved to {}'.format(npy_path))


def load_binarized_mnist(where=None,
                         dir=BINARIZED_MNIST_DIR,
                         url_prefix=BINARIZED_MNIST_URL_PREFIX,
                         splits=['train', 'valid', 'test']):
    """Downloads the binarized MNIST dataset and saves to .npy files.

    Args:
        where: local or cc_cedar; overrides dir
        dir: directory where to save dataset
        url_prefix: prefix of url where to download from
        splits: list of url suffixes; subset of train, valid, test

    Returns: tuple of np.arrays each of shape [num_data, 784] correponding
        to the splits
    """

    if where is not None:
        if where == 'local':
            dir = BINARIZED_MNIST_DIR
        elif where == 'cc_cedar':
            dir = BINARIZED_MNIST_DIR_CC

    binarized_mnist = []
    for split in splits:
        npy_filename = 'binarized_mnist_{}.npy'.format(split)
        npy_path = os.path.join(dir, npy_filename)
        if not os.path.exists(npy_path):
            download_binarized_mnist(dir, url_prefix, [split])

        binarized_mnist.append(np.load(npy_path))
        logging.info('Loaded {}'.format(npy_path))

    return tuple(binarized_mnist)


def download_omniglot(path=OMNIGLOT_PATH, url=OMNIGLOT_URL):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
        logging.info('Downloaded {} to {}'.format(url, path))


def load_binarized_omniglot(path=OMNIGLOT_PATH, url=OMNIGLOT_URL):
    download_omniglot(path, url)
    n_validation = 1345

    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28 * 28),
                                                  order='fortran')

    omni_raw = scipy.io.loadmat(path)
    logging.info('Loaded {}'.format(path))

    train_data = reshape_data(omni_raw['data'].T.astype('float32'))
    test_data = reshape_data(omni_raw['testdata'].T.astype('float32'))

    # Binarize the data with a fixed seed
    np.random.seed(5)
    train_data = (np.random.rand(*train_data.shape) < train_data).astype(float)
    test_data = (np.random.rand(*test_data.shape) < test_data).astype(float)

    shuffle_seed = 123
    permutation = np.random.RandomState(seed=shuffle_seed).permutation(
        train_data.shape[0])
    train_data = train_data[permutation]

    x_train = train_data[:-n_validation]
    x_valid = train_data[-n_validation:]
    x_test = test_data

    return x_train, x_valid, x_test


def get_data_loader(np_array, batch_size, device):
    """Args:
        np_array: shape [num_data, data_dim]
        batch_size: int
        device: torch.device object

    Returns: torch.utils.data.DataLoader object
    """
    if device == torch.device('cpu'):
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {}
    return torch.utils.data.DataLoader(
        dataset=torch.tensor(np_array, dtype=torch.float, device=device),
        batch_size=batch_size,
        shuffle=True,
        **kwargs)
