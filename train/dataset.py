"""Implement the Dataset class for CNN denoiser train/eval/test.
Extends torch.utils.data.Dataset class for training and testing.

Example:
    To create a torch.utils.data.Dataloader
    from a DenoiseDataset dataset, use::
        data_loader = DataLoader(dataset).
Note:
    A .h5 file storing the dataset must be formatted as
        dataset.h5
            - key 1 : image 1
            - key 2 : image 2
            - ...
        where each image has dimensions (C, H, W).
"""

import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import torch
from torch.utils.data import Dataset
import h5py
from util import transform

class DenoiseDataset(Dataset):
    def __init__(self, datadir, std_range, wavetype='haar', level=4, transforms=None):
        """
        Args:
            datadir (str): path to .h5 file.
            transform: image transformation, for data augmentation.
            
        Note:
            If sigma or alpha are lists of two entries, the value used will be
            uniformly sampled from the two values.
        """
        super(DenoiseDataset, self).__init__()
        self.datadir = datadir
        self.std_range = std_range
        self.wavetype = wavetype
        self.level = level
        self.num_stds = 3 * level + 1
        self.transforms = transforms
        h5f = h5py.File(datadir, 'r')
        self.keys = list(h5f.keys())
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        """Return image, noise pair of the same size (C, H, W)."""
        # Load image
        h5f = h5py.File(self.datadir, 'r')
        key = self.keys[idx]
        image = torch.Tensor(h5f[key])
        h5f.close()
        if self.transforms is not None:
            image = self.transforms(image)

        noisy_image, stds = self._generate_noisy_image(image)

        return image, noisy_image, stds

    def _generate_noisy_image(self, image):
        wavelet = transform.forward(image, wavelet=self.wavetype, level=self.level)
        stds = torch.FloatTensor(self.num_stds).uniform_(self.std_range[0], self.std_range[1])
        noisy_wavelet = transform.add_noise_subbandwise(wavelet, stds, is_complex=False)
        noisy_image = noisy_wavelet.inverse(wavelet=self.wavetype)
        return noisy_image, stds