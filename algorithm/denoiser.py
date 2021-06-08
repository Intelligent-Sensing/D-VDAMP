"""Collection of denoiser wrapper functions for easy use.

    Denoisers
        * BM3D_denoiser
        * DnCNN_denoiser
        * DnCNN_ensemble_denoiser

    Denoising CNN setup
        * setup_DnCNN
        * setup_DnCNN_ensemble

    Model definition
        * DnCNN
        * ColoredDnCNN

Notes:
    It is the client's resposibility to call these functions under
    torch.no_grad() environment where appropriate.
"""

import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from bm3d import bm3d
from util import general as gutil

class BM3D_denoiser:
    """BM3D denoiser.

    This is a wrapper for the bm3d function from the bm3d package from
    https://pypi.org/project/bm3d/
    """
    def __init__(self, std=None):
        """Initialize BM3D denoiser.

        Args:
            std (float): the default standard deviation (std) as input to BM3D denoiser.
            This value is used when std is not given when the denoiser is called.
        """
        self.std = std

    def __call__(self, image, std=None):
        """Denoise image with BM3D.

        Args:
            image (array/tensor): image with shape (C, H, W)
            std (float): std as input to BM3D denoiser. If None, use default std value instead.

        Returns:
            denoised_image (tensor): denoised image with shape (C, H, W)
        """
        if std is None:
            std = self.std
        return torch.Tensor(bm3d(image[0], std)).unsqueeze(0)

class DnCNN_denoiser:
    """DnCNN denoiser

    Note:
        The DnCNN outputs the predicted noise. Hence, the denoised image is the image subtracted
        by the DnCNN output.
    """
    def __init__(self, model, device=torch.device('cpu')):
        """Initialize DnCNN denoiser

        Args:
            model: the DnCNN denoiser model. See setup_DnCNN function for loading the model.
            batch_size (int): the batch size in case this denoiser is called on tensor of multiple images.
            device: the device to run the model on e.g. torch.device('cpu'), torch.device('cuda'), etc.
        """
        self.model = model
        self.device = device

    def __call__(self, image, std=None):
        """Denoise images with DnCNN.

        Args:
            image (tensor): the noisy image tensor. The shape can be (C, H, W) for single image and
                (N, C, H, W) for multiple images.
            std: dummy argument so that this denoiser is compatible with
                algorithm.heatmap.calc_sure.

        Returns:
            output (tensor): denoised image(s).
        """
        image = image.to(device=self.device, dtype=torch.float32)
        output = self.model(image)
        return output.cpu()

class DnCNN_ensemble_denoiser:
    """Ensemble of DnCNN denoisers for multiple noise levels.

    This is a wrapper for DnCNN denoisers, each one is trained for a specific noise level.
    """
    def __init__(self, models, std_ranges, device=torch.device('cpu'), verbose=False, std=None):
        """Initialize DnCNN ensemble denoiser.

        Args:
            models (list): list of DnCNN models.
            std_ranges (np.ndarray): array of ranges of noise std corresponding to the model
                in the increasing order. For example, array([0, 5, 10]) means the ranges are
                [0, 5] and [5, 10].
            device: the device to run the model on.
            verbose (bool): whether to print which DnCNN is selected.
            std (float): the default std of the noise image.
        """
        self.models = models
        self.std_ranges = std_ranges
        self.device = device
        self.verbose = verbose
        self.std = std

    def __call__(self, image, std=None):
        """Denoise images with DnCNN ensemble.

        Args:
            image (tensor): single noise image with shape (C, H, W).
            std (float): std of the noise for selecting DnCNN trained on
                this noise level. If None, use the default value.

        Returns:
            output (tensor): denoised image.
        """
        if std is None:
            std = self.std

        select = np.sum(std > self.std_ranges) - 1
        if select < 0:
            if self.verbose:
                print('denoiser.DnCNN_ensemble_denoiser: The noise level is lower than models available')
            select += 1
        elif select > len(self.models) - 1:
            if self.verbose:
                print('denoiser.DnCNN_ensemble_denoiser: The noise level is higher than models available')
            select -= 1
        if self.verbose:
            print('denoiser.DnCNN_ensemble_denoiser: select = {:d}'.format(select))
        image = image.to(device=self.device, dtype=torch.float32)
        output = self.models[select](image)
        return output.cpu()

def setup_DnCNN(modedir, num_layers=17, device=torch.device('cpu')):
    """Load a DnCNN model

    Args:
        modeldir (str): path to model.
        num_layers (int): number of layers of the DnCNN model.
        device: device to run DnCNN on.
    """
    model = DnCNN(1, num_layers=num_layers)
    gutil.load_checkpoint(modedir, model, None, device=device)
    model.to(device=device)
    model.eval()
    return model

def setup_DnCNN_ensemble(path, modelnames, num_layers=20, device=torch.device('cpu')):
    """Set up a DnCNN ensemble from saved DnCNN models.

    Args:
        path (str): path to directory containing DnCNN models.
        modelnames (list): list of saved model file names.
        num_layers (int): number of layers of the DnCNN models.
        device: device to run DnCNN on.
    """
    models = [None] * len(modelnames)
    for i, name in enumerate(modelnames):
        models[i] = setup_DnCNN(os.path.join(path, '{}.pth'.format(name)),
                                num_layers=num_layers, device=device)
    return models

class DnCNN(nn.Module):
    """DnCNN model.

    This is an implementation of the DnCNN denoiser described in

        Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, and Lei Zhang,
        “Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising,”
        IEEE Transactions on Image Processing, vol. 26, no. 7, pp. 3142–3155, 2017.
    """
    def __init__(self, channels, num_layers=17):
        super(DnCNN, self).__init__()

        # Fixed parameters
        kernel_size = 3
        padding = 1
        features = 64

        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features,
                                kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features,
                                    kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels,
                                kernel_size=kernel_size, padding=padding, bias=False))
        self.layers = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        """Model forward function.

        Args:
            x (tensor): image of shape (C, H, W)

        Note:
            The expected image shape is different from the model when training which
            expects (N, C, H, W) where N is the batch size.
        """
        noise = self.layers(x.expand(1, -1, -1, -1)).squeeze(dim=0)
        out = x - noise
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

class ColoredDnCNN(nn.Module):
    """CNN model for removing colored noise.

    This is an implementation of the denoiser proposed in

        C. A. Metzler and G. Wetzstein, "D-VDAMP: Denoising-Based Approximate Message Passing
        for Compressive MRI," ICASSP 2021 - 2021 IEEE International Conference on Acoustics, 
        Speech and Signal Processing (ICASSP), 2021, pp. 1410-1414, doi: 10.1109/ICASSP39728.2021.9414708.

    This denoiser is designed for denoising images where each wavelet subband contains additive white
    Gaussian noise of a known standard deviation.

    Note:
        Since this model is used only in D-VDAMP, refer to algorithm.vdamp for denoiser wrappers of this model.
    """
    def __init__(self, channels=1, num_layers=20, std_channels=13):
        super(ColoredDnCNN, self).__init__()

        self.num_layers = num_layers

        # Fixed parameters
        kernel_size = 3
        padding = 1
        features = 64

        conv_layers = []
        bn_layers = []

        self.first_conv = nn.Conv2d(in_channels=channels+std_channels, out_channels=features,
                                kernel_size=kernel_size, padding=padding, bias=False)
        for _ in range(num_layers - 2):
            conv_layers.append(nn.Conv2d(in_channels=features+std_channels, out_channels=features,
                                    kernel_size=kernel_size, padding=padding, bias=False))
            bn_layers.append(nn.BatchNorm2d(features))
        self.last_conv = nn.Conv2d(in_channels=features+std_channels, out_channels=channels,
                                kernel_size=kernel_size, padding=padding, bias=False)

        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        self._initialize_weights()

    def forward(self, x, std):
        """Model forward function.

        Args:
            x (tensor): image of shape (C, H, W)
            std (tensor): standard deviation of noise in each wavelet subband. Expect (num_subbands,) shape.

        Note:
            The expected input shapes are different from the model when training which
            expects (N, C, H, W) and (N, num_subbands) where N is the batch size.
        """
        _, H, W = x.shape
        x = x.unsqueeze(0)
        std = std.unsqueeze(0)
        std_channels = self._generate_std_channels(std, H, W)
        noise = torch.cat((x, std_channels), dim=1)
        noise = F.relu(self.first_conv(noise))
        for i in range(self.num_layers - 2):
            noise = torch.cat((noise, std_channels), dim=1)
            noise = F.relu(self.bn_layers[i](self.conv_layers[i](noise)))
        noise = torch.cat((noise, std_channels), dim=1)
        noise = self.last_conv(noise)
        out = (x - noise).squeeze(0)
        return out

    def _generate_std_channels(self, std, H, W):
        concat_channels = std.shape[1]
        std_channels = std.reshape(1, concat_channels, 1, 1).repeat(1, 1, H, W)
        return std_channels

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)