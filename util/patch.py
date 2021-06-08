"""Utility functions for patch processing

    * Compose (from torchvision.transforms)
    * calc_num_patches
    * ImageToPatches
    * PatchesToImage
    * VecToPatches
"""

import torch
from torchvision.transforms import Compose

def calc_num_patches(height, width, window, stride=None, ret_hw=False):
    """Calculate number of patches given image dimension and window size"""
    if stride is None:
        stride = window
    numPatchesH = (height - window) // stride + 1
    numPatchesW = (width - window) // stride + 1
    if ret_hw:
        return numPatchesH, numPatchesW
    else:
        return numPatchesH * numPatchesW

class ImageToPatches:
    """Extract patches from an image.

    Note:
        Call an instantiation of this class, specified by window and stide,
        on an image with shape (C, H, W) to extract patches.
    """
    def __init__(self, window, stride=None):
        self.window = window
        if stride is None:
            self.stride = window
        else:
            self.stride = stride

    def __call__(self, image):
        C, H, W = image.shape
        numPatchesH = (H - self.window) // self.stride + 1
        numPatchesW = (W - self.window) // self.stride + 1
        numPatches = numPatchesH * numPatchesW
        patches = torch.zeros(numPatches, C, self.window, self.window)
        idx = 0
        for kh in range(numPatchesH):
            for kw in range(numPatchesW):
                patches[idx] = image[:,
                                     kh * self.stride : kh * self.stride + self.window,
                                     kw * self.stride : kw * self.stride + self.window]
                idx += 1
        return patches

class PatchesToImage:
    """Reconstruct image from patches.

    Note:
        Call an instantiation of this class on patches with shape (N, C, H, W) to
        reconstruct image.

        The stride is the stride specified when extracting patches.
        If the stride is None, assume that patches are non-overlapping.
    """
    def __init__(self, height, width, stride=None):
        self.height = height
        self.width = width
        self.stride = stride

    def __call__(self, patches):
        _, c, h, w = patches.shape
        if self.stride is None:
            numPatchesH = self.height // h
            numPatchesW = self.width // w
            image = torch.zeros(c, self.height, self.width)
            idx = 0
            for kh in range(numPatchesH):
                for kw in range(numPatchesW):
                    image[:, kh * h : (kh + 1) * h,
                          kw * w : (kw + 1) * w] = patches[idx]
                    idx += 1
        else:
            numPatchesH = (self.height - h) // self.stride + 1
            numPatchesW = (self.width - w) // self.stride + 1
            one_patches = torch.ones(*patches.shape)
            overlap_factor = torch.zeros(c, self.height, self.width)
            image = torch.zeros(c, self.height, self.width)
            idx = 0
            for kh in range(numPatchesH):
                for kw in range(numPatchesW):
                    image[:, kh * self.stride : kh * self.stride + h,
                          kw * self.stride : kw * self.stride + w] += patches[idx]
                    overlap_factor[:, kh * self.stride : kh * self.stride + h,
                          kw * self.stride : kw * self.stride + w] += one_patches[idx]
                    idx += 1
            image /= overlap_factor
        return image

class VecToPatches:
    """Generate patches, each having the same entry values as the input Tensor

    Note:
        For input Tensor (N), return patches (N, C, H, W) where values in each patch
        are identical and the same as values in the corresponding index of input Tensor.

    """
    def __init__(self, window, channels=1):
        self.window = window
        self.channels = channels

    def __call__(self, vec):
        vec = vec.unsqueeze(1).expand(-1, self.channels)
        vec = vec.unsqueeze(2).expand(-1, -1, self.window)
        vec = vec.unsqueeze(3).expand(-1, -1, -1, self.window)
        return vec
