"""Utility function for train/test/val CNN denoisers"""

import numpy as np
import torch
from torchvision.transforms.functional import rotate
from util.general import calc_psnr, load_checkpoint

dtype = torch.float32

@torch.no_grad()
def batch_psnr(test_image, target_image, max=1.):
    """Calculate average PSNR of a batch of denoised image
    Note:
        The first dimension of the batches must be N (batch size).
    Args:
        test_image (torch.Tensor): batch to calculate PSNR.
        target_image (torch.Tensor): groud truth batch.
        max (float): maximum pixel value on the scale e.g. 1. from [0., 1.].
    Returns:
        psnr (float): average PSNR value.
    """
    psnr = 0
    num_images = test_image.shape[0]
    for i in range(num_images):
        psnr += calc_psnr(test_image[i], target_image[i], max=max)
    psnr /= num_images
    return psnr

def load_checkpoint_train(cpdir, model, optimizer):
    """Load model and optimizer parameters for training
    Note:
        This is simply a wrapper to load_checkpoint so that
        global_step and epoch are updated correctly.
        If cpdir is None, do not load checkpoint and returns
        0 for global_step and epoch.
    Args:
        cpdir (str): path to the checkpoint.
        model: the model to load the parameters to.
        optimizer: the optimizer to load parameters to.
    Returns:
        start_global_step (int): the global step from the checkpoint.
        start_epoch (int): the epoch from the checkpoint.
    """
    start_epoch = 0
    start_global_step = 0
    if cpdir is not None:
        start_global_step, start_epoch = load_checkpoint(
            cpdir, model, optimizer)
        start_global_step += 1
        start_epoch += 1
    return start_global_step, start_epoch

class FixedAngleRotation:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = float(np.random.choice(self.angles))
        return rotate(x, angle)