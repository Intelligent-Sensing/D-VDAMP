"""Compressive sensing reconstruction algorithms.

    * DIT
    * DAMP

Notes:
    It is the client's resposibility to call these functions under
    torch.no_grad() environment where appropriate.
"""

import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
import torch
from util.general import calc_psnr, generate_noise

class IterativeDenoisingCSRecon:
    def __init__(self,
                 image_shape,
                 denoiser,
                 iters,
                 image,
                 verbose):
        """Initialize an iterative denoising CS solver.

        Args:
            image_shape (list/array): shape of the ground truth image in the format (C, H, W).
            denoiser: the denoiser for thresholding.
            iters: number of iterations.
            image: the ground truth image. If image is given, calculate the PSNR of the 
                thresholding result at every iteration.
            verbose (bool): whether to print standard deviation of the effective noise 
                and/or PSNR at every iteration.
        """
        self.H = image_shape[1]
        self.W = image_shape[2]
        self.denoiser = denoiser
        self.iters = iters
        self.image = image
        self.verbose = verbose

    def __call__(self, y, Afun, Atfun, m, n):
        """Solve the CS reconstruction problem.

        Args:
            y (tensor): the measurement with dimension m.
            Afun: the forward CS measurement operator.
            Atfun: the transpose of Afun.
            m (int): dimension of the measurement.
            n (int): dimension of the ground truth.

        Returns:
            output: CS reconstruction result.
            psnr: PSNR of the thresholding results at every iteration.
            r_t: The noisy image before thresholding at the last iteration.
            sigma_hat_t: the estimated standard deviation of the effective noise at the last iteration.
        """
        pass


class DIT(IterativeDenoisingCSRecon):
    """CS solver with Denoising-based Iterative Thresholding.

    Note:
        DIT is similar to DAMP, but without the Onsager correction.
    """
    def __init__(self,
                 image_shape,
                 denoiser,
                 iters,
                 image=None,
                 verbose=False):
        super().__init__(image_shape, denoiser, iters,
                         image, verbose)

    def __call__(self, y, Afun, Atfun, m, n):
        psnr = _setup(self)
        x_t = torch.zeros(n, 1)
        z_t = y.clone()
        for i in range(self.iters):
            # Update x_t
            r_t = (x_t + Atfun(z_t)).view(1, self.H, self.W)
            sigma_hat_t = z_t.norm() / np.sqrt(m)
            x_t = self.denoiser(r_t, std=sigma_hat_t.item())
            _calc_psnr(self, i, x_t, psnr)
            x_t = x_t.view(-1, 1)

            # Update z_t
            z_t = y - Afun(x_t)

            _print(self, i, x_t, sigma_hat_t, psnr)

        output = x_t.view(1, self.H, self.W).cpu()
        return output, psnr, r_t.view(1, self.H, self.W).cpu(), sigma_hat_t.item()

class DAMP(IterativeDenoisingCSRecon):
    """CS solver with Denoising-based Approximate Message Passing."""
    def __init__(self,
                 image_shape,
                 denoiser,
                 iters,
                 image=None,
                 verbose=False):
        super().__init__(image_shape, denoiser, iters,
                         image, verbose)

    def __call__(self, y, Afun, Atfun, m, n):
        psnr = _setup(self)
        eps = 0.001
        x_t = torch.zeros(n, 1)
        z_t = y.clone()
        for i in range(self.iters):
            # Update x_t
            r_t = (x_t + Atfun(z_t)).view(1, self.H, self.W)
            sigma_hat_t = z_t.norm() / np.sqrt(m)
            x_t = self.denoiser(r_t, std=sigma_hat_t.item()) # shape (1, H, W)
            _calc_psnr(self, i, x_t, psnr)
            x_t = x_t.view(-1, 1)

            # Calculate Divergence of r_t
            noise = generate_noise(r_t.shape, std=1.)
            div = (noise * (self.denoiser(r_t + eps * noise, std=sigma_hat_t.item()) -
                            self.denoiser(r_t, std=sigma_hat_t.item())) / eps).sum()

            # Update z_t
            z_t = y - Afun(x_t) + z_t * (div / m)

            _print(self, i, x_t, sigma_hat_t, psnr)

        output = x_t.view(1, self.H, self.W).cpu()
        return output, psnr, r_t.view(1, self.H, self.W).cpu(), sigma_hat_t.item()

def _setup(self):
    if self.image is not None:
        psnr = torch.zeros(self.iters)
    return psnr

def _calc_psnr(self, i, x_t, psnr):
    if self.image is not None:
        psnr[i] = calc_psnr(x_t, self.image)

def _print(self, i, x_t, sigma_hat_t, psnr):
    if self.verbose:
        if self.image is None:
            print('iter {}, approx. std of effective noise {:.3f}'.format(i, sigma_hat_t))
        else:
            print('iter {}, approx. std of effective noise {:.3f}, PSNR {:.3f}'.format(i, sigma_hat_t, psnr[i]))