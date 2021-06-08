"""Image reconstruction simulations.

    * denoise_sim: simulate and solve denoising problem where the noise
        is additive white Gaussian noise (AWGN).
    * cs_sim: simulate and solve compressive sensing problem given
        a measurement operator.
    * vdamp_se_sim: simulate and solve the denoising problem in
        VDAMP state evolution where AWGN is added to each wavelet subband.
    * dvdamp_sim: simulate MRI measurement with variable-density sampling mask
        and reconstruct the image with D-VDAMP.
"""

import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
import torch
from algorithm import dvdamp
from util import general as gutil
from util import transform as tutil
import time

def denoise_sim(image, std, denoiser):
    """Simulate denoising problem

    Args:
        image (torch.Tensor): image tensor with shape (C, H, W).
        std (float): standard deviation of additive Gaussian noise
            on the scale [0., 1.].
        denoiser: a denoiser instance (as in algorithms.denoiser).
            The std argument for this denoiser is already specified
            if applicable.

    Returns:
        denoised_image (torch.Tensor): tensor of denoised image
        noisy_image (torch.Tensor): tensor of noisy image
    """
    print('deploy.sim.denoise_sim: Simulating noisy image...')
    noisy_image = gutil.add_noise(image, std)

    print('deploy.sim.denoise_sim: Begin image denoising...')
    denoised_image = denoiser(noisy_image, std=std)

    return denoised_image, noisy_image

def cs_sim(image, cs_transform, std, cs_algo):
    """Simulate compressive sensing problem

    Args:
        image (torch.Tensor): image tensor with shape (C, H, W).
        cs_transform (CStransform): a CS transformation instance.
        std (float): standard deviation of additive Gaussian noise
            on the scale [0., 1.].
        cs_algo (algorithms.csalgo.IterativeDenoisingCSRecon):
            an instance of CS reconstruction algorithm.

    Returns:
        recon_image (torch.Tensor): tensor of reconstructed image.
        r_t (torch.Tensor): tensor of the pseudo noisy image (r_t)
            from last iteration of cs_algo.
        std_est (float): estimated standard deviation of noise in r_t
        psnr (np.ndarray): PSNR at each iteration of reconstruction with cs_algo.
            psnr is None if cs_algo does not have the reference ground truth image.
    """
    print('deploy.sim.cs_sim: Simulating compressive sensing...')
    y = cs_transform.Afun(image.view(-1, 1))
    if std > 0:
        y += torch.normal(mean=torch.zeros(y.shape), std=std)

    print('deploy.sim.cs_sim: Begin compressive sensing reconstruction...')
    start_time = time.time()
    recon_image, psnr, r_t, std_est = cs_algo(y, cs_transform.Afun, cs_transform.Atfun,
                                              cs_transform.get_m(), cs_transform.get_n())
    print('deploy.sim.cs_sim: Reconstruction took {:.3f} s'.format(time.time() - start_time))

    return recon_image, r_t, std_est, psnr

def vdamp_se_sim(image, tau, denoiser, level=4, is_complex=True):
    """Simulate VDAMP State Evolution

    Args:
        image (torch.Tensor): image tensor with shape (C, H, W).
        tau (array): array of variance of noise in each wavelet subband.
        denoiser: a denoiser instance from algorithm.vdamp.
        level (int): the level of wavelet decomposition.
        is_complex (bool): whether to add complex noise.

    Returns:
        image_denoised (tutil.Wavelet): the denoised image i.e. inverse wavelet transform
            of the denoised wavelet.
        wavelet_noisy (tutil.Wavelet): the noisy wavelet.
        wavelet_denoised (tutil.Wavelet): the denoised wavelet.

    Notes:
        The denoiser must be the kind that handles wavelet coefficients.
        Some can be found in algorithm.vdamp.
    """

    if complex:
        dtype = np.complex64
    else:
        dtype = np.float32

    std = np.sqrt(tau)
    wavelet_image = tutil.forward(image[0], level=level).astype(dtype)   # x
    wavelet_noisy = tutil.add_noise_subbandwise(wavelet_image, std,
                                                 is_complex=is_complex)  # r
    wavelet_denoised = denoiser(wavelet_noisy, tau, calc_divergence=False)                           # w_hat
    image_denoised = wavelet_denoised.inverse().unsqueeze(0)             # x_hat

    return image_denoised, wavelet_noisy, wavelet_denoised

def dvdamp_sim(image, sampling_rate, snr, denoiser, iters, 
                level=4, wavetype='haar', stop_on_increase=True):
    """Simulate MRI measurement with variable-density sampling and reconstruct
        the image with D-VDAMP.

    Args:
        image (torch.Tensor): image tensor with shape (C, H, W).
        sampling_rate (float): sampling rate of the measurement.
        snr (float): signal-to-noise ratio of the measurement.
        denoiser: a denoiser instance from algorithm.vdamp.
        iters (int): maximum number of iterations for D-VDAMP.
        level (int): the level of wavelet decomposition.
        wavetype (str): type of the wavelet in wavelet transform.
        stop_on_increase (bool): whether to stop D-VDAMP when the predicted MSE increases.

    Returns:
        recon_image (torch.Tensor): the reconstructed image.
        log (dict): log of D-VDAMP containing the following information in each iteration.
            - reconstruction, x_hat
            - noisy wavelet, r
            - denoised wavelet, w_hat
            - predicted noise variance, tau
            - mean squared error, err (if the ground truth image is given to D-VDAMP)
        true_iters (int): the number of iterations before D-VDAMP terminates.

    Notes:
        We assume that the ground truth is a real image.
        The denoiser must be the kind that handles wavelet coefficients.
        Some can be found in algorithm.vdamp.
    """
    prob_map = dvdamp.gen_pdf(image.shape, sampling_rate)
    mask = np.random.binomial(1, prob_map)
    var0 = ((image.abs() ** 2).mean() / (10 ** (0.1 * snr))).item()
    y = mask * (tutil.fftnc(image[0]) + gutil.generate_noise(mask.shape, np.sqrt(var0), ret_array=True))
    x_hat, log, true_iters = dvdamp.dvdamp(y, prob_map, mask, var0, denoiser,
                                    image=image[0], iters=iters, level=level, 
                                    wavetype=wavetype, stop_on_increase=stop_on_increase)
    recon_image = gutil.im_numpy_to_tensor(x_hat)
    return recon_image, log, true_iters