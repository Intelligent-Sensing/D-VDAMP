"""Error heatmap generators.

    * calc_sure
    * calc_mse
    * calc_perpixel_mse
"""

import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
import torch
from util.general import generate_noise
from util import patch
from util import transform as tutil

def calc_sure(noisy_image,
              denoised_image,
              denoiser,
              std,
              window,
              stride=None,
              num_noise=1,
              heatmap=True,
              test=False):
    """Generate SURE heatmap
    
    Args:
        noisy_image (tensor): tensor of noisy image.
        denoised_image (tensor): tensor of denoised image.
        denoiser: the denoiser used to generate denoised_image.
        std (float): the standard deviation of the Gaussian noise in [0, 1] scale.
        window (int): the width in pixels of the square patch to calculate each SURE value.
        stride (int): the stride in pixels between each square patch to calculate SURE value.
            If None, stride = window.
        num_noise (int): number of noise samples for calculating the divergence term.
        heatmap (bool): whether to return a heatmap. If false, return a 1D tensor of
            SURE values. This tensor can be composed to the heatmap using functions in util.patch.
        test (bool): whether to return the fidelity term and the divergence term along with SURE.

    Returns:
        sure (tensor): SURE
        fidelity (tensor): the fidelity term 1/n ||f(y) - y||^2.
        divergence (tensor): the divergence term (2 std^2)/n div f(y).

    Note:
        The image tensors have shape (CHW). The returns are heatmap or 1D tensor depending on
        the heatmap flag.
    """
    # Prepare image, parameters, and function handles
    C, H, W = noisy_image.shape
    num_patches = patch.calc_num_patches(H, W, window, stride=stride)
    eps = 0.001
    patch_decom = patch.ImageToPatches(window, stride=stride)

    # Fidelity term
    fidelity = patch_decom((noisy_image - denoised_image) ** 2)
    fidelity = fidelity.view(num_patches, -1).sum(1) / (window ** 2)

    # Divergence term
    divergence = torch.zeros(num_patches)
    for _ in range(num_noise):
        random_image = generate_noise(noisy_image.shape, std=1.)
        image_perturbed = noisy_image + eps * random_image
        denoised_perturbed = denoiser(image_perturbed, std=std)
        div_patches = patch_decom(
            random_image * ((denoised_perturbed - denoised_image) / eps))
        divergence += div_patches.view(num_patches, -1).sum(1)
    divergence /= num_noise
    divergence *= (2 * (std ** 2)) / (window ** 2)

    # Combine terms
    sure = fidelity + divergence - std ** 2

    if heatmap:
        patch_recon = patch.Compose([
            patch.VecToPatches(window),
            patch.PatchesToImage(H, W, stride=stride),
        ])
        if test:
            return patch_recon(sure), patch_recon(fidelity), patch_recon(divergence)
        else:
            return patch_recon(sure)
    else:
        if test:
            return sure, fidelity, divergence
        else:
            return sure

def calc_sure_vdamp(noisy_wavelet,
                    denoised_wavelet,
                    denoised_image,
                    denoiser,
                    tau,
                    window,
                    stride=None,
                    num_noise=1,
                    heatmap=True,
                    test=False):
    """Generate SURE heatmap for D-VDAMP
    
    Args:
        noisy_wavelet (util.transform.Wavelet): noisy wavelet (r in D-VDAMP).
        denoised_wavelet (util.transform.Wavelet): denoised wavelet (w_hat in D-VDAMP)
        denoised_image (tensor): denoised image of shape (C, H, W) i.e. inverse of denoised_wavelet.
        denoiser: the denoiser used to generate denoised_image.
        tau (list/array of floats): the variance of noise in each wavelet subband.
        window (int): the width in pixels of the square patch to calculate each SURE value.
        stride (int): the stride in pixels between each square patch to calculate SURE value.
            If None, stride = window.
        num_noise (int): number of noise samples for calculating the divergence term.
        heatmap (bool): whether to return a heatmap. If false, return a 1D tensor of
            SURE values. This tensor can be composed to the heatmap using functions in util.patch.
        test (bool): whether to return the fidelity term and the divergence term along with SURE.

    Returns:
        sure (tensor): SURE
        fidelity (tensor): the fidelity term 1/n ||f(y) - y||^2.
        divergence (tensor): the divergence term (2 std^2)/n div f(y).

    Note:
        When calculating SURE map, we take x_hat to be the inverse wavelet transform of w_hat
        directly without further correction. In many cases, we found that the quality of x_hat
        without the final correction is comparable to the case with final correction.
        See line 11 of Algorithm 2 in

        Charles Millard, Aaron T Hess, Boris Mailhe, and Jared Tanner, 
        “Approximate message passing with a colored aliasing model for variable density 
        fourier sampled images,” arXiv preprint arXiv:2003.02701, 2020.

        for this final correction term.

        The image tensors have shape (CHW). The returns are heatmap or 1D tensor depending on
        the heatmap flag.
    """
    C, H, W = denoised_image.shape
    level = noisy_wavelet.get_bands()
    num_patches = patch.calc_num_patches(H, W, window, stride=stride)
    patch_decom = patch.ImageToPatches(window, stride=stride)
    patch_recon = patch.Compose([patch.VecToPatches(window),
                                patch.PatchesToImage(H, W, stride=stride),
                                ])

    # SURE fidelity term
    fid = (noisy_wavelet.inverse().unsqueeze(0) - denoised_image).abs() ** 2
    fid = patch_decom(fid).view(num_patches, -1).mean(1)

    # SURE bias term
    num_coeffs = noisy_wavelet.count_subbandwise()
    bias = np.sum(num_coeffs * tau) / np.sum(num_coeffs)

    # SURE divergence term (Monte Carlo)
    eps = 0.001
    div = torch.zeros(num_patches)
    for _ in range(num_noise):
        b = generate_noise((H, W), std=1.0)
        jittered_real = tutil.add(noisy_wavelet, tutil.mul_subbandwise(tutil.forward(b, level=level), eps * tau / 2))
        den_jit_real = denoiser(jittered_real, tau, calc_divergence=False)
        diff_real = tutil.sub(den_jit_real, denoised_wavelet)
        div_real = (2 / eps) * (b * diff_real.real().inverse())
        jittered_imag = tutil.add(noisy_wavelet, tutil.mul_subbandwise(tutil.forward(1j * b, level=level), eps * tau / 2))
        den_jit_imag = denoiser(jittered_imag, tau, calc_divergence=False)
        diff_imag = tutil.sub(den_jit_imag, denoised_wavelet)
        div_imag = (2 / eps) * (b * diff_imag.imag().inverse())
        div += patch_decom(div_real.unsqueeze(0) + div_imag.unsqueeze(0)).view(num_patches, -1).mean(1)
    div /= num_noise

    sure = fid + div - bias

    if heatmap:
        patch_recon = patch.Compose([
            patch.VecToPatches(window),
            patch.PatchesToImage(H, W, stride=stride),
        ])
        if test:
            return patch_recon(sure), patch_recon(fid), patch_recon(div)
        else:
            return patch_recon(sure)
    else:
        if test:
            return sure, fid, div
        else:
            return sure

def calc_mse(image,
             denoised_image,
             window,
             stride=None,
             heatmap=True):
    """Generate MSE heatmap
    
    Args:
        image (tensor): tensor of ground truth image.
        denoised_image (tensor): tensor of denoised image.
        window (int): the width in pixels of the square patch to calculate each SURE value.
        stride (int): the stride in pixels between each square patch to calculate SURE value.
        heatmap (bool): whether to return a heatmap. If false, return a 1D tensor of
            SURE values. This tensor can be composed to the heatmap using functions in util.patch.

    Returns:
        mse (tensor): patch-average mean squared error.

    Note:
        The image tensors have shape (C, H, W). The returns are heatmap or 1D tensor depending on
        the heatmap flag.
    """
    # Prepare image, parameters, and function handles
    C, H, W = image.shape
    num_patches = patch.calc_num_patches(H, W, window, stride=stride)
    image_to_patches = patch.ImageToPatches(window, stride=stride)

    # Calculate MSE
    mse = image_to_patches((image - denoised_image).abs() ** 2)
    mse = mse.view(num_patches, -1).sum(1)
    mse = mse / (window ** 2)

    if heatmap:
        recon = patch.Compose([
            patch.VecToPatches(window),
            patch.PatchesToImage(H, W, stride=stride),
        ])
        return recon(mse)
    else:
        return mse

def calc_perpixel_mse(image, denoised_image):
    """Generate per-pixel squared error heatmap

    Args:
        image (tensor): ground truth image.
        denoised_image (tensor): denoiesd image.

    Returns:
        result (tensor): per-pixel squared difference between image and denoised image.
    """
    return (denoised_image - image) ** 2