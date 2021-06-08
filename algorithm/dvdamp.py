"""D-VDAMP algorithm, MRI measurement, and wavelet denoisers.

    D-VDAMP algorithm
        * dvdamp
        * calc_wavespec

    MRI measurement
        * gen_pdf

    Denoisers
        * calc_MC_divergence
        * ColoredDnCNN_VDAMP
        * BM3D_VDAMP
        * SoftThresholding_VDAMP

    Note:
        The "wavelet" format means list of lists where the position of each value
        belongs to the corresponding wavelet subband. For example, the predicted
        variance tau in dvdamp with level of decomposition = 2 is in the format:

            [A1, [H1, V1, D1], [H2, V2, D2]]

        where each value is the variance corresponding to a subband. Note that
        earlier index (lower number in the example above) means smaller scale.
"""

import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
from numpy import fft
import torch
from bm3d import bm3d
from algorithm.denoiser import ColoredDnCNN
from util import transform as tutil
from util.general import load_checkpoint, save_image

"""VDAMP algorithm"""

def dvdamp(y,
        prob_map,
        mask, 
        var0,
        denoiser,
        image=None,
        iters=30,
        level=4,
        wavetype='haar',
        stop_on_increase=True):
    """Perform VDAMP

    Args:
        y (np.ndarray, (H, W)): MRI measurement.
        prob_map (np.ndarray, (H, W)): sampling probability map.
        mask (np.ndarray, (H, W)): sampling mask generated from prob_map.
        var0 (float): variance of measurement noise.
        image (np.ndarray, (H, W)): Ground truth image.
        iters (int): number of iterations.
        level (int): number of levels for wavelet transform.
        wavetype (str): wavelet type. Refer to pywt for options.
        stop_on_increase (bool): whether to stop D-VDAMP when the predicted MSE increases.

    Returns:
        x_hat (np.ndarray, (H, W)): reconstructed image.
        log (dict): reconstruction log, containing reconstructed image (x_hat),
            wavelet pyramids before (r) and after thresholding (w_hat), and
            true (err) and estimated (tau) RMSE of effective noise for each iteration.
        true_iter (int): actual number of iterations before stopping. This is equal to
            iters if stop_on_increase is False.

    Notes:
        The algorithm is Algorithm 1 described in

            C. A. Metzler and G. Wetzstein, "D-VDAMP: Denoising-Based Approximate Message Passing
            for Compressive MRI," ICASSP 2021 - 2021 IEEE International Conference on Acoustics, 
            Speech and Signal Processing (ICASSP), 2021, pp. 1410-1414, doi: 10.1109/ICASSP39728.2021.9414708.

        This function follows the MATLAB code of VDAMP closely with the soft-thresholding denoiser
        changed to a generic denoiser.

            Charles Millard, Aaron T Hess, Boris Mailhe, and Jared Tanner, 
            “Approximate message passing with a colored aliasing model for variable density 
            fourier sampled images,” arXiv preprint arXiv:2003.02701, 2020.
    """

    # Precompute
    H, W = y.shape
    specX = calc_wavespec(H, level)
    specY = calc_wavespec(W, level)
    Pinv = 1 / prob_map      # Pinv is element-wise inverse
    Pinvm1 = Pinv - 1
    if image is not None:
        w0 = tutil.forward(image, wavelet=wavetype, level=level)
        log = {
            'x_hat' : np.zeros((iters, H, W), dtype=complex),
            'r' : np.zeros((iters, H, W), dtype=complex),
            'w_hat' : np.zeros((iters, H, W), dtype=complex),
            'err' : [None] * iters,
            'tau' : [None] * iters
        }

    # Initialize
    r = tutil.forward(tutil.ifftnc(Pinv * mask * y), wavelet=wavetype, level=level)
    tau_y = mask * Pinv * (Pinvm1 * np.abs(y) ** 2 + var0)
    tau = [None] * (level + 1)
    tau[0] = (specX[:, 0, 0].reshape(1, -1) @ tau_y @ specY[:, 0, 0].reshape(-1, 1)).item()
    for b in range(level):
        tau_b = [None] * 3
        tau_b[0] = (specX[:, b, 1].reshape(1, -1) @ tau_y @ specY[:, b, 0].reshape(-1, 1)).item()
        tau_b[1] = (specX[:, b, 0].reshape(1, -1) @ tau_y @ specY[:, b, 1].reshape(-1, 1)).item()
        tau_b[2] = (specX[:, b, 1].reshape(1, -1) @ tau_y @ specY[:, b, 1].reshape(-1, 1)).item()
        tau[b + 1] = tau_b
    pred_MSE_prev_iter = np.inf
    true_iters = 0

    _, slices = r.pyramid_forward(get_slices=True)
    log['slices'] = slices

    # Loop
    for it in range(iters):

        # Thresholding
        w_hat, alpha = denoiser(r, tutil.reformat_subband2array(tau))

        # Calculate x_hat
        # x_hat = w_hat.inverse(to_tensor=False)
        x_tilde = w_hat.inverse(to_tensor=False)
        x_hat = x_tilde + tutil.ifftnc(mask * (y - tutil.fftnc(x_tilde)))

        log['r'][it] = r.pyramid_forward(to_tensor=False)
        log['w_hat'][it] = w_hat.pyramid_forward(to_tensor=False)
        log['x_hat'][it] = x_hat
        log['tau'][it] = tau
        if image is not None:
            log['err'][it] = _calc_mse(r, w0)

        if stop_on_increase:
            true_iters += 1
            pred_MSE_this_iter = _calc_pred_mse(tau, level)
            if pred_MSE_this_iter > pred_MSE_prev_iter:
                break
            else:
                pred_MSE_prev_iter = pred_MSE_this_iter
        else:
            true_iters += 1

        # Onsager correction
        w_tilde_coeff = [None] * (level + 1)
        w_tilde_coeff[0] = w_hat.coeff[0] - alpha[0] * r.coeff[0]
        w_div = np.sum(r.coeff[0] * w_tilde_coeff[0]) / (np.sum(w_tilde_coeff[0] ** 2))
        w_tilde_coeff[0] *= w_div
        for b in range(1, level + 1):
            w_tilde_coeff_b = [None] * 3
            for s in range(3):
                w_tilde_coeff_b[s] = w_hat.coeff[b][s] - alpha[b][s] * r.coeff[b][s]
                w_div = np.sum(r.coeff[b][s] * w_tilde_coeff_b[s]) / (np.sum(w_tilde_coeff_b[s] ** 2))
                w_tilde_coeff_b[s] *= w_div
            w_tilde_coeff[b] = w_tilde_coeff_b
        w_tilde = tutil.Wavelet(w_tilde_coeff)

        # Reweighted gradient step
        z = mask * (y - tutil.fftnc(w_tilde.inverse(to_tensor=False)))
        r = tutil.add(w_tilde, tutil.forward(tutil.ifftnc(Pinv * z), wavelet=wavetype, level=level))

        # Noise power re-estimation
        tau_y = mask * Pinv * (Pinvm1 * np.abs(z) ** 2 + var0)
        tau = [None] * (level + 1)
        tau[0] = (specX[:, 0, 0].reshape(1, -1) @ tau_y @ specY[:, 0, 0].reshape(-1, 1)).item()
        for b in range(level):
            tau_b = [None] * 3
            tau_b[0] = (specX[:, b, 1].reshape(1, -1) @ tau_y @ specY[:, b, 0].reshape(-1, 1)).item()
            tau_b[1] = (specX[:, b, 0].reshape(1, -1) @ tau_y @ specY[:, b, 1].reshape(-1, 1)).item()
            tau_b[2] = (specX[:, b, 1].reshape(1, -1) @ tau_y @ specY[:, b, 1].reshape(-1, 1)).item()
            tau[b + 1] = tau_b

    return x_hat, log, true_iters

def _calc_mse(test, ref):
    """Calculate band-wise mean squared error (MSE).

    Args:
        test (util.transform.Wavelet): noisy (test) wavelet.
        ref (util.transform.Wavelet): ground truth (reference) wavelet.

    Returns:
        mse (list): list of MSE in the "wavelet" format.
    """
    mse = [None] * (test.get_bands() + 1)
    mse[0] = np.mean(np.abs(test.coeff[0] - ref.coeff[0]) ** 2)
    for b in range(1, test.get_bands() + 1):
        mse_band = [None] * 3
        for s in range(3):
            mse_band[s] = np.mean(np.abs(test.coeff[b][s] - ref.coeff[b][s]) ** 2)
        mse[b] = mse_band
    return mse

def _calc_pred_mse(tau, level):
    """Calculate a scaled predicted MSE based on 
    the predicted noise variance and the level of wavelet decomposition.

    Args:
        tau (list): predicted noise variance in each subband in the "wavelet" format.

    Returns:
        pred_mse (float): scaled predicted MSE.

    Note:
        The predicted MSE is used to determine whether to stop D-VDAMP early.
    """
    pred_mse = 0
    pred_mse += tau[0] * (4 ** (-level))
    for b in range(level):
        weight = 4 ** (b - level)
        for s in range(3):
            pred_mse += tau[b + 1][s] * weight
    return pred_mse

def calc_wavespec(numsamples, level, wavetype='haar', ret_tensor=False):
    """Calculate power spectrum of wavelet decomposition kernels.

    Returns:
        spec: (numsamples, level, [lowpass, highpass]) power spectrum.
            In axis=1 (level), higher indices mean larger scales.
    """

    wavelet = tutil.Wavelet_bank(wavetype)
    spec = np.zeros([numsamples, level, 2])

    # Zero-pad decomposition filters
    L = np.zeros(numsamples)
    L[0:len(wavelet.dec_lo)] = wavelet.dec_lo
    H = np.zeros(numsamples)
    H[0:len(wavelet.dec_hi)] = wavelet.dec_hi

    # Spectrum of the largest scale
    spec[:, 0, 0] = np.abs(fft.fft(L)) ** 2
    spec[:, 0, 1] = np.abs(fft.fft(H)) ** 2

    # Spectrum of other scales
    numblock = 1
    for s in range(1, level):
        numblock *= 2
        spec[:, s, 0] = spec[:, s - 1, 0] * \
            np.transpose(spec[::numblock, 0, 0].reshape(-1, 1) @ np.ones([1, numblock])).reshape(-1)
        spec[:, s, 1] = spec[:, s - 1, 0] * \
            np.transpose(spec[::numblock, 0, 1].reshape(-1, 1) @ np.ones([1, numblock])).reshape(-1)
    spec = fft.fftshift(spec, axes=0) / numsamples

    if ret_tensor:
        return torch.from_numpy(spec).flip(1).to(torch.float32)
    else:
        return np.flip(spec, axis=1)

"""Functions for MRI sampling simulation"""

def gen_pdf(shape, sampling_rate, p=8, dist_type='l2', radius=0., ret_tensor=False):
    """Generate probability density function (PDF) for variable density undersampling masking in MRI simulation

    Args:
        shape: shape of image
        sampling_rate (float): ratio of sampled pixels to ground truth pixels (n/N)
        p (int): polynomial power
        dist_type (str): distance type - l1 or l2
        radius (float): radius of fully sampled center

    Returns:
        pdf (np.ndarray): the desired PDF (sampling probability map)

    Notes:
        This is the Python implementation of the genPDF function from the SparseMRI package.
        (http://people.eecs.berkeley.edu/~mlustig/Software.html). The sampling scheme is described
        in the paper M. Lustig, D.L Donoho and J.M Pauly “Sparse MRI: The Application of Compressed
        Sensing for Rapid MR Imaging” Magnetic Resonance in Medicine, 2007 Dec; 58(6):1182-1195.

    """
    C, H, W = shape

    num_samples = np.floor(sampling_rate * H * W)

    x, y = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W))
    if dist_type == 'l1':
        r = np.maximum(np.abs(x), np.abs(y))
    elif dist_type == 'l2':
        r = np.sqrt(x ** 2 + y ** 2)
        r /= np.max(np.abs(r))
    else:
        raise ValueError('genPDF: invalid dist_type')

    idx = np.where(r < radius)

    pdf = (np.ones_like(r) - r) ** p
    pdf[idx] = 1

    if np.floor(np.sum(pdf)) > num_samples:
        raise RuntimeError('genPDF: infeasible without undersampling dc, increase p')

    # Bisection
    minval = 0
    maxval = 1
    val = 0.5
    it = 0
    for _ in range(20):
        it += 1
        val = (minval + maxval) / 2
        pdf = (np.ones_like(r) - r) ** p + val * np.ones_like(r)
        pdf[np.where(pdf > 1)] = 1
        pdf[idx] = 1
        N = np.floor(np.sum(pdf))
        if N > num_samples:		# Infeasible
            maxval = val
        elif N < num_samples:	# Feasible, but not optimal
            minval = val
        elif N == num_samples:	# Optimal
            break
        else:
            raise RuntimeError('genPDF: error with calculation of N')

    if ret_tensor:
        return torch.from_numpy(pdf).to(dtype=torch.float32)
    else:
        return pdf

"""Denoising functions for VDAMP"""

def calc_MC_divergence(denoiser, denoised, wavelet, variances):
    """Calculate the divergence required by D-VDAMP using a Monte Carlo approach.

    Note:
        See section 2.1 of Metzler and Wetzstein 2020, "D-VDAMP: Denoising-Based Approximate Message Passing
        for Compressive MRI" for the equation.
    """
    level = wavelet.get_bands()
    alpha = [None] * (level + 1)
    wavelet_jittered = wavelet.copy()
    eta = np.abs(wavelet_jittered.coeff[0].max()) / 1000.
    noise_vec = np.random.randn(*wavelet_jittered.coeff[0].shape) + 1j * np.random.randn(*wavelet_jittered.coeff[0].shape)
    wavelet_jittered.coeff[0] += eta * noise_vec
    denoised_jittered = denoiser(wavelet_jittered, variances)
    alpha[0] = 0.5 * (
                1. / wavelet_jittered.coeff[0].size * np.dot(np.real(noise_vec).reshape(-1),
                    np.real(denoised_jittered.coeff[0] - denoised.coeff[0]).reshape(-1) / eta) + # real part
                1. / wavelet_jittered.coeff[0].size * np.dot(np.imag(noise_vec).reshape(-1),
                    np.imag(denoised_jittered.coeff[0] - denoised.coeff[0]).reshape(-1) / eta)) # img part
    for s in range(level):
        alpha[s + 1] = 3 * [None]
        for b in range(3):
            wavelet_jittered = wavelet.copy()
            eta = np.abs(wavelet_jittered.coeff[s + 1][b].max()) / 1000.
            noise_vec = np.random.randn(*wavelet_jittered.coeff[s + 1][b].shape) + 1j * np.random.randn(*wavelet_jittered.coeff[s + 1][b].shape)
            wavelet_jittered.coeff[s + 1][b] += eta * noise_vec
            denoised_jittered = denoiser(wavelet_jittered, variances)
            alpha[s + 1][b] = 0.5 * (
                    1. / wavelet_jittered.coeff[s + 1][b].size * np.dot(np.real(noise_vec).reshape(-1),
                        np.real(denoised_jittered.coeff[s + 1][b] - denoised.coeff[s + 1][b]).reshape(-1) / eta) +  # real part
                    1. / wavelet_jittered.coeff[s + 1][b].size * np.dot(np.imag(noise_vec).reshape(-1),
                        np.imag(denoised_jittered.coeff[s + 1][b] - denoised.coeff[s + 1][b]).reshape(-1) / eta)) # img part
    return alpha

class ColoredDnCNN_VDAMP:
    """Wrapper of algorithm.denoiser.ColoredDnCNN for using with D-VDAMP.
    
    Note:
        In D-VDAMP, the noisy wavelets have complex coefficients.

        For real ground truths (model channel = 1), we appply the denoiser to
        only the real part and scale the imaginary part by 0.1.

        For complex ground truths (model channel = 2), we pass a tensor where
        the first channel is the real part and the second channel is the imaginary part
        to the model. However, this feature is not currently supported.
    """
    def __init__(self, modeldir, std_ranges, channels=1, wavetype='haar',
                num_layers=20, std_channels=13, device=torch.device('cpu'),
                std_pool_func=np.mean, verbose=False):
        """Initialize ColoredDnCNN_VDAMP

        Args:
            modeldirs (str): path to directory containing model weights.
            std_ranges (array): range of noise std for each denoiser.
                For example, [0, 20, 50, 120, 500] / 255 means that
                denoiser 1 is for noise with std 0 to 20 / 255.
                denoiser 2 is for noise with std 20 / 255 to 50 / 255.
                denoiser 3 is for noise with std 50 / 255 to 120 / 255.
                denoiser 4 is for noise with std 120 / 255 to 500 / 255.
            channels (int): number of channels in the model.
            wavetype (str): type of wavelet transform.
            num_layers (int): number of layers in the model.
            std_channels (int): number of std channels for the model i.e.
                number of wavelet subbands.
            device: the device to run the model on.
            std_pool_func (callable): function for pooling the std values in all subbands
                to determine which denoiser model to use.
        """
        self.channels = channels
        self.std_ranges = std_ranges
        self.wavetype = wavetype
        self.device = device
        self.models = self._load_models(modeldir, num_layers, std_channels)
        self.std_pool_func = std_pool_func
        self.verbose = verbose

    def __call__(self, wavelet, variances, gamma=1., calc_divergence=True):
        """Denoise the wavelet and calculate the divergence

        Args:
            wavelet (util.transform.Wavelet): the noisy wavelet.
            variances (array): the variance of noise in each wavelet subband.
            gamma (float): scaling on the variances.

        Returns:
            denoised (util.transform.Wavelet): the denoised wavelet.
            alpha (list): the divergence in each subband.
        """
        # variances = tutil.reformat_subband2array(variances) * gamma
        variances *= gamma
        denoised = self._denoise(wavelet, variances)
        if calc_divergence:
            alpha = calc_MC_divergence(self._denoise, denoised, wavelet, variances)
            return denoised, alpha
        else:
            return denoised

    @torch.no_grad()
    def _denoise(self, wavelet, variances):
        level = wavelet.get_bands()

        # Select the model to use
        stds = torch.from_numpy(variances).sqrt().to(device=self.device, dtype=torch.float32)
        std_pooled = self.std_pool_func(np.sqrt(variances))
        select = np.sum(std_pooled > self.std_ranges) - 1
        if select < 0:
            select += 1
        elif select > len(self.models) - 1:
            select -= 1
        if self.verbose:
            print('ColoredDnCNN_VDAMP select: {}'.format(select))

        # Denoise
        noisy_image = wavelet.inverse().unsqueeze(0)
        if self.channels == 1:
            noisy_real = noisy_image.real.to(device=self.device, dtype=torch.float32)
            noisy_imag = noisy_image.imag
            denoised_real = self.models[select](noisy_real, stds).cpu()
            denoised_imag = noisy_imag * 0.1
            denoised_image = denoised_real + 1j * denoised_imag
            denoised_wavelet = tutil.forward(denoised_image[0], wavelet=self.wavetype, level=level)
        elif self.channels == 2:
            noisy_image = torch.vstack([noisy_image.real, noisy_image.imag])
            noisy_image = noisy_image.to(device=self.device, dtype=torch.float32)
            denoised_image = self.models[select](noisy_image, stds).cpu()
            denoised_image = denoised_image[0] + 1j * denoised_image[1]
            denoised_wavelet = tutil.forward(denoised_image, wavelet=self.wavetype, level=level)
        else:
            raise ValueError('Only support channel == 1 or 2.')
        return denoised_wavelet

    def _load_models(self, modeldirs, num_layers, std_channels):
        models = [None] * len(modeldirs)
        for i, modeldir in enumerate(modeldirs):
            model = ColoredDnCNN(channels=self.channels, num_layers=num_layers, std_channels=std_channels)
            load_checkpoint(modeldir, model, None, device=self.device)
            model.to(device=self.device)
            model.eval()
            models[i] = model
        return models

class BM3D_VDAMP:
    """Wrapper of BM3D for using with D-VDAMP."""
    def __init__(self, channels, wavetype='haar', std_pool_func=np.max):
        """Initialize BM3D_VDAMP

        Args:
            channels (int): number of channels to apply BM3D.
                If channels == 1, apply BM3D to the real part and scale the imaginary part by 0.1.
                If channels == 2, apply BM3D to both real and imaginary parts seperately.
            wavetype (str): type of wavelet transform.
            std_pool_func (callable): function for pooling the std values in all subbands
                to determine which denoiser model to use.
        """
        self.channels = channels
        self.std_pool_func = std_pool_func
        self.wavetype = wavetype

    def __call__(self, wavelet, variances, calc_divergence=True):
        """Denoise the wavelet and calculate the divergence

        Args:
            wavelet (util.transform.Wavelet): the noisy wavelet.
            variances (array): the variance of noise in each wavelet subband.

        Returns:
            denoised (util.transform.Wavelet): the denoised wavelet.
            alpha (list): the divergence in each subband.
        """
        # variances = tutil.reformat_subband2array(variances)
        denoised = self._denoise(wavelet, variances)
        if calc_divergence:
            alpha = calc_MC_divergence(self._denoise, denoised, wavelet, variances)
            return denoised, alpha
        else:
            return denoised

    def _denoise(self, wavelet, variances):
        level = wavelet.get_bands()
        std_pooled = self.std_pool_func(np.sqrt(variances))
        noisy_image = wavelet.inverse()
        if self.channels == 1:
            noisy_real = noisy_image.real
            noisy_imag = noisy_image.imag
            denoised_real = torch.Tensor(bm3d(noisy_real, std_pooled))
            denoised_imag = noisy_imag * 0.1
            denoised_image = denoised_real + 1j * denoised_imag
            denoised_wavelet = tutil.forward(denoised_image, wavelet=self.wavetype, level=level)
        elif self.channels == 2:
            noisy_real = noisy_image.real
            noisy_imag = noisy_image.imag
            denoised_real = torch.Tensor(bm3d(noisy_real, std_pooled))
            denoised_imag = torch.Tensor(bm3d(noisy_imag, std_pooled))
            denoised_image = denoised_real + 1j * denoised_imag
            denoised_wavelet = tutil.forward(denoised_image, wavelet=self.wavetype, level=level)
        else:
            raise ValueError('Only support channel == 1 or 2.')
        return denoised_wavelet

class SoftThresholding_VDAMP:
    """Wrapper of soft-thresholding for using with D-VDAMP.
    
    Note:
        With soft-thresholding as the denoiser, the D-VDAMP algorithm becomes the base VDAMP.

            Charles Millard, Aaron T Hess, Boris Mailhe, and Jared Tanner, 
            “Approximate message passing with a colored aliasing model for variable density 
            fourier sampled images,” arXiv preprint arXiv:2003.02701, 2020.

    """
    def __init__(self, MC_divergence, debug=False):
        """Initialize SoftThresholding_VDAMP

        Args:
            MC_divergence (bool): whether to calculate the divergence with the 
                Monte Carlo approach (True) or analytically (False).
            debug (bool): whether to calculate the analytical divergence as well
                for comparison when MC_divergence is True.
        """
        self.MC_divergence = MC_divergence
        self.debug = debug

    def __call__(self, wavelet, variances, calc_divergence=True):
        variances = tutil.reformat_array2subband(variances)
        denoised, df, _ = multiscaleSureSoft(wavelet, variances)
        if calc_divergence:
            if self.MC_divergence:
                alpha = calc_MC_divergence(self._denoise, denoised, wavelet, variances)
                if self.debug:
                    alpha_ana = self._calc_ana_div(df)
                    alpha_array = tutil.reformat_subband2array(alpha)
                    alpha_ana_array = tutil.reformat_subband2array(alpha_ana)
                    error = ((alpha_array - alpha_ana_array) / alpha_ana_array * 100).mean()
                    print('MC div error: {} %'.format(error))
            else:
                alpha = self._calc_ana_div(df)
            return denoised, alpha
        else:
            return denoised

    # wrapper for calc_MC_divergence
    def _denoise(self, wavelet, variances):
        denoised, _, _ = multiscaleSureSoft(wavelet, variances)
        return denoised

    def _calc_ana_div(self, df):
        level = len(df) - 1
        alpha = [None] * (level + 1)
        alpha[0] = np.mean(df[0]) / 2
        for b in range(level):
            alpha_b = [None] * 3
            for s in range(3):
                alpha_b[s] = np.mean(df[b + 1][s]) / 2
            alpha[b + 1] = alpha_b
        return alpha

"""Base implementation of soft thresholding"""

def complexSoft(wavelet_coeff, threshold):
    """Perform soft-thresholding on a wavelet subband given a threshold.

    Args:
        wavelet_coeff (np.ndarray): array of wavelet coefficients.
        threshold (float): the threshold.

    Returns:
        thresholded_coeff (np.ndarray): the thresholded wavelet coefficients.
        df (np.ndarray): degree of freedom, shape like thresholded_coeff

    Notes:
        The original MATLAB implementation can be found at https://github.com/charlesmillard/VDAMP.
    """
    ones = np.ones_like(wavelet_coeff, dtype=float)
    mag = np.abs(wavelet_coeff)
    gdual = np.minimum(threshold / mag, ones)
    thresholded_coeff = wavelet_coeff * (ones - gdual)
    df = 2 * ones - (2 * ones - (gdual < 1)) * gdual
    return thresholded_coeff, df

def sureSoft(wavelet_coeff, var):
    """
    Perform soft-thresholding on a wavelet subband using optimal threshold estimated with SURE.

    Args:
        wavelet_coeff (np.ndarray): array of wavelet coefficients.
        var (float): variance of wavelet_coeff.

    Returns:
        thresholded_coeff (np.ndarray): the thresholded wavelet coefficients.
        df (np.ndarray): degree of freedom, shape like thresholded_coeff
        threshold (float): the threshold used.

    Notes:
        The original MATLAB implementation can be found at https://github.com/charlesmillard/VDAMP.
    """

    mag_flat = np.abs(wavelet_coeff.reshape(-1))
    index = np.flipud(np.argsort(np.abs(mag_flat)))
    lamb = mag_flat[index]

    V = var * np.ones_like(mag_flat)
    V = V[index]

    z0 = np.ones_like(lamb)

    SURE_inf = np.flipud(np.cumsum(np.flipud(lamb ** 2)))
    SURE_sup = np.cumsum(z0) * (lamb ** 2) - lamb * np.cumsum(V / lamb) + 2 * np.cumsum(V)
    SURE = SURE_inf + SURE_sup - np.sum(V)

    idx = np.argmin(SURE)
    thresholded_coeff, df = complexSoft(wavelet_coeff, lamb[idx])
    threshold = lamb[idx]

    return thresholded_coeff, df, threshold

def multiscaleComplexSoft(wavelet, variances, lambdas):
    """
    Perform soft-thresholding on wavelet coefficients given sparse weighings.

    Args:
        wavelet (util.transform.Wavelet): the target wavelet.
        variances (list): estimated variances of the bands.
        lambdas (list): sparse weighings.

    Returns:
        thresholded_wavelet (util.transform.Wavelet): the thresholded wavelet.
        df (list): degrees of freedom of bands.

    Notes:
        The original MATLAB implementation can be found at https://github.com/charlesmillard/VDAMP.
    """
    
    scales = wavelet.get_bands()
    thresholded_wavelet = wavelet.copy()
    df = [None] * (scales + 1)

    thresholded_wavelet.coeff[0], df[0] = complexSoft(wavelet.coeff[0], variances[0] * lambdas[0])
    for i in range(1, scales + 1):
        df_subband = [None] * 3
        for j in range(3):
            thresholded_wavelet.coeff[i][j], df_subband[j] = complexSoft(wavelet.coeff[i][j], variances[i][j] * lambdas[i][j])
        df[i] = df_subband

    return thresholded_wavelet, df

def multiscaleSureSoft(wavelet, variances):
    """
    Perform soft-thresholding on wavelet coefficients using optimal thresholds estimated with SURE.

    Args:
        wavelet (util.transform.Wavelet): the target wavelet.
        variances (list): estimated variances of the bands.

    Returns:
        thresholded_wavelet (util.transform.Wavelet): the thresholded wavelet.
        df (list): degrees of freedom of bands.
        thres (list): used thresholds for bands.

    Notes:
        The original MATLAB implementation can be found at https://github.com/charlesmillard/VDAMP.
    """
    
    scales = wavelet.get_bands()
    thresholded_wavelet = wavelet.copy()
    df = [None] * (scales + 1)
    thres = [None] * (scales + 1)

    thresholded_wavelet.coeff[0], df[0], thres[0] = sureSoft(wavelet.coeff[0], variances[0])
    for i in range(1, scales + 1):
        df_subband = [None] * 3
        thres_subband = [None] * 3
        for j in range(3):    
            thresholded_wavelet.coeff[i][j], df_subband[j], thres_subband[j] = sureSoft(wavelet.coeff[i][j], variances[i][j])
        df[i] = df_subband
        thres[i] = thres_subband

    return thresholded_wavelet, df, thres