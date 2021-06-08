"""Fourier transform, Wavelet transform, and related functions

    Fourier transform
        * fftnc
        * ifftnc

    Wavelet transform
        * reformat_subband2array
        * forward
        * pyramid_backward
        * add
        * sub
        * mul_subbandwise
        * threshold_subbandwise
        * add_noise_subbandwise
        * Wavelet class
            * get_bands
            * get_subbands
            * count_subbandwise
            * inverse
            * pyramid_forward
            * real
            * imag
            * copy
            * astype
    
    Misc.
        * Wavelet_bank
        * SUBBAND
        * SUBBAND_COLOR

Note that the format of the wavelet coefficients in the list is [cA, [cH, cV, cD], ...].
"""

import numpy as np
import numpy.fft as fft
import torch
import pywt
from copy import deepcopy
from util import general as gutil

"""Fourier Transform functions"""

def fftnc(x, ret_tensor=False):
    """Normalized FFT of x with low frequency at center"""
    X = fft.fftshift(fft.fft2(fft.ifftshift(x)))
    X /= np.sqrt(np.prod(x.shape))
    if ret_tensor:
        return torch.from_numpy(X).to(dtype=torch.complex64)
    else:
        return X.astype(np.complex64)

def ifftnc(X, ret_tensor=False):
    """Inverse FFT for normalized X with low frequency at center"""
    x = fft.fftshift(fft.ifft2(fft.ifftshift(X)))
    x *= np.sqrt(np.prod(X.shape))
    if ret_tensor:
        return torch.from_numpy(x).to(dtype=torch.complex64)
    else:
        return x.astype(np.complex64)

"""Wavelet Transform functions"""

def reformat_subband2array(subband_val):
    """Reformat variable formatted by subband, e.g. per-subband variance, into an array.

    Args:
        subband_val (list): subband-wise values, organized as (A, (H1, W1, D1), (H2, W2, D2), ...)

    Returns:
        result (np.ndarray): subband-wise values sorted in order ([A, H1, W1, D1, H2, W2, D2, ...])
    """
    level = len(subband_val) - 1
    result = np.zeros(1 + 3 * level)
    result[0] = subband_val[0]
    for b in range(level):
        for s in range(3):
            result[3 * b + s + 1] = subband_val[b + 1][s]
    return result

def reformat_array2subband(array):
    """Reformat array values to the wavelet subband format.

    Args:
        array (np.ndarray): subband-wise values sorted in order ([A, H1, W1, D1, H2, W2, D2, ...])

    Returns:
        subband_val (list): subband-wise values, organized as (A, (H1, W1, D1), (H2, W2, D2), ...)
        
    """
    num_values = len(array)
    level = (num_values - 1) // 3
    subband_val = [None] * (level + 1)
    subband_val[0] = array[0]
    idx = 1
    for b in range(level):
        this_band = [None] * 3
        for s in range(3):
            this_band[s] = array[idx]
            idx += 1
        subband_val[b + 1] = this_band
    return subband_val

def forward(image, wavelet='haar', level=4):
    """Wavelet transform.

    Args:
        image (np.ndarray): image to apply wavelet transform.
        wavelet (str): type of wavelet. Refer to PyWavelets for the options.
        level (int): level of wavelet decomposition.

    Returns:
        result (Wavelet): resulting wavelet.
    """
    return Wavelet(pywt.wavedec2(image, wavelet, level=level))

def pyramid_backward(pyramid, slices):
    """Recover Wavelet object from the wavelet pyramid.

    Args:
        pyramid (np.ndarray): the wavelet pyramid.
        slices (list of tuples): list of slices obtained from pyramid_forward required by pywt.array_to_coeffs.

    Returns:
        result (Wavelet): resulting wavelet.
    """
    return Wavelet(pywt.array_to_coeffs(pyramid, slices, output_format='wavedec2'))

def add(w1, w2, safe=False):
    """Add two wavelets element-wise (band-by-band).

    Args:
        w1 (Wavelet): a wavelet to add.
        w2 (Wavelet): another wavelet to add.
        safe (bool): whether to check if the computation is legal.

    Returns:
        result (Wavelet): the sum of w1 and w2 element-wise.
    """
    pyramid_1, slices_1 = w1.pyramid_forward(get_slices=True, to_tensor=False)
    pyramid_2, slices_2 = w2.pyramid_forward(get_slices=True, to_tensor=False)
    if safe and slices_1 != slices_2:
        raise RuntimeError('utils.wavelet.add: levels of the wavelets are different.')
    pyramid_sum = pyramid_1 + pyramid_2
    return pyramid_backward(pyramid_sum, slices_1)

def sub(w1, w2, safe=False):
    """Subtract two wavelets element-wise (band-by-band).

    Args:
        w1 (Wavelet): a wavelet to be subtracted.
        w2 (Wavelet): a wavelet to subtract.
        safe (bool): whether to check if the computation is legal.

    Returns:
        result (Wavelet): the result of w1 - w2 element-wise.
    """
    pyramid_1, slices_1 = w1.pyramid_forward(get_slices=True, to_tensor=False)
    pyramid_2, slices_2 = w2.pyramid_forward(get_slices=True, to_tensor=False)
    if safe and slices_1 != slices_2:
        raise RuntimeError('utils.wavelet.add: levels of the wavelets are different.')
    pyramid_sum = pyramid_1 - pyramid_2
    return pyramid_backward(pyramid_sum, slices_1)

def mul_subbandwise(wavelet, scalars):
    """Multiply each subband of a wavelet by a scalar.

    Args:
        wavelet (Wavelet): a wavelet.
        scalars (list of float): list of scalars to multiply the wavelet. The length of scalars
            must match the number of wavelet subbands.

    Returns:
        result (Wavelet): scalars * wavelet subband-wise.
    """
    result = wavelet.copy()
    result.coeff[0] *= scalars[0]
    idx = 1
    for i in range(1, len(result.coeff)):
        for j in range(len(result.coeff[i])):
            result.coeff[i][j] *= scalars[idx]
            idx += 1
    return result

def add_noise_subbandwise(wavelet, stds, is_complex=False):
    """Add Gaussian noise to each wavelet subband.

    Args:
        wavelet (Wavelet): a wavelet.
        stds (list of float): list of Gaussian noise standard deviation in each subband.
        is_complex (bool): whether the noise to add is complex

    Returns:
        result (Wavelet): noisy wavelet.

    Note:
        When is_complex is True, for a standard deviation sigma of a subband, the noise is added independently
        to the real and imaginary part in that subband, each part with standard deviation sigma / sqrt(2).
    """
    result = wavelet.copy()
    result.coeff[0] += gutil.generate_noise(result.coeff[0].shape, stds[0], ret_array=True, is_complex=is_complex)
    idx = 1
    for i in range(1, len(result.coeff)):
        for j in range(len(result.coeff[i])):
            result.coeff[i][j] += gutil.generate_noise(result.coeff[i][j].shape, stds[idx], ret_array=True, is_complex=is_complex)
            idx += 1
    return result

class Wavelet:
    """Wrapper of the list of wavelet coefficients.

    To generate a Wavelet object by a wavelet transform, use the forward function.
    """
    def __init__(self, coeff):
        self.coeff = coeff
        for i in range(1, len(self.coeff)):
            self.coeff[i] = list(coeff[i])

    def get_bands(self):
        """Return level of decomposition"""
        return len(self.coeff) - 1

    def get_subbands(self):
        """Return total number of wavelet subbands, generally 3 * level + 1"""
        subbands = 1
        for i in range(1, len(self.coeff)):
            subbands += len(self.coeff[i])
        return subbands

    def count_subbandwise(self):
        """Count number of coefficients in each subband"""
        result = np.zeros(self.get_subbands(), dtype=int)
        result[0] = np.prod(self.coeff[0].shape)
        idx = 1
        for i in range(self.get_bands()):
            for j in range(len(self.coeff[i + 1])):
                result[idx] += np.prod(self.coeff[i + 1][j].shape)
                idx += 1
        return result

    def inverse(self, wavelet='haar', to_tensor=True):
        """Inverse wavelet transform"""
        self._subband_to_tuple()
        if to_tensor:
            return torch.from_numpy(pywt.waverec2(self.coeff, wavelet))
        else:
            return pywt.waverec2(self.coeff, wavelet)

    def pyramid_forward(self, get_slices=False, to_tensor=True):
        """Build wavelet pyramid i.e. place wavelet coefficients into an array

        Note:
            If you want to be able to recover the wavelet from a pyramid, choose to
            return the slices as well since it is needed for pyramid_backward.
            Note that the slices depend only on the image shape.
        """
        self._subband_to_tuple()
        pyramid, slices = pywt.coeffs_to_array(self.coeff, axes=(-2, -1))
        if to_tensor:
            pyramid = torch.from_numpy(pyramid)
        if get_slices:
            return pyramid, slices
        else:
            return pyramid

    def real(self):
        """Take real part of the wavelet coefficients"""
        pyramid, slices = self.pyramid_forward(get_slices=True, to_tensor=False)
        pyramid = np.real(pyramid)
        return pyramid_backward(pyramid, slices)

    def imag(self):
        """Take imaginary part of the wavelet coefficients"""
        pyramid, slices = self.pyramid_forward(get_slices=True, to_tensor=False)
        pyramid = np.imag(pyramid)
        return pyramid_backward(pyramid, slices)

    def copy(self):
        """Deep copy self"""
        return Wavelet(deepcopy(self.coeff))

    def astype(self, type):
        """Cast the wavelet coefficients"""
        pyramid, slices = self.pyramid_forward(get_slices=True, to_tensor=False)
        pyramid = pyramid.astype(type)
        return pyramid_backward(pyramid, slices)

    def _subband_to_tuple(self):
        for i in range(1, len(self.coeff)):
            self.coeff[i] = tuple(self.coeff[i])

"""Wavelet object as defined in pywt"""
Wavelet_bank = pywt.Wavelet

"""Name of wavelet subbands and corresponding default color code for plotting"""
SUBBAND = ['A', 'H', 'W', 'D']
SUBBAND_COLOR = ['g', 'b', 'r', 'm']