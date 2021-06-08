"""Main D-VDAMP experiment.

Note:
    Also generate MSE/SURE heatmaps if savemode is not recon-only.
"""

import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import torch
from algorithm import dvdamp
from algorithm import simulation as sim
from algorithm import heatmap
from util import general as gutil
from util import plot as putil
from util import transform as tutil

print('Parsing arguments...')
parser = argparse.ArgumentParser(description='DVDAMP experiments.')

# Common simulation arguments
parser.add_argument('--datadir', type=str,
                    default='data/mri', help='Path image directory.')
parser.add_argument('--savedir', type=str,
                    default='result/dvdamp', help='Path to save the results.')
parser.add_argument('--modeldir', type=str,
                    default='data/models',
                    help='Path to the directory containing CNN denoiser models.')
parser.add_argument('--oneim', action='store_true',
                    help='Whether to use only the first image from directory for experiment.')
parser.add_argument('--verbose', action='store_true',
                    help='Whether to print a lot.')

# Measurement simulation arguments
parser.add_argument('--samplingrate', type=float,
                    default=0.125, help='Sampling rate m/n for compressive sensing measurement.')
parser.add_argument('--snr', type=float,
                    default=40, help='Signal to noise ratio of the measurement.')

# DVDAMP arguments
parser.add_argument('--dentype', type=str, choices=['soft', 'bm3d', 'cdncnn'],
                    default='cdncnn', help='Denoiser type.')
parser.add_argument('--wavetype', type=str,
                    default='haar', help='Type of wavelet used in DVDAMP.')
parser.add_argument('--level', type=int, default=4,
                    help='Number of wavelet scales.')
parser.add_argument('--iters', type=int, default=10,
                    help='Maximum number of iterations in DVDAMP.')
parser.add_argument('--stop_on_increase', action='store_true',
                    help='Whether to stop DVDAMP before maximum number of iterations when MSE estimate increases.')

# Saving arguments
parser.add_argument('--selected_it', type=int, default=[1, 5], nargs='*',
                    help='Selected iterations to plot QQ plot and effective noise.')
parser.add_argument('--selected_scale', type=int, default=[0, 2], nargs='*',
                    help='Selected scale of wavelet to plot QQ plot (smaller number means smaller scale).')

# SURE map arguments
parser.add_argument('--savemode', type=str, choices=['full', 'raw', 'plot', 'recon-only'],
                    default='full', help='full: save raw outputs and plots; raw: save only raw outputs; plot: save only plots; recon-only: no heatmap generation.')
parser.add_argument('--windows', type=int, default=[32], nargs='*',
                    help='Patch sizes.')
parser.add_argument('--stride', type=int, default=1,
                    help='Stride for patch extraction/reconstruction.')
parser.add_argument('--numnoises', type=int, default=[2], nargs='*',
                    help='Numbers of noise used to calculate divergence with MC-SURE.')
parser.add_argument('--errorlog', action='store_true',
                    help='Whether to save mean squared error of SURE with respect to MSE.')

args = parser.parse_args()
datadir = args.datadir
savedir = args.savedir
modeldir = args.modeldir
oneim = args.oneim
verbose = args.verbose
sampling_rate = args.samplingrate
snr = args.snr
dentype = args.dentype
wavetype = args.wavetype
level = args.level
iters = args.iters
stop_on_increase = args.stop_on_increase
selected_it = args.selected_it
selected_scale = args.selected_scale
savemode = args.savemode

# Constants (Colored DnCNN model parameters)
modelnames = ['cdncnn_00_20_real', 'cdncnn_20_50_real', 'cdncnn_50_120_real', 'cdncnn_120_500_real']
std_ranges = np.array([0, 20, 50, 120, 500]) / 255

def main():

    print('---VDAMP Simulation---')
    gutil.mkdir_if_not_exists(savedir)
    gutil.print_arguments(args)
    gutil.log_arguments(args, os.path.join(savedir, 'args.txt'))
    loadlist, namelist, num_images = gutil.prepare_image_path(datadir, oneim=oneim)

    # Setup denoiser
    if dentype == 'soft':
        denoiser = dvdamp.SoftThresholding_VDAMP(MC_divergence=True, debug=True)
    elif dentype == 'bm3d':
        denoiser = dvdamp.BM3D_VDAMP(channels=1)
    elif dentype == 'cdncnn':
        modeldirs = [None] * len(modelnames)
        std_channels = 3 * level + 1
        for i, modelname in enumerate(modelnames):
            modeldirs[i] = os.path.join(modeldir, '{}.pth'.format(modelname))
        denoiser = dvdamp.ColoredDnCNN_VDAMP(modeldirs, std_ranges, std_channels=std_channels, verbose=verbose)

    # Setup log variable
    psnr_log = torch.zeros(num_images)
    if args.errorlog and savemode != 'recon-only':
        error_log = torch.zeros(num_images, len(args.windows), len(args.numnoises))

    for i, (loadname, name) in enumerate(zip(loadlist, namelist)):

        print('{}. image: {}'.format(i + 1, name))
        image = gutil.read_image(loadname)
        this_savedir = os.path.join(savedir, name)

        # MRI sampling simulation and VDAMP
        recon_image, log, true_iters = sim.dvdamp_sim(image, sampling_rate, snr, denoiser, iters, 
                                                level=level, wavetype=wavetype, stop_on_increase=stop_on_increase)
        torch.save(log, os.path.join(savedir, 'log.pt'))
        psnr_log[i] = gutil.calc_psnr(recon_image.real, image)
        print('PSNR = {:.3f}'.format(psnr_log[i]))

        # Saving results and statistics
        error_map = (recon_image - image).abs()
        gutil.save_image(recon_image.real, '{}-recon.png'.format(this_savedir))
        putil.save_heatmap(error_map, '{}-recon_err'.format(this_savedir), savemode='plot')
        w0 = tutil.forward(image[0], wavelet=wavetype, level=level)
        plot_psnr(log, image, this_savedir, true_iters)
        plot_effective_noise(log, w0, this_savedir, true_iters)
        plot_qq(log, w0, this_savedir, true_iters)
        plot_err(log, w0, this_savedir, true_iters)

        if savemode != 'recon-only':
            print('Begin generating MSE and SURE heatmaps...')

            print('true_iters', true_iters)

            # Get ingredients for SURE map
            noisy_wavelet = tutil.pyramid_backward(log['r'][true_iters - 1], log['slices'])
            denoised_wavelet = tutil.pyramid_backward(log['w_hat'][true_iters - 1], log['slices'])
            denoised_image = denoised_wavelet.inverse(to_tensor=True).unsqueeze(0)
            tau = tutil.reformat_subband2array(log['tau'][true_iters - 1])
            gutil.save_image(denoised_image.real, '{}-recon-no-final-correction.png'.format(this_savedir))

            for iwin, window in enumerate(args.windows):

                mse = heatmap.calc_mse(image, recon_image, window, stride=args.stride)

                for inoi, num_noise in enumerate(args.numnoises):

                    print('patch size = {:d}, k = {:d}'.format(window, num_noise))

                    sure = heatmap.calc_sure_vdamp(noisy_wavelet, denoised_wavelet, denoised_image, denoiser, tau,
                                                    window, stride=args.stride, num_noise=num_noise)

                    vmin, vmax = putil.get_heatmap_limits([mse, sure])

                    # Note: even though k does not affect mse calculation, we save mse for each k anyway so that
                    # the scaling (color bar) of the MSE heatmap matches the corresponding SURE heatmap.
                    putil.save_heatmap(mse, os.path.join(savedir, '{}-w{:d}-k{:d}-mse'.format(name, window, num_noise)),
                                        savemode=savemode, title='MSE, patch size = {:d}, k = {:d}'.format(window, num_noise),
                                        vmin=vmin, vmax=vmax)
                    putil.save_heatmap(sure, os.path.join(savedir, '{}-w{:d}-k{:d}-sure'.format(name, window, num_noise)),
                                        savemode=savemode, title='SURE, patch size = {:d}, k = {:d}'.format(window, num_noise),
                                        vmin=vmin, vmax=vmax)

                    if args.errorlog:
                        error_log[i, iwin, inoi] = ((mse - sure) ** 2).mean().item()

    print('Saving log file(s)...')
    torch.save(psnr_log, os.path.join(savedir, 'psnr_log.pt'))
    with open(os.path.join(savedir, 'psnr_mean.txt'), 'w') as f:
        f.write('{}'.format(psnr_log.mean().item()))
    if args.errorlog and savemode != 'recon-only':
            torch.save(error_log, os.path.join(savedir, 'error.pt'))
            putil.save_sqerror_plot(error_log, args.windows, args.numnoises, os.path.join(savedir, 'error.png'))

def plot_psnr(log, image, savedir, true_iters):
    """Plot PSNR history"""
    psnr = np.zeros(true_iters)
    for i in range(true_iters):
        x_hat = gutil.im_numpy_to_tensor(log['x_hat'][i].real)
        psnr[i] = gutil.calc_psnr(x_hat, image)
    plt.plot(psnr)
    plt.xlabel('Iteration')
    plt.ylabel('PSNR')
    plt.title('PSNR of Reconstructed Image (x_hat)')
    plt.savefig('{}-psnr.png'.format(savedir), bbox_inches='tight')
    plt.clf()

def plot_effective_noise(log, w0, savedir, true_iters):
    """Plot pyramid of effective noise in wavelet coefficients"""
    w0_pyramid = w0.pyramid_forward(to_tensor=False)
    H, W = w0_pyramid.shape
    eff_noise = np.zeros((len(selected_it), H, W))
    for i, it in enumerate(selected_it):
        if it >= true_iters:
            break
        eff_noise[i] = np.abs(log['r'][it] - w0_pyramid)
    vmin = np.min(eff_noise)
    vmax = np.max(eff_noise)

    for i, it in enumerate(selected_it):
        if it >= true_iters:
            break
        putil.save_heatmap(eff_noise[i], '{}-eff_noise-{}'.format(savedir, it),
                            title='|w_hat - r|, iteration {}'.format(it),
                            savemode='plot', vmin=vmin, vmax=vmax, tensor=False)

def plot_qq(log, w0, savedir, true_iters):
    """QQ plot of effective noise in wavelet coefficients"""
    for i, it in enumerate(selected_it):
        if it >= true_iters:
            break
        fig, ax = plt.subplots(len(selected_scale), 3, figsize=(8 * len(selected_scale), 5 * 3))
        _, slices = w0.pyramid_forward(get_slices=True, to_tensor=False)
        r = tutil.pyramid_backward(log['r'][it], slices)
        for i_b, b in enumerate(selected_scale):
            for i_s, s in enumerate(tutil.SUBBAND[1:]):
                eff_noise = r.coeff[b + 1][i_s] - w0.coeff[b + 1][i_s]
                eff_noise = (eff_noise - np.mean(eff_noise)) / np.std(eff_noise)
                sm.qqplot(eff_noise.reshape(-1).real, line='45', ax=ax[i_b, i_s])
                ax[i_b, i_s].set_title('{}, level {}'.format(s, b))
        fig.suptitle('QQ plot of effective noise at iteration {}'.format(it))
        fig.savefig('{}-qq-{}.png'.format(savedir, it))
        fig.clear()
        plt.close(fig)

def plot_err(log, w0, savedir, true_iters):
    """Plot predicted and true normalized mean squared erro"""
    norms, sizes = _calc_wavelet_coeff_norm_and_count(w0)
    fig, ax = plt.subplots(1, level, figsize=(20, 10))
    err = _band_iter_reformat(log['err'], true_iters)
    tau = _band_iter_reformat(log['tau'], true_iters)
    ax[0].plot(10 * np.log10(sizes[0] * err[0] / (norms[0] ** 2)),
                ls='-', c=tutil.SUBBAND_COLOR[0],
                label='{} True'.format(tutil.SUBBAND[0]))
    ax[0].plot(10 * np.log10(sizes[0] * tau[0] / (norms[0] ** 2)),
                ls=':', c=tutil.SUBBAND_COLOR[0],
                label='{} Model'.format(tutil.SUBBAND[0]))
    for b in range(level):
        for s in range(3):
            ax[b].plot(10 * np.log10(sizes[b + 1][s] * err[b + 1][s] / (norms[b + 1][s] ** 2)),
                    ls='-', c=tutil.SUBBAND_COLOR[s + 1],
                    label='{} True'.format(tutil.SUBBAND[s + 1]))
            ax[b].plot(10 * np.log10(sizes[b + 1][s] * tau[b + 1][s] / (norms[b + 1][s] ** 2)),
                    ls=':', c=tutil.SUBBAND_COLOR[s + 1],
                    label='{} Model'.format(tutil.SUBBAND[s + 1]))
        ax[b].set_xlabel('Iteration')
        ax[b].set_ylabel('NMSE (dB)')
        ax[b].set_title('Level {}'.format(b + 1))
    fig.suptitle('True and Estimated NMSE of r')
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels)
    fig.savefig('{}-err_est.png'.format(savedir))
    fig.clear()
    plt.close(fig)

def _calc_wavelet_coeff_norm_and_count(wavelet):
    level = wavelet.get_bands()
    norms = [None] * (level + 1)
    sizes = [None] * (level + 1)
    norms[0] = np.linalg.norm(wavelet.coeff[0])
    sizes[0] = np.prod(wavelet.coeff[0].shape)
    for b in range(level):
        norms_band = [None] * 3
        sizes_band = [None] * 3
        for s in range(3):
            norms_band[s] = np.linalg.norm(wavelet.coeff[b + 1][s])
            sizes_band[s] = np.prod(wavelet.coeff[b + 1][s].shape)
        norms[b + 1] = norms_band
        sizes[b + 1] = sizes_band
    return norms, sizes

def _band_iter_reformat(target, true_iters):
    result = [None] * len(target[0])
    result[0] = np.zeros(true_iters)
    for i in range(true_iters):
        result[0][i] = target[i][0]
    for b in range(level):
        result_band = [None] * 3
        for s in range(3):
            result_subband = np.zeros(true_iters)
            for i in range(true_iters):
                result_subband[i] = target[i][b + 1][s]
            result_band[s] = result_subband
        result[b + 1] = result_band
    return result

if __name__ == '__main__':
    main()