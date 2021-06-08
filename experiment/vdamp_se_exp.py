"""Simulate VDAMP state evolution

Note:
    Also generate MSE/SURE heatmaps if savemode is not recon-only.
"""

import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import argparse
import numpy as np
import torch
from algorithm import dvdamp
from algorithm import simulation as sim
from algorithm import heatmap
from util import general as gutil
from util import plot as putil

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

# DVDAMP arguments
parser.add_argument('--dentype', type=str, choices=['soft', 'bm3d', 'cdncnn'],
                    default='cdncnn', help='Denoiser type.')
parser.add_argument('--wavetype', type=str,
                    default='haar', help='Type of wavelet used in DVDAMP.')
parser.add_argument('--level', type=int, default=4,
                    help='Number of wavelet scales.')

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
dentype = args.dentype
wavetype = args.wavetype
level = args.level
savemode = args.savemode

# Constants (Colored DnCNN model parameters)
modelnames = ['cdncnn_00_20_real', 'cdncnn_20_50_real', 'cdncnn_50_120_real', 'cdncnn_120_500_real']
std_ranges = np.array([0, 20, 50, 120, 500]) / 255

# Variance value for each subband. Numbers taken from applying D-VDAMP in a MRI simulation.
tau = np.array([0.00027460222492213646, 
        0.00026393712581617996, 0.0004002784838955675, 0.0002745781392178108, 
        0.0004484239278547852, 0.0008156956564901096, 0.00044903024479627754, 
        0.001897407047245574, 0.0012666475232024125, 0.0005994850471120772, 
        0.0008184462495338877, 0.0007362659053241197, 0.00031610034826961707])

print('noise std', np.sqrt(tau) * 255)

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

        denoised_image, noisy_wavelet, denoised_wavelet = sim.vdamp_se_sim(image, tau, denoiser)
        psnr_log[i] = gutil.calc_psnr(denoised_image.real, image)
        print('PSNR = {:.3f}'.format(psnr_log[i]))

        # Saving results and statistics
        error_map = (denoised_image - image).abs()
        gutil.save_image(denoised_image.real, '{}-recon.png'.format(this_savedir))
        putil.save_heatmap(error_map, '{}-recon_err'.format(this_savedir), savemode='plot')

        if savemode != 'recon-only':
            print('Begin generating MSE and SURE heatmaps...')

            for iwin, window in enumerate(args.windows):

                mse = heatmap.calc_mse(image, denoised_image, window, stride=args.stride)

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

if __name__ == '__main__':
    main()