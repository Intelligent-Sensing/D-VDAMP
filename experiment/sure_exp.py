"""Main SURE map experiment.
    * MSE/SURE heatmaps for denoising problem
    * MSE/SURE heatmaps for compressive sensing problem with D-AMP solver
"""

import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import argparse
import numpy as np
import torch
from algorithm import simulation as sim
from algorithm import heatmap
from algorithm import denoiser as den
from algorithm import csalgo
from util import general as gutil
from util import plot as putil
from util import cs as cutil

print('Argument parsing...')
parser = argparse.ArgumentParser(description='Uncertainty heatmap experiments.')

# Common simulation arguments
parser.add_argument('mode', type=str, choices=['den', 'cs'], default='den',
                    help='Action: simulate denoising or compressive sensing and generate uncertainty heatmaps.')
parser.add_argument('--datadir', type=str,
                    default='data/mini', help='Path image directory.')
parser.add_argument('--savedir', type=str,
                    default='results/mini', help='Path to save the results.')
parser.add_argument('--modeldir', type=str,
                    default='data/models',
                    help='Path to saved DnCNN model, or path to the collection of DnCNN models, or bm3d to use BM3D')
parser.add_argument('--std', type=float,
                    default=25, help='Standard deviation of Gaussian noise on [0, 255] scale.')
parser.add_argument('--gpu', action='store_true',
                    help='Whether to use GPU for computations when applicable.')
parser.add_argument('--oneim', action='store_true',
                    help='Whether to use only the first image from directory for experiment.')
parser.add_argument('--verbose', action='store_true',
                    help='Whether to print a lot.')      

# Arguments for compressive sensing
parser.add_argument('--samplingrate', type=float,
                    default=0.2, help='Sampling rate m/n for compressive sensing measurement.')
parser.add_argument('--iters', type=int, default=10,
                    help='Number of iterations for compressive sensing solver.')
parser.add_argument('--csmode', type=str, choices=['matmul', 'jl', 'jl-legacy'], default='matmul',
                    help='Transformation method for compressive sensing (plain matrix multiplication or JL transform).')

# Arguments for generating SURE map plots
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
mode = args.mode
datadir = args.datadir
savedir = args.savedir
modeldir = args.modeldir
std = args.std / 255.
gpu = args.gpu
oneim = args.oneim
savemode = args.savemode

if __name__ == '__main__':

    print('Setting up device...')
    dtype = torch.float32
    if gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device: {}'.format(device))

    print('Setting up denoiser...')
    if modeldir == 'bm3d':
        print('Denoiser: BM3D')
        denoiser = den.BM3D_denoiser()
    elif modeldir.endswith('.pth'):
        if mode == 'cs':
            print('Warning: DAMP requires a denoiser applicalbe to multiple noise levels')
        print('Denoiser: DnCNN from saved model: {}'.format(modeldir))
        model = den.setup_DnCNN(modeldir, num_layers=20, device=device)
        denoiser = den.DnCNN_denoiser(model)
    else:
        # Our standard collection of DnCNN models
        print('Denoiser: DnCNN Ensemble')
        std_ranges = np.array([0, 10, 20, 40, 60, 80, 100, 150, 300]) / 255
        modelnames = ['b-0-10', 'b-10-20', 'b-20-40', 'b-40-60',
                      'b-60-80', 'b-80-100', 'b-100-150', 'b-150-300']
        models = den.setup_DnCNN_ensemble(modeldir, modelnames, num_layers=20, device=device)
        denoiser = den.DnCNN_ensemble_denoiser(models, std_ranges, device=device, verbose=True)

    print('Preparing image paths...')
    loadlist, namelist, num_images = gutil.prepare_image_path(datadir, oneim=oneim)

    # Preparing log variables
    if mode == 'den':
        psnr_log = torch.zeros(num_images)
    else: # mode == 'cs'
        psnr_log = torch.zeros(num_images, args.iters)
        noise_log = torch.zeros(num_images)
    if args.errorlog and savemode != 'recon-only':
        error_log = torch.zeros(num_images, len(args.windows), len(args.numnoises))

    gutil.mkdir_if_not_exists(savedir)

    with torch.no_grad():
        for i, (loadname, name) in enumerate(zip(loadlist, namelist)):

            print('{}. image: {}'.format(i, name))
            image = gutil.read_image(loadname)

            if mode == 'den':

                print('Simulating a denoising problem...')
                recon_image, noisy_image = sim.denoise_sim(image, std, denoiser)
                psnr_log[i] = gutil.calc_psnr(recon_image, image)
                print('Denoised Image PSNR = {:.3f}'.format(psnr_log[i].item()))
                gutil.save_image(recon_image, os.path.join(savedir, '{}-denoised.png'.format(name)))
                gutil.save_image(noisy_image, os.path.join(savedir, '{}-noisy.png'.format(name)))

            else: # mode == 'cs'

                print('Simulating a compressive sensing problem...')
                m, n = cutil.get_cs_param(image.shape, args.samplingrate)
                cs_transform = cutil.CStransform(m, n, mode=args.csmode)
                cs_algo = csalgo.DAMP(image.shape, denoiser, args.iters, image=image)
                recon_image, noisy_image, std_est, psnr = sim.cs_sim(image, cs_transform, std, cs_algo)
                psnr_log[i, :] = psnr
                noise_log[i] = std_est
                print('Estimated noise std = {:.3e}'.format(std_est))
                print('Reconstructed Image PSNR = {:.3f}'.format(psnr_log[i, -1].item()))
                gutil.save_image(recon_image, os.path.join(savedir, '{}-recon.png'.format(name)))

            error_map = (recon_image - image).abs() ** 2

            if savemode != 'recon-only':
                print('Begin generating MSE and SURE heatmaps...')

                for iwin, window in enumerate(args.windows):

                    mse = heatmap.calc_mse(image, recon_image, window, stride=args.stride)

                    for inoi, num_noise in enumerate(args.numnoises):

                        print('patch size = {:d}, k = {:d}'.format(window, num_noise))

                        if mode == 'den':
                            std_for_sure = std
                        else: # mode == 'cs'
                            std_for_sure = std_est

                        sure = heatmap.calc_sure(noisy_image, recon_image, denoiser, std_for_sure,
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
                        putil.save_heatmap(error_map, os.path.join(savedir, '{}-w{:d}-k{:d}-err'.format(name, window, num_noise)),
                                            savemode=savemode, title='Squared error'.format(window, num_noise),
                                            vmin=vmin, vmax=vmax)

                        if args.errorlog:
                            error_log[i, iwin, inoi] = ((mse - sure) ** 2).mean().item()

    print('Saving log file(s)...')
    torch.save(psnr_log, os.path.join(savedir, 'psnr.pt'))
    if mode == 'cs':
        torch.save(noise_log, os.path.join(savedir, 'noise_std_est.pt'))
    if args.errorlog and savemode != 'recon-only':
            torch.save(error_log, os.path.join(savedir, 'error.pt'))
            putil.save_sqerror_plot(error_log, args.windows, args.numnoises, os.path.join(savedir, 'error.png'))

    print('experiments.sure-exp: Done!')
