"""Utility functions for plotting

    * plot_heatmap
    * save_heatmap
    * get_heatmap_limits
    * save_sqerror_plot
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

FONT = {'fontname' : 'Serif',
        'size' : 14}

def _fmt(x, pos):
    """Formatting for plot_heatmap function"""
    if x == 0:
        return '0'
    else:
        return format(x, '1.0e').replace("e-0", "e-")

def plot_heatmap(input, vmin=None, vmax=None, tensor=True, cmap='inferno', add_cbar=True, cbarfsize=12):
    """Plot heatmap of tensor with colorbar"""
    if tensor:
        input = input.numpy()[0]
    plt.imshow(input, vmin=vmin, vmax=vmax, cmap=cmap)
    if add_cbar:
        cbar = plt.colorbar(shrink=0.8, format=ticker.FuncFormatter(_fmt))
        cbar.ax.tick_params(labelsize=cbarfsize)
    plt.axis('off')

def save_heatmap(input, path, savemode='full', title=None,
                 vmin=None, vmax=None, tensor=True, cmap='inferno', add_cbar=True,
                 font=FONT, adjust=None):
    assert savemode in ['full', 'raw', 'plot']
    if savemode in ['full', 'plot']:
        plot_heatmap(input, vmin=vmin, vmax=vmax, tensor=tensor, cmap=cmap, add_cbar=add_cbar)
        if title is not None:
            plt.title(title, **font)
        if adjust is None:
            plt.tight_layout()
        else:
            plt.subplots_adjust(bottom=adjust[0], top=adjust[1], left=adjust[2], right=adjust[3])
        plt.savefig('{}.png'.format(path))
        plt.clf()
    if savemode in ['full', 'raw']:
        torch.save(input, '{}.pt'.format(path))

def get_heatmap_limits(input_list):
    vmin = np.infty
    vmax = -np.infty
    for input in input_list:
        if vmin > input.min().item():
            vmin = input.min().item()
        if vmax < input.max().item():
            vmax = input.max().item()
    return vmin, vmax

def save_sqerror_plot(error, windows, num_noises, path, font={}):
    for i, num_noise in enumerate(num_noises):
        plt.plot(windows, error.mean(dim=0)[:, i], label='k = {:d}'.format(num_noise))
    plt.xlabel('patch size', **font)
    plt.ylabel('$|MSE - SURE|^2$', **font)
    plt.title('Mean squared error of SURE heatmap', **font)
    plt.legend()
    plt.savefig(path, bbox_inches='tight')
    plt.clf()
