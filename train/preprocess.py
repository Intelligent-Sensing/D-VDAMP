"""Functions for preprocessing and saving image datasets as .h5 files.
Assume that the dataset is organized as
    - folder
        - image 1
        - image 2
        - ...
where each image is named as [prefix, number, suffix] e.g. train_01.png.
Load images and generate patches as specified. Then, store the patches in .h5
with the following organization,
    - .h5 file
        - key 1 : patch 1
        - key 2 : patch 2
        - ...
"""

import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
import h5py
from util.general import normalize, read_image, generate_loadlist, remove_if_exists

def generate_datasets(rootdir, num_train=None, num_val=None, num_test=None, train_window=40):
    """Generate train and test datasets.
    Args:
        rootdir (str): path to the data directory
        num_train (int): number of images to be read for train set.
        num_val (int): number of images to be read for validation set.
        num_test (int): number of images to be read for test set.
    """
    # Generate train dataset
    loadlist = generate_loadlist(os.path.join(rootdir, 'train'), num_files=num_train, suffix='.jpg')
    savename = os.path.join(rootdir, 'train.h5')
    remove_if_exists(savename)
    generate_h5(loadlist, savename, train_window, 10)

    # Generate validation dataset (Set12)
    loadlist = generate_loadlist(os.path.join(rootdir, 'val'), num_files=num_val, suffix='.jpg')
    savename = os.path.join(rootdir, 'val.h5')
    remove_if_exists(savename)
    generate_h5(loadlist, savename, None, None)

    # Generate test dataset (Set68)
    loadlist = generate_loadlist(os.path.join(rootdir, 'test'), num_files=num_test, suffix='.jpg')
    savename = os.path.join(rootdir, 'test.h5')
    remove_if_exists(savename)
    generate_h5(loadlist, savename, None, None)
    
def generate_h5(loadlist, savename, window, stride):
    """Generate a .h5 file from patches of specified images.
    Note:
        If window or stride are None, store images to the .h5 file without
        extracting patches.
    Args:
        loadlist (list of str): list paths to images.
        savename (str): name of the dataset file to save (must ends with .h5).
        window (int): window size to extract patches.
        stride (int): stride to extract patches.
    """
    h5f = h5py.File(savename, 'w')
    idx = 0
    for loadname in loadlist:
        image = read_image(loadname)
        if window is None or stride is None:
            patches = np.expand_dims(image, 0)
        else:
            patches = image_to_patches(image, window, stride=stride)
        for i in range(patches.shape[0]):
            h5f.create_dataset(str(idx), data=patches[i])
            idx += 1
    h5f.close()

def image_to_patches(image, window, stride=1):
    """Generate patches of images.
    Args:
        image (np.ndarray): the image with dimensions (C, H, W).
        window (int): height and width of a patch.
        stride (int): stride across pixels to extract patches.
    Returns:
        patches (np.ndarray): the resulting patches of dimensions (N, C, H, W).
    """
    C, H, W = image.shape
    numPatchesH = (H - window) // stride + 1
    numPatchesW = (W - window) // stride + 1
    numPatches = numPatchesH * numPatchesW
    patches = np.zeros([numPatches, C, window, window])
    idx = 0
    for kh in range(numPatchesH):
        for kw in range(numPatchesW):
            patches[idx, :, :, :] = image[:, kh * stride : kh * stride + window,
                                          kw * stride : kw * stride + window]
            idx += 1
    return patches