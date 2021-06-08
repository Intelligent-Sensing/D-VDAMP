"""General utility functions

    Tensor manipulation
        * dtype
        * to_tensor
        * to_PIL
        * normalize
        * clamp
        * im_numpy_to_tensor

    Noise-related functions
        * calc_psnr
        * generate_noise
        * add_noise

    File IO
        * save_image (from torchvision.utils)
        * save_cpx_image
        * read_image
        * prepare_image_path
        * generate_loadlist
        * generate_namelist
        * remove_if_exists
        * mkdir_if_not_exists

    Argument parsing
        * print_arguments
        * log_arguments

    Neural-network-related functions
        * load_checkpoint
"""

import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.nn.functional import mse_loss
from PIL import Image

"""Default data type"""
dtype = torch.float32

"""Transform PIL image to torch.Tensor (C, H, W)"""
to_tensor = transforms.ToTensor()

"""Transform torch.Tensor (C, H, W) to PIL Image"""
to_PIL = transforms.ToPILImage()

def normalize(input):
    """Normalize pixel values from [0, 255] to [0., 1.]"""
    return input / 255.

def clamp(image, min=0., max=1.):
    """Clamp values in input tensor exceeding (min, max) to (min, max)"""
    return torch.clamp(image, min, max)

def im_numpy_to_tensor(image):
    """Reformat (H, W) np.ndarray to (1, H, W) tensor"""
    return torch.from_numpy(image).unsqueeze(0)

def calc_psnr(test_image, target_image, max=1.):
    """Calculate PSNR of images."""
    mse = mse_loss(test_image, target_image)
    return 20 * torch.log10(max / torch.sqrt(mse)).item()

def generate_noise(size, std, ret_array=False, is_complex=False):
    """Generate zero-mean white Gaussian noise."""
    if is_complex:
        # Real part and imaginary part are independent. Each has variance = (std ** 2) / 2.
        noise = torch.normal(mean=torch.zeros(*size, dtype=torch.complex64), std=std)
    else:
        noise = torch.normal(mean=torch.zeros(*size), std=std)
    if ret_array:
        return noise.numpy()
    else:
        return noise

def add_noise(image, std):
    """Add zero-mean white Gaussian noise to image"""
    noise = generate_noise(image.shape, std)
    return image + noise

def save_cpx_image(image, name):
    """Save a complex image (represented by real and imaginary channels) as two images."""
    save_image(image[0], '{}_real.png'.format(name))
    save_image(image[1], '{}_imag.png'.format(name))

def read_image(path, rgb=False, scale=1.0):
    """Read image from path using PIL.Image

    Args:
        path (str): path to the image file.
        rgb (bool): whether to read image as rgb. Otherwise, read image as grayscale.
        scale (float): resize the image by the scale.

    Returns:
        image (torch.Tensor): image Tensor in (C, H, W) shape.
    """
    image = Image.open(path)
    if rgb:
        image = image.convert('rgb')
    else:
        image = image.convert('L')
    if scale != 1.0:
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size, Image.ANTIALIAS)
    return to_tensor(image)

def prepare_image_path(datadir, oneim=False):
    """Generate loadlist and namelist"""
    loadlist = generate_loadlist(datadir)
    namelist = generate_namelist(datadir, no_exten=True)
    if oneim:
        loadlist = [loadlist[0]]
        namelist = [namelist[0]]
    num_images = len(loadlist)
    return loadlist, namelist, num_images

def generate_loadlist(datadir, prefix=None, suffix=None, num_files=None):
    """Generate list of paths to images"""
    namelist = generate_namelist(datadir, prefix=prefix, suffix=suffix, num_files=num_files)
    if num_files is None or len(namelist) < num_files:
        num_files = len(namelist)
    loadlist = [None] * num_files
    for i, name in enumerate(namelist):
        loadlist[i] = os.path.join(datadir, name)
    return loadlist

def generate_namelist(datadir, num_files=None, prefix=None, suffix=None, no_exten=False, no_hidden=True):
    """Generate list of file names in a directory
    
    Args:
        datadir (std): path to directory containing files.
        num_files (int): number of files to read. If a number is given, return first num_files names by 
            lexicographical order. If None, read all files satisfying other criteria (prefix, etc.).
        prefix (str): return only file names beginning with this prefix.
        suffix (str): return only file names (including the extension) ending with this suffix.
        no_exten (bool): whether to include the extension in the returning file names.
        no_hidden (bool): whether to include hidden files.

    Returns:
        namelist (list): list of file names.
    """
    raw_list = sorted(os.listdir(datadir))
    if prefix is not None:
        prefix_filtered_list = []
        for name in raw_list:
            if name.startswith(prefix):
                prefix_filtered_list.append(name)
    else:
        prefix_filtered_list = raw_list
    if suffix is not None:
        filtered_list = []
        for name in prefix_filtered_list:
            if name.endswith(suffix):
                filtered_list.append(name)
    else:
        filtered_list = prefix_filtered_list

    if num_files is None or len(filtered_list) < num_files:
        num_files = len(filtered_list)

    if no_exten:
        namelist = [None] * num_files
        namelist_with_exten = filtered_list[:num_files]
        for i, name in enumerate(namelist_with_exten):
            namelist[i] = os.path.splitext(name)[0]
    else:
        namelist = filtered_list[:num_files]
    
    if no_hidden:
        namelist = [name for name in namelist if not name.startswith('.')]

    return namelist

def remove_if_exists(path):
    """Remove file if exists"""
    try:
        os.remove(path)
    except OSError:
        pass

def mkdir_if_not_exists(path):
    """Make a directory if not already exists"""
    if not os.path.exists(path):
        os.mkdir(path)

def print_arguments(args):
    """Print arguments in a given argparse.Namespace object"""
    for arg in vars(args):
        print('{}: {}'.format(arg, getattr(args, arg)))

def log_arguments(args, path):
    """Write arguments in a given argparse.Namespace object to a text file."""
    with open(path, 'w') as f:
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))

def load_checkpoint(cpdir, model, optimizer, device=torch.device('cpu')):
    """Load model and optimizer parameters from checkpoint

    Note:
        If optimizer is None, do not load optimizer.
        The checkpoint is expected to be a dict containing theh following keys,
            'model_state_dict': state dict of the model,
            'optimizer_state_dict': state dict of the optimizer,
            'epoch': the epoch count.
            'global_step': the global step count.

    Args:
        cpdir (str): path to the checkpoint.
        model: the model to load the parameters to.
        optimizer: the optimizer to load parameters to.
            If None (e.g. test, deploy, etc.), do not load optimizer.

    Returns:
        start_global_step (int): the global step from the checkpoint.
        start_epoch (int): the epoch from the checkpoint.
    """
    checkpoint = torch.load(cpdir, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    start_global_step = checkpoint['global_step']
    return start_global_step, start_epoch