"""Main program for training/testing Colored DnCNN
Perform one of the following actions based on the command line arguments.
    - preprocess
    - train
"""

import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from preprocess import generate_datasets
from dataset import DenoiseDataset
from model import Colored_DnCNN
import solve
import train_util
from util.general import mkdir_if_not_exists, log_arguments, print_arguments

# Argument parsing
parser = argparse.ArgumentParser(description='Preprocess or Train/Test CNN denoiser')

# Common arguments
parser.add_argument('mode', type=str, choices=[
                    'preprocess', 'train'], help='Action of this program.')
parser.add_argument('--datadir', type=str,
                    help='Path to images or .h5 files.')
parser.add_argument('--gpu', action='store_true',
                    help='Whether to use GPU for computations when applicable.')

# Arguments for preprocess
parser.add_argument('--numtrain', type=int, default=None,
                    help='Number of train images. If None, use all images in train directory.')
parser.add_argument('--numval', type=int, default=None,
                    help='Number of val images. If None, use all images in val directory.')
parser.add_argument('--numtest', type=int, default=None,
                    help='Number of test images. If None, use all images in test directory.')
parser.add_argument('--trainwindow', type=int, default=48,
                    help='Patch size of training images.')

# Arguments for train
parser.add_argument('--modeldir', type=str,
                    help='Path to saving checkpoint for training or loading checkpoint for training (including .pth).')
parser.add_argument('--logdir', type=str,
                    help='Path to tensorboard log directory.')
parser.add_argument('--numlayers', type=int, default=20,
                    help='Number of layers in the CNN model.')
parser.add_argument('--logimage', type=int, default=[-1], nargs='*',
                    help='Whether to log images when training/testing.')
parser.add_argument('--stdrange', type=int, default=[20, 40], nargs='*',
                    help='Uniform random STDs between the two values are used.')
parser.add_argument('--wavetype', type=str, default='haar',
                    help='Wavelet type.')
parser.add_argument('--level', type=int, default=4,
                    help='Number of levels of the wavelet transform.')
parser.add_argument('--cpdir', type=str, default=None,
                    help='Path to checkpoint to resume model training. If None, train a new model.')
parser.add_argument('--batchsize', type=int, default=100,
                    help='Number of patches in a batch.')
parser.add_argument('--learnrate', type=float, default=1e-4,
                    help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=1, help='Total epochs for training.')
parser.add_argument('--patience', type=int, default=5, help='Number of epochs with no improvement in PSNR after which learning rate is scaled by 0.1.')
parser.add_argument('--logevery', type=int, default=10,
                    help='Log loss and PSNR every this number of iterations in an epoch (1 iteration contains #batchsize images).')

# Set up device
args = parser.parse_args()
USE_GPU = args.gpu
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Using device: {}'.format(device))

mode = args.mode
if __name__ == '__main__':
    if mode == 'preprocess':

        print('Parsing training arguments...')
        datadir = args.datadir
        num_train = args.numtrain
        num_val = args.numval
        num_test = args.numtest
        train_window = args.trainwindow

        print('Generating datasets...')
        generate_datasets(datadir, num_train=num_train,
                          num_val=num_val, num_test=num_test, train_window=train_window)

        print('Generating dataset: Done!')

    else: # mode == 'train'

        print('Parsing training arguments...')
        cpdir = args.cpdir
        datadir = args.datadir
        logdir = args.logdir
        modeldir = args.modeldir
        num_layers = args.numlayers
        std_range = torch.tensor(args.stdrange) / 255.
        wavetype = args.wavetype
        level = args.level
        batch_size = args.batchsize
        lr = args.learnrate
        epochs = args.epochs
        patience = args.patience
        log_every = args.logevery
        log_image = args.logimage

        print('Parameters:')
        print_arguments(args)
        print()

        mkdir_if_not_exists(logdir)
        mkdir_if_not_exists(modeldir[:modeldir.rfind('/')])
        log_arguments(args, os.path.join(logdir, 'args.txt'))

        print('Setting up training...')
        transforms_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            train_util.FixedAngleRotation([0, 90, 180, 270]),
            transforms.ToTensor(),
        ])

        dataset_train = DenoiseDataset(os.path.join(datadir, 'train.h5'), std_range, 
                                        wavetype=wavetype, level=level, transforms=transforms_train)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

        dataset_val = DenoiseDataset(os.path.join(datadir, 'val.h5'), std_range,
                                        wavetype=wavetype, level=level)
        loader_val = DataLoader(dataset_val)
        dataset_test = DenoiseDataset(os.path.join(datadir, 'test.h5'), std_range,
                                        wavetype=wavetype, level=level)
        loader_test = DataLoader(dataset_test)

        # Insert your model here!
        std_channels = 3 * level + 1
        model = Colored_DnCNN(num_layers=num_layers, std_channels=std_channels)
        model = model.to(device=device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=patience)

        start_global_step, start_epoch = train_util.load_checkpoint_train(cpdir, model, optimizer)

        writer = SummaryWriter(logdir)

        print('Begin training...')
        solve.train(model, loader_train, optimizer, epochs=epochs, scheduler=scheduler,
                    loader_val=loader_val, loader_test=loader_test, device=device, writer=writer,
                    log_every=log_every, log_image=log_image, savedir=modeldir,
                    start_epoch=start_epoch, start_global_step=start_global_step)

        writer.close()
        print('Training: Done!')