"""Train and test functions."""

import torch
from torch.nn.functional import mse_loss
from train_util import batch_psnr, dtype
import time

def train(model,
          loader_train,
          optimizer,
          scheduler=None,
          epochs=1,
          loader_val=None,
          loader_test=None,
          device=torch.device('cpu'),
          savedir=None,
          writer=None,
          log_every=20,
          log_image=[-1],
          start_epoch=0,
          start_global_step=0):
    """Train the DnCNN model.
    Note:
        If specified, the model is validated and saved every epoch.
        The loss, PSNR, and validation images are logged every epoch.
    Args:
        model: the DnCNN instance to be trained.
        loader_train (torch.utils.data.DataLoader): train dataset loader.
        optimizer (torch.optim.Optimizer): the optimizer.
        scheduler: (torch.optim.lr_scheduler) a learning rate scheduler
            Assumed to be ReduceLROnPlateau type for now.
        epoches (int): the number of epochs.
        loader_val (torch.utils.data.DataLoader): validation dataset loader.
            If None, do not perform validation.
        device (torch.device): the device to perform computations on.
        savedir (string): path to save model every epoch. Do not save if None.
        writer (tensorboardX.SummaryWriter): log writer for a train/test session.
            If None, do not log
        log_every (int): print and log the loss and PSNR every this number of
            iterations within an epoch.
        log_image (list): Log validation images of indices in this list.
        start_epoch (int): epoch to begin (for resuming from checkpoint)
        start_global_step (int): global step to begin (for resuming from checkpoint)
        objective_params: parameters for calculating loss/objective function
    """
    start_time = time.time()
    _check_log_image(log_image, len(loader_val))
    global_step = start_global_step
    for e in range(start_epoch, start_epoch + epochs):
        model.train()
        for i, (image, noisy_image, std) in enumerate(loader_train):
            image = image.to(device=device, dtype=dtype)
            noisy_image = noisy_image.to(device=device, dtype=dtype)
            std = std.to(device=device, dtype=dtype)
            denoised_image = model(noisy_image, std)
            loss = mse_loss(denoised_image, image, reduction='mean')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % log_every == 0:
                _log_train(image, denoised_image, e, i, loss, global_step, writer)
            global_step += 1
        if loader_val is not None:
            print('Validation after epoch {:d}'.format(e))
            psnr = eval(model, loader_val, is_train=True, device=device, writer=writer, epoch=e, log_image=log_image)
            scheduler.step(psnr)
        _log_epoch(writer, optimizer, e, start_time)
        _save_checkpoint(savedir, e, global_step, model, optimizer)
        print()
    print('Training: Done!')
    if loader_test is not None:
        print('Begin testing...')
        eval(model, loader_test, is_train=False, device=device, writer=writer, log_image=log_image)

def eval(model,
         loader,
         is_train,
         device=torch.device('cpu'),
         writer=None,
         epoch=0,
         log_image=[-1]):
    """Validate or test a DnCNN model.
    Note:
        Report average PSNR of the whole validation or test set.
        Log validation PSNR. Note that passing summary writer
        at test time does not do anything.
    Args:
        model: the DnCNN instance to be tested/validated.
        loader (torch.utils.data.DataLoader): test/val dataset loader.
        is_train: whether this function is called during training.
        device (torch.device): the device to perform computations on.
        writer (tensorboardX.SummaryWriter): log writer for a train/test session.
            If None, do not log
        epoch (int): training epoch when using this function for validation.
        log_image (list): Log validation or test images of indices in this list.
            If None, do not log images.
    Returns:
        psnr (float): validation PSNR
    """
    model.eval()
    image_log_dir = _select_imlogdir(is_train)
    with torch.no_grad():
        psnr = 0
        for i, (image, noisy_image, std) in enumerate(loader):
            image = image.to(device=device, dtype=dtype)
            noisy_image = noisy_image.to(device=device, dtype=dtype)
            std = std.to(device=device, dtype=dtype)
            denoised_image = model(noisy_image, std)
            psnr += batch_psnr(denoised_image, image, max=1.)
            if i in log_image and writer is not None:
                writer.add_image('{}{}'.format(image_log_dir, i), torch.squeeze(denoised_image, dim=0), epoch)
        psnr /= len(loader.dataset)
    _log_eval(is_train, psnr, epoch, writer)
    return psnr

def _check_log_image(log_image, loader_len):
    """Verify that indices in log_image list do not exceed size of the loader."""
    for i in log_image:
        if i > loader_len - 1:
            raise RuntimeError('solve.train._check_log_image: index in log_image exceeds size of val loader')

def _save_checkpoint(savedir, epoch, global_step, model, optimizer):
    """Save current checkpoint when training if the path is provided"""
    if savedir is not None:
        print('Saving the model at epoch {:d}...'.format(epoch))
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, savedir)

@torch.no_grad()
def _log_train(image,
               denoised_image,
               epoch,
               iteration,
               loss,
               global_step,
               writer):
    """Log the training loss and PSNR."""
    psnr = batch_psnr(denoised_image, image, max=1.)
    mse = mse_loss(denoised_image, image, reduction='sum') / (2 * image.shape[0])
    diff = (mse - loss).abs()
    print('Epoch {:d} Iteration {:d}, Loss = {:.4f}, PSNR = {:.4f}'.format(epoch, iteration, loss.item(), psnr))
    if writer is not None:
        writer.add_scalar('loss/train', loss.item(), global_step)
        writer.add_scalar('mse/train', mse.item(), global_step)
        writer.add_scalar('mse_diff_loss/train', diff.item(), global_step)
        writer.add_scalar('PSNR/train', psnr, global_step)

def _select_imlogdir(is_train):
    """Choose log name for image logging."""
    if is_train:
        return 'denoised/val/'
    else:
        return 'denoised/test/'

def _log_eval(is_train, psnr, epoch, writer):
    """Log the validation/testing PSNR and learning rate."""
    if is_train:
        print_message = 'Validation PSNR = '
        log_message = 'PSNR/val'
    else:
        print_message = 'Test PSNR = '
        log_message = 'PSNR/test'
    print(print_message + '{:.4f}'.format(psnr))

    if writer is not None:
        writer.add_scalar(log_message, psnr, epoch)

def _log_epoch(writer, optimizer, epoch, start_time):
    """Log optimizer learning rate and time"""
    lr_log = 'lr/'
    for i, param_group in enumerate(optimizer.param_groups):
        writer.add_scalar(lr_log + '{:d}'.format(i), param_group['lr'], epoch)
    time_spent = time.time() - start_time
    print('Time from begin training: {}'.format(time_spent))
    writer.add_scalar('time', time_spent, epoch)