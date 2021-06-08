"""Denoiser model.
Colored_DnCNN is DnCNN but with 1xHxW tensors of estimated standard deviation concatenated to the input before every convolution layer.
The model is proposed in

    C. A. Metzler and G. Wetzstein, "D-VDAMP: Denoising-Based Approximate Message Passing
    for Compressive MRI," ICASSP 2021 - 2021 IEEE International Conference on Acoustics, 
    Speech and Signal Processing (ICASSP), 2021, pp. 1410-1414, doi: 10.1109/ICASSP39728.2021.9414708.
"""

import torch
from torch import nn
from torch.nn import functional as F

class Colored_DnCNN(nn.Module):
    def __init__(self, channels=1, num_layers=20, std_channels=13):
        super(Colored_DnCNN, self).__init__()

        self.num_layers = num_layers

        # Fixed parameters
        kernel_size = 3
        padding = 1
        features = 64

        conv_layers = []
        bn_layers = []

        self.first_conv = nn.Conv2d(in_channels=channels+std_channels, out_channels=features,
                                kernel_size=kernel_size, padding=padding, bias=False)
        for _ in range(num_layers - 2):
            conv_layers.append(nn.Conv2d(in_channels=features+std_channels, out_channels=features,
                                    kernel_size=kernel_size, padding=padding, bias=False))
            bn_layers.append(nn.BatchNorm2d(features))
        self.last_conv = nn.Conv2d(in_channels=features+std_channels, out_channels=channels,
                                kernel_size=kernel_size, padding=padding, bias=False)

        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        self._initialize_weights()

    def forward(self, x, std):
        _, _, H, W = x.shape
        std_channels = self._generate_std_channels(std, H, W)
        noise = torch.cat((x, std_channels), dim=1)
        noise = F.relu(self.first_conv(noise))
        for i in range(self.num_layers - 2):
            noise = torch.cat((noise, std_channels), dim=1)
            noise = F.relu(self.bn_layers[i](self.conv_layers[i](noise)))
        noise = torch.cat((noise, std_channels), dim=1)
        noise = self.last_conv(noise)
        out = x - noise
        return out

    def _generate_std_channels(self, std, H, W):
        N, concat_channels = std.shape
        std_channels = std.reshape(N, concat_channels, 1, 1).repeat(1, 1, H, W)
        return std_channels

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)