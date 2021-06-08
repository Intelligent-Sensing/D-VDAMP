"""Utility functions for compressive sensing (CS)

    * CStransform: measurement operator for CS.
    * get_cs_param: calculate m and n (size of measurement and image respectively) 
        for a CS simulation.
"""

import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
import torch
import torch_dct as dct

class CStransform:
    def __init__(self, m, n, mode='matmul'):
        """Initialize the CS measurement operator.

        Args:
            m (int): dimension of the measurement.
            n (int): dimension of the image.
            mode (str): how to apply the CS measurement operator A.
                - matmul: matrix multiplication by a Gaussian matrix normalized by column.
                - jl: Fast Johonson-Lindenstrauss Transform.
        """
        self.mode = mode
        if mode == 'matmul':
            A = torch.normal(mean=torch.zeros(m, n), std=1)
            A /= (A ** 2).sum(dim=0).sqrt()
            self.A = A
        elif mode == 'jl':
            idx = np.zeros(m)
            idx[1:] = np.random.choice(n - 1, m - 1, replace=False) + 1
            self.idx = idx
            self.sign_vector = (2 * torch.rand(n).round() - 1).view(-1, 1)
        else:
            raise ValueError('deploy.cs.CStransform: Invalid mode')
        self.m = m
        self.n = n

    def Afun(self, x):
        """Operator for A @ x matrix-vector multiplication."""
        if self.mode == 'matmul':
            return self.A.matmul(x)
        elif self.mode == 'jl':
            dct_result = dct.dct((self.sign_vector * x).view(-1), norm='ortho').view(-1, 1)
            sampling_result = dct_result[self.idx]
            return np.sqrt(self.n / self.m) * sampling_result
        else:
            raise ValueError('deploy.cs.CStransform: Invalid mode')

    def Atfun(self, x):
        """Operator for A.T @ x matrix-vector multiplication."""
        if self.mode == 'matmul':
            return self.A.transpose(0, 1).matmul(x)
        elif self.mode == 'jl':
            sampling_result = torch.zeros(self.n, 1)
            sampling_result[self.idx] = x
            return np.sqrt(self.n / self.m) * self.sign_vector * dct.idct(sampling_result.view(-1), norm='ortho').view(-1, 1)
        else:
            raise ValueError('deploy.cs.CStransform: Invalid mode')

    def get_m(self):
        return self.m

    def get_n(self):
        return self.n

    def get_A(self):
        if self.mode == 'matmul':
            return self.A
        else:
            raise ValueError('deploy.cs.CStransform: Incompatible mode')

    def set_A_device(self, device):
        if self.mode == 'matmul':
            self.A = self.A.to(device=device)
        else:
            raise RuntimeError('util.cs.CStransform.set_A_device: Incompatible mode')

def get_cs_param(shape, sampling_rate):
    _, H, W = shape
    n = int(H * W)
    m = int(np.round(n * sampling_rate))
    return m, n
