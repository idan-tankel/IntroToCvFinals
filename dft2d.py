import numpy as np
import matplotlib.pyplot as plt
import PIL
import cmath
import torch
import torch.fft


def DFT2D(image):
    data = np.asarray(image)
    M, N = image.size  # (img x, img y)
    dft2d = np.zeros((M, N), dtype=complex)
    for k in range(M):
        for l in range(N):
            sum_matrix = 0.0
            for m in range(M):
                for n in range(N):
                    e = cmath.exp(- 2j * np.pi * ((k * m) / M + (l * n) / N))
                    sum_matrix += data[m, n, 1] * e
            dft2d[k, l] = sum_matrix
    return dft2d


''' implementaion of the following medioum page 
https://kai760.medium.com/how-to-use-torch-fft-to-apply-a-high-pass-filter-to-an-image-61d01c752388
'''
def roll_n(X, axis, n):
    '''

    '''
    f_idx = tuple(slice(None, None, None)
                  if i != axis else slice(0, n, None)
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None)
                  if i != axis else slice(n, None, None)
                  for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def fftshift(X):
    real, imag = X.chunk(chunks=2, dim=-1)
    real, imag = real.squeeze(dim=-1), imag.squeeze(dim=-1)
    for dim in range(2, len(real.size())):
        real = roll_n(real, axis=dim,
                      n=int(np.ceil(real.size(dim) / 2)))
        imag = roll_n(imag, axis=dim,
                      n=int(np.ceil(imag.size(dim) / 2)))
    real, imag = real.unsqueeze(dim=-1), imag.unsqueeze(dim=-1)
    X = torch.cat((real, imag), dim=1)
    return torch.squeeze(X)


def ifftshift(X):
    real, imag = X.chunk(chunks=2, dim=-1)
    real, imag = real.squeeze(dim=-1), imag.squeeze(dim=-1)

    for dim in range((len(real.size()) — 1), 1, -1)):
        real = roll_n(real, axis=dim,
                      n=int(np.floor(real.size(dim) / 2)))
        imag = roll_n(imag, axis=dim,
                      n=int(np.floor(imag.size(dim) / 2)))
    real, imag = real.unsqueeze(dim=-1), imag.unsqueeze(dim=-1)
    X = torch.cat((real, imag), dim=1)
    return torch.squeeze(X)
