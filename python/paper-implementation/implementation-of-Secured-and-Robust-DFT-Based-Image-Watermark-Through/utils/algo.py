import numpy as np


def fft(img):
    freq = np.fft.fft2(img)

    return freq


def ifft(freq):
    img = np.fft.ifft2(freq).real

    return img


# svd input a should be an n*n matrix
def svd(a):
    u, s, vh = np.linalg.svd(a)

    return u, s, vh


def isvd(u, s, vh):
    a = np.dot(u[:, :len(s)] * s, vh)

    return a
