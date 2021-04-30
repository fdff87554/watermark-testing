import numpy as np


def fft(img):
    freq = np.fft.fft2(img)

    return freq


def ifft(freq):
    img = np.fft.ifft2(freq).real

    return img
