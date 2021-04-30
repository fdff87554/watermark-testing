import numpy as np
import cv2 as cv

from scipy import fftpack


def cv_dct(img):
    freq = cv.dct(img.astype(np.float32))

    return freq


def cv_idct(freq):
    img = cv.idct(freq)

    return img


def scipy_dct(img):
    freq = fftpack.dct(fftpack.dct(img.T, norm='ortho').T, norm='ortho')

    return freq


def scipy_idct(freq):
    img = fftpack.idct(fftpack.idct(freq.T, norm='ortho').T, norm='ortho')

    return img
