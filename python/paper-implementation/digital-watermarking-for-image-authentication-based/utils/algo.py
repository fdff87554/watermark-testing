import cv2
import pywt
import numpy as np


def dct(img):
    freq = cv2.dct(img.astype(np.float32))

    return freq


def idct(freq):
    img = cv2.idct(freq)

    return img


def dwt(img, level, wavelet):
    coeffs = pywt.wavedec2(img, level=level, wavelet=wavelet)
    arr, coeffs_slices = pywt.coeffs_to_array(coeffs)

    return arr, coeffs_slices


def idwt(arr, coeffs_slices, wavelet):
    coeffs = pywt.array_to_coeffs(arr, coeffs_slices, output_format='wavedec2')
    img = pywt.waverec2(coeffs, wavelet=wavelet)

    return img


# svd input a should be an n*n matrix
def svd(a):
    u, s, vh = np.linalg.svd(a)

    return u, s, vh


def isvd(u, s, vh):
    a = np.dot(u[:, :len(s)] * s, vh)

    return a
