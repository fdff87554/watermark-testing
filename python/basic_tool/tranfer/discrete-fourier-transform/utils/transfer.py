import cv2 as cv
import numpy as np


def dft(img):
    freq = cv.dft(img.astype(np.float32), flags=cv.DFT_COMPLEX_OUTPUT)
    print(freq.shape)

    return freq


def idft(freq):
    img = cv.idft(freq)
    print(img.shape)
    img = cv.magnitude(img[:, :, 0], img[:, :, 1])
    img = (img/np.max(img)*255).astype(np.float32)

    return img


def rdft(img):
    freq = cv.dft(img.astype(np.float32), flags=cv.DFT_REAL_OUTPUT)
    print(freq.shape)

    return freq


def irdft(freq):
    img = cv.idft(freq)
    print(img.shape)
    # img = (img / np.max(img) * 255).astype(np.float32)
    img = ((img/np.max(img))*255)
    print(img)

    return img

