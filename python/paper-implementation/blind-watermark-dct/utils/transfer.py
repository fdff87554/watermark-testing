import cv2
import numpy as np


def place_define(cvr, size):
    hei, wei = cvr.shape
    bin_cvr = np.zeros((hei, wei))
    for h in range(hei):
        for w in range(wei):
            num = np.binary_repr(cvr[h, w], width=8)
            bin_cvr[h, w] = num[:1]

    p_x, p_y, pick = 0, 0, 0
    for h in range(0, hei, size):
        for w in range(0, wei, size):
            sub = bin_cvr[h:h+size, w:w+size]
            cnt = np.sum(sub)
            # the pick place should be the max of the picture
            # also should not be the edge of image
            if pick < cnt and h >= 10 and w >= 10 and (h+size) <= hei and (w+size) <= wei:
                pick = cnt
                p_x, p_y = h, w

    if p_x == 0 and p_y == 0:
        p_x = hei/2 - size/2
        p_y = wei/2 - size/2

    return p_x, p_y


def dct(img):
    freq = cv2.dct(img.astype(np.float32))

    return freq


def idct(freq):
    img = cv2.idct(freq.astype(np.float32))

    return img
