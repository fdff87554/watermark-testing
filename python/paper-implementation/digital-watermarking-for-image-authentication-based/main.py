import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils.algo import dwt, idwt, dct, idct, svd, isvd
from utils.zigzag import mapping, imapping


def enbed_image(cvr, mark, level, wavelet, alpha):
    # cover preprocess
    cvr_hei, cvr_wei = cvr.shape
    cvr_arr, cvr_cs = dwt(cvr, level, wavelet)
    cvr_hh = cvr_arr[int(cvr_hei / 2):cvr_hei, int(cvr_wei / 2):cvr_wei]
    cvr_freq = dct(cvr_hh)
    bk1, bk2, bk3, bk4 = mapping(cvr_freq)
    bk1_u, bk1_s, bk1_vh = svd(bk1)
    bk2_u, bk2_s, bk2_vh = svd(bk2)
    bk3_u, bk3_s, bk3_vh = svd(bk3)
    bk4_u, bk4_s, bk4_vh = svd(bk4)

    # mark preprocess
    mark_hei, mark_wei = mark.shape
    mark_arr, mark_cs = dwt(mark, level, wavelet)
    mark_hh = mark_arr[int(mark_hei / 2):mark_hei, int(mark_wei / 2):mark_wei]
    mark_freq = dct(mark_hh)
    mark_u, mark_s, mark_vh = svd(mark_freq)
    # print(bk1_s.shape, bk2_s.shape, bk3_s.shape, bk4_s.shape, mark_s.shape)

    # embedding
    bk1_s = bk1_s + alpha * mark_s
    bk2_s = bk2_s + alpha * mark_s
    bk3_s = bk3_s + alpha * mark_s
    bk4_s = bk4_s + alpha * mark_s

    # return image
    bk1 = isvd(bk1_u, bk1_s, bk1_vh)
    bk2 = isvd(bk2_u, bk2_s, bk2_vh)
    bk3 = isvd(bk3_u, bk3_s, bk3_vh)
    bk4 = isvd(bk4_u, bk4_s, bk4_vh)

    cvr_freq = imapping(bk1, bk2, bk3, bk4)
    cvr_hh = idct(cvr_freq)
    cvr_arr[int(cvr_hei / 2):cvr_hei, int(cvr_wei / 2):cvr_wei] = cvr_hh
    cvr = idwt(cvr_arr, cvr_cs, wavelet)
    plt.imshow(cvr)
    plt.show()

    return cvr


def detect_image(mrg, level, wavelet, alpha):
    mrg_hei, mrg_wei = mrg.shape
    mrg_arr, mrg_cs = dwt(mrg, level, wavelet)
    mrg_hh = mrg_arr[int(mrg_hei / 2):mrg_hei, int(mrg_wei / 2):mrg_wei]


def main():
    level, wavelet, alpha = 1, 'db1', 1
    cover = cv2.imread('./images/inputs/lena_gray.png', cv2.IMREAD_GRAYSCALE)
    watermark = cv2.imread('./images/inputs/mark_256.png', cv2.IMREAD_GRAYSCALE)
    merge = enbed_image(cover, watermark, level, wavelet, alpha)
    cv2.imwrite('./images/outputs/merge.png', merge)
    merge = cv2.imread('./images/outputs/merge.png', cv2.IMREAD_GRAYSCALE)



if __name__ == '__main__':
    main()
