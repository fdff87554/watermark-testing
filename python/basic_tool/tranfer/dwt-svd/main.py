import cv2
import numpy as np
import pywt


# svd input a should be an n*n matrix
def svd(a):
    u, s, vh = np.linalg.svd(a)

    return u, s, vh


def isvd(u, s, vh):
    a = np.dot(u[:, :len(s)] * s, vh)

    return a


def dwt(img, level, wavelet):
    coeffs = pywt.wavedec2(img, level=level, wavelet=wavelet)
    arr, coeffs_slices = pywt.coeffs_to_array(coeffs)

    return arr, coeffs_slices


def idwt(arr, coeffs_slices, wavelet):
    coeffs = pywt.array_to_coeffs(arr, coeffs_slices, output_format='wavedec2')
    img = pywt.waverec2(coeffs, wavelet=wavelet)

    return img


def main():
    cover = cv2.imread('./images/inputs/lena_gray.png', cv2.IMREAD_GRAYSCALE)


if __name__ == '__main__':
    main()
