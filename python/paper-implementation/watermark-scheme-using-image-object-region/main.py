import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.algo import dwt, idwt


def main(img_path, mark_path, level):
    wavelet = 'db1'

    # open image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    print(image.shape)
    freq, coeffs_slices = dwt(image, level, wavelet)
    hei, wei = freq.shape
    plt.imshow(freq)
    plt.show()
    mask = np.zeros((int(hei/3), int(wei/3)))
    mask_freq = freq
    mask_freq[0:int(hei/3), 0:int(wei/3)] = mask
    plt.imshow(mask_freq)
    plt.show()
    back_img = idwt(mask_freq, coeffs_slices, wavelet)
    plt.imshow(back_img)
    plt.show()


if __name__ == '__main__':
    img_path = './images/inputs/cover_gray.png'
    mark_path = './images/inputs/mark.png'
    level = 2
    main(img_path, mark_path, level)
