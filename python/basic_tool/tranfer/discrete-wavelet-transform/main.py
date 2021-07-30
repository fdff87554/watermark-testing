import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from utils.tranfer import dwt, idwt


def main():
    image = cv.imread('./images/inputs/cover_gray.png', cv.IMREAD_UNCHANGED)
    a, c = dwt(image)
    hei, wei = a.shape
    ll, lh, hl, hh = a[:hei // 2, :wei // 2], a[:hei // 2, wei // 2:], a[hei // 2:, :wei // 2], a[hei // 2:, wei // 2:]
    plt.imshow(a)
    plt.show()
    plt.imshow(ll)
    plt.show()
    plt.imshow(lh)
    plt.show()
    plt.imshow(hl)
    plt.show()
    plt.imshow(hh)
    plt.show()
    trans_image = idwt(a, c)
    # np.allclose(image, trans_image)
    cv.imwrite('./images/outputs/dwt_basic.png', trans_image)


if __name__ == '__main__':
    main()
