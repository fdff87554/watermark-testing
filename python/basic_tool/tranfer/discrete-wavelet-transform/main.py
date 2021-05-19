import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from utils.tranfer import dwt, idwt


def main():
    image = cv.imread('./images/inputs/cover_gray.png', cv.IMREAD_UNCHANGED)
    a, c = dwt(image)
    plt.imshow(a)
    plt.show()
    trans_image = idwt(a, c)
    # np.allclose(image, trans_image)
    cv.imwrite('./images/outputs/dwt_basic.png', trans_image)


if __name__ == '__main__':
    main()
