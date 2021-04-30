import cv2 as cv
import numpy as np

from utils.tranfer import cv_dct, cv_idct, scipy_dct, scipy_idct


def main():
    image = cv.imread('./images/inputs/cover_gray.png', cv.IMREAD_UNCHANGED)
    cv_trans_image = cv_idct(cv_dct(image))
    scipy_trans_image = scipy_idct(scipy_dct(image))
    np.allclose(image, cv_trans_image)
    np.allclose(image, scipy_trans_image)
    np.allclose(cv_trans_image, scipy_trans_image)
    cv.imwrite('./images/outputs/cv_dct.png', cv_trans_image)
    cv.imwrite('./images/outputs/scipy_dct.png', scipy_trans_image)


if __name__ == '__main__':
    main()
