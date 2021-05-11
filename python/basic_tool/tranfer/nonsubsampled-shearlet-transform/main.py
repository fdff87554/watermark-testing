import cv2
import matplotlib.pyplot as plt
import numpy as np


# def nsst(img):
#     # get laplacian pyramid low and high pass sub-images
#     gauss = img.copy()
#     gp = [gauss, cv2.pyrDown(gauss)]
#     lp_low = gp[1]
#     lp_high = cv2.subtract(gp[0], cv2.pyrUp(gp[1]))
#     lp_high_ft =


def main():
    img = cv2.imread('./images/inputs/cover.png', cv2.IMREAD_GRAYSCALE)
    # nsst(img)
    test(img)


if __name__ == '__main__':
    main()
