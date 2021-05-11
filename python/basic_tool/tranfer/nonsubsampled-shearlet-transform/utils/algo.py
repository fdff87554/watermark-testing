import numpy as np

import cv2


def nsst(img):
    # get laplacian pyramid low and high pass sub-images
    gauss = img.copy()
    gp = [gauss, cv2.pyrDown(gauss)]
    lp_low = gp[1]
    lp_high = cv2.subtract(gp[0], cv2.pyrUp(gp[1]))

    # transfer high pass sub-images by fourier transformations and transformed into polar coordinate system
    lp_high_ft = np.fft.fft2(lp_high)


# # The example here is a level one laplacian pyramid
# def laplacian(img):
#     # generate Gaussian pyramid
#     gauss = img.copy()
#
#     # get laplacian pyramid
#     gp = [gauss, cv2.pyrDown(gauss)]
#     lp = [gp[1], cv2.subtract(gp[0], cv2.pyrUp(gp[1]))]
#     for l in lp:
#         print(l.shape)
#         plt.imshow(l)
#         plt.show()
#
#     # recover the image
#     for i in range(len(lp) - 1):
#         re_img = cv2.pyrUp(lp[i])
#         re_img = cv2.add(re_img, lp[i+1])
#
#     plt.imshow(re_img)
#     plt.show()
#     print(np.allclose(img, re_img))
#     print(img)
#     print(re_img)
