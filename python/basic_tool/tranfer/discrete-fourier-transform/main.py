import cv2 as cv
import matplotlib.pyplot as plt

from utils.transfer import dft, idft, rdft, irdft


def main():
    image = cv.imread('./images/inputs/cover_gray.png', cv.IMREAD_UNCHANGED)
    image_freq = dft(image)
    trans_image = idft(image_freq)
    plt.imshow(trans_image)
    plt.show()
    cv.imwrite('./images/outputs/dft.png', trans_image)
    trans_image = cv.imread('./images/outputs/dft.png', cv.IMREAD_UNCHANGED)
    print(image - trans_image)
    image_freq = rdft(image)
    trans_image = irdft(image_freq)
    plt.imshow(trans_image)
    plt.show()
    cv.imwrite('./images/outputs/dft_r.png', trans_image)
    trans_image = cv.imread('./images/outputs/dft_r.png', cv.IMREAD_UNCHANGED)
    print(image - trans_image)


if __name__ == '__main__':
    main()
