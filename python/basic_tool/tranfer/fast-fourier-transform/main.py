import cv2 as cv

from utils.tranfers import fft, ifft


def main():
    # open image
    image = cv.imread('./images/inputs/cover_gray.png', cv.IMREAD_UNCHANGED)
    image_freq = fft(image)
    trans_image = ifft(image_freq)
    cv.imwrite('./images/outputs/fft.png', trans_image)
    trans_image = cv.imread('./images/outputs/fft.png', cv.IMREAD_UNCHANGED)
    print(image - trans_image)


if __name__ == '__main__':
    main()
