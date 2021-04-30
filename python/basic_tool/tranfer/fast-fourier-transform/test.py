import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def forward_fft(img):
    freq = np.fft.fft2(img)
    print(img.shape, freq.shape)
    print("freq from fft", freq)

    return freq


def backward_fft(freq):
    img = np.fft.ifft2(freq).real
    print(freq.shape, img.shape)
    print("img from ifft", img)

    return img


def forward_rfft(img):
    freq = np.fft.rfft2(img)
    print(img.shape, freq.shape)
    print("freq from rfft", freq)

    return freq


def backward_rfft(freq):
    img = np.fft.irfft2(freq)
    print(freq.shape, img.shape)
    print("img from irfft", img)

    return img


def main():
    image = cv.imread('./images/inputs/cover_gray.png', cv.IMREAD_UNCHANGED)
    plt.imshow(image)
    plt.show()
    r_image = backward_fft(forward_fft(image))
    plt.imshow(r_image)
    plt.show()
    r_image = backward_rfft(forward_rfft(image))
    plt.imshow(r_image)
    plt.show()


if __name__ == '__main__':
    main()
