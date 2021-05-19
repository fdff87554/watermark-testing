import cv2
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


def forward_2fft(img):
    freq = np.zeros(img.shape)
    hei, wei = img.shape
    for h in range(hei):
        freq[h, :] = np.fft.fft(img[h, :])
    for w in range(wei):
        freq[:, w] = np.fft.fft(freq[:, w])

    print(img.shape, freq.shape)

    return freq


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
    image = cv.imread('./images/inputs/lena_color.png', cv.IMREAD_GRAYSCALE)
    plt.imshow(image)
    plt.show()
    freq_np = forward_fft(image)
    r_image = backward_fft(forward_fft(image))
    # plt.imshow(r_image)
    # plt.show()
    plt.imshow(np.log(np.abs(np.fft.fftshift(freq_np))))
    plt.savefig('./images/outputs/np_2fft.png')
    plt.show()
    freq_own = forward_2fft(image)
    plt.imshow(np.log(np.abs(np.fft.fftshift(freq_own))))
    plt.savefig('./images/outputs/np_1fft_2.png')
    plt.show()
    print(freq_np)
    print()
    print(freq_own)
    # r_image = backward_rfft(forward_rfft(image))
    # plt.imshow(r_image)
    # plt.show()


if __name__ == '__main__':
    main()
