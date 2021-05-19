import cv2
import matplotlib.pyplot as plt
import numpy as np


def test(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    plt.show()

    angles, radiuses, m = transform_data(magnitude_spectrum)

    plt.contourf(angles, radiuses, m)
    plt.show()

    sample_angles = np.linspace(0, 2 * np.pi, len(m.sum(axis=0))) / np.pi * 180
    turn_angle_in_degrees = 90 - sample_angles[np.argmax(m.sum(axis=0))]
    print(turn_angle_in_degrees)
    plt.plot(sample_angles, m.sum(axis=0))
    plt.show()


def transform_data(m):
    dpix, dpiy = m.shape
    x_c, y_c = np.unravel_index(np.argmax(m), m.shape)
    angles = np.linspace(0, np.pi*2, min(dpix, dpiy))
    mrc = min(abs(x_c - dpix), abs(y_c - dpiy), x_c, y_c)
    radiuses = np.linspace(0, mrc, max(dpix, dpiy))
    A, R = np.meshgrid(angles, radiuses)
    X = R * np.cos(A)
    Y = R * np.sin(A)

    return A, R, m[X.astype(int) + mrc - 1, Y.astype(int) + mrc - 1]


def main():
    img = cv2.imread('./images/inputs/test.png', cv2.IMREAD_GRAYSCALE)
    test(img)


if __name__ == '__main__':
    main()
