import cv2
import matplotlib.pyplot as plt
import numpy as np


def logistic_map(lens, mu, x_0):
    x = np.zeros(lens)
    x[0] = x_0
    for i in range(1, lens):
        x[i] = mu * x[i - 1] * (1 - x[i - 1])

    return x


def permutation_map(lens, lm):
    p = np.zeros(lens)
    for i in range(lens):
        p[i] = int((10 ** 8) * lm[i]) % lens + 1

    p = np.sort(p)

    unique, count = np.unique(p, return_counts=True)
    print(dict(zip(unique, count)))

    return p


def image_encrypt(img, mu, x_0):
    hei, wei = img.shape
    lm = logistic_map(hei * wei, mu, x_0)
    pm = permutation_map(hei * wei, lm)
    e_img = img.flatten()
    for i in range(1, len(pm)):
        j = int((i - 1) / wei + 1)
        k = int(i - wei * (j - 1))
        img[j - 1][k - 1] = e_img[int(pm[i]) - 1]

    plt.imshow(img)
    plt.show()

    return img


def image_decrypt(img, mu, x_0):
    hei, wei = img.shape
    lm = logistic_map(hei * wei, mu, x_0)
    pm = permutation_map(hei * wei, lm)
    d_img = img.flatten()
    for i in range(1, len(pm)):
        j = int((i - 1) / wei + 1)
        k = int(i - wei * (j - 1))
        d_img[int(pm[i]) - 1] = img[j - 1][k - 1]

    d_img = d_img.reshape((hei, wei))

    plt.imshow(d_img)
    plt.show()

    return d_img


def main():
    image = cv2.imread('./images/inputs/mark.png', cv2.IMREAD_GRAYSCALE)
    mu, x_0 = 4, 0.3
    en_img = image_encrypt(img=image, mu=mu, x_0=x_0)
    dn_img = image_decrypt(img=en_img, mu=mu, x_0=x_0)


if __name__ == '__main__':
    main()
