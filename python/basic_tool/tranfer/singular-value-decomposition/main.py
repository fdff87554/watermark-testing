import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.algo import svd, isvd


def main():
    image = cv2.imread('./images/inputs/lena_color.png', cv2.IMREAD_GRAYSCALE)
    u, s, vh = svd(image)
    img = isvd(u, s, vh)
    plt.imshow(img)
    plt.show()
    print(np.allclose(img, image))

if __name__ == '__main__':
    main()
