import cv2
import matplotlib.pyplot as plt


def laplacian_pyramid(img, level):
    gauss = img.copy()
    # get gauss pyramid
    gp = [gauss]
    for i in range(0, level):
        gauss = cv2.pyrDown(gauss)
        gp.append(gauss)

    lp = [gp[level]]
    for i in range(level, 0, -1):
        lp.append(cv2.subtract(gp[i - 1], cv2.pyrUp(gp[i])))

    return lp


def main():
    image = cv2.imread('./images/inputs/lena_color.png', cv2.IMREAD_GRAYSCALE)
    laplacian = laplacian_pyramid(image, 2)
    for i in range(len(laplacian)):
        plt.imshow(laplacian[i])
        plt.show()


if __name__ == '__main__':
    main()
