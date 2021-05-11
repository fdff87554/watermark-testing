import cv2
import matplotlib.pyplot as plt
#
# from utils.algo import sift


def main():
    cover = cv2.imread('./images/inputs/cover_gray.png')
    print(cover.shape)
    gray = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
    print(gray.shape)

    sift = cv2.SIFT_create()
    kp = sift.detect(gray, None)
    print(len(kp))
    img = None
    img = cv2.drawKeypoints(gray, kp, img)
    print(img)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
