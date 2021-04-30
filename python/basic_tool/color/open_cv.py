import cv2 as cv
import matplotlib.pyplot as plt


def main():
    # both read in IMREAD_UNCHANGE type
    # if the color is grayscale, there will be only one channel
    unchanged_image = cv.imread('./images/inputs/cover_gray.png', cv.IMREAD_UNCHANGED)
    print(unchanged_image.shape)
    plt.imshow(unchanged_image, cmap='gray')
    plt.show()
    # the color image read by cv will be bgr type not rgb
    unchanged_image = cv.imread('./images/inputs/cover.png', cv.IMREAD_UNCHANGED)
    print(unchanged_image.shape)
    plt.imshow(unchanged_image)
    plt.show()
    # split b, g, r three channel and merge back
    b, g, r = cv.split(unchanged_image)
    merge_image = cv.merge((b, g, r))
    print(merge_image.shape)
    plt.imshow(merge_image)
    plt.show()
    # check the merge image different with original
    cv.imwrite('./images/outputs/cv_merge.png', merge_image)
    merge_image = cv.imread('./images/outputs/cv_merge.png', cv.IMREAD_UNCHANGED)
    print(unchanged_image - merge_image)
    # if need to change back to rgb, need to convert bgr to rgb by cv.cvtColor
    rgb_image = cv.cvtColor(unchanged_image, cv.COLOR_BGR2RGB)
    print(rgb_image.shape)
    plt.imshow(rgb_image)
    plt.show()
    # JPEG
    unchanged_image = cv.imread('./images/inputs/cat_picture.jpg', cv.IMREAD_UNCHANGED)
    rgb_image = cv.cvtColor(unchanged_image, cv.COLOR_BGR2RGB)
    print(rgb_image.shape)
    plt.imshow(rgb_image)
    plt.show()


if __name__ == '__main__':
    main()
