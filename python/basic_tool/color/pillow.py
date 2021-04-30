import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def main():
    # rgb image is open in r, g, b type
    unchanged_image = Image.open('./images/inputs/cover.png')
    print(unchanged_image.format, unchanged_image.size, unchanged_image.mode)
    plt.imshow(unchanged_image)
    plt.show()
    r, g, b = unchanged_image.split()
    merge_image = Image.merge('RGB', (r, g, b))
    print(merge_image.format, merge_image.size, merge_image.mode)
    plt.imshow(merge_image)
    plt.show()
    merge_image.save('./images/outputs/pil_merge.png')

    # check the merge image different with original
    merge_image = Image.open('./images/outputs/pil_merge.png')
    unchanged_image = np.asarray(unchanged_image)
    merge_image = np.asarray(merge_image)
    print(unchanged_image - merge_image)

    unchanged_image = Image.open('./images/inputs/cover_gray.png')
    print(unchanged_image.format, unchanged_image.size, unchanged_image.mode)
    plt.imshow(unchanged_image, cmap='gray')
    plt.show()
    unchanged_image = Image.open('./images/inputs/cat_picture.jpg')
    print(unchanged_image.format, unchanged_image.size, unchanged_image.mode)
    plt.imshow(unchanged_image)
    plt.show()


if __name__ == '__main__':
    main()
