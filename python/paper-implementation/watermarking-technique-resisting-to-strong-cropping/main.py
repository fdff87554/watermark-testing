import cv2
import numpy as np

from sklearn.decomposition import NMF


def nmf(martix):
    model = NMF(n_components=32, init='random')
    W = model.fit_transform(martix)
    print(type(W))
    # H = model.components_

    return W


def main():
    cover = cv2.imread('./images/inputs/gray512x512.png', cv2.IMREAD_GRAYSCALE)
    mark = cv2.imread('./images/inputs/mark.png', cv2.IMREAD_GRAYSCALE).flatten()
    alpha = 1
    hei, wei = cover.shape[:2]
    print(hei, wei)
    N = int(64 * 64) / ((512 / 16) ** 2)
    for h in range(0, hei, 16):
        for w in range(0, wei, 16):
            sub_block = cover[h:h+16, w:w+16]
            sub_w = nmf(sub_block)
            max_pick = sub_w.flatten()
            print(max_pick)
            max_pick = max_pick.sort()
            print(max_pick)
            sub_w = sub_w.tolist()
            break
        break




if __name__ == '__main__':
    main()
