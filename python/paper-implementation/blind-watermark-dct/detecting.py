import argparse

import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils.transfer import place_define, dct, idct


place = [27, 15, 14, 6, 26, 16, 13, 7, 25, 17, 12, 8, 24, 18, 11, 9, 23, 19, 10, 22, 20, 21]
table = [0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11, 18, 24, 31, 40,
         44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34, 37, 47, 50, 56, 59, 61, 35,
         36, 48, 49, 57, 58, 62, 63]


def watermark_detect(mrg, size, alpha, pls_len):
    cnt = 0
    mrg_f = dct(mrg.astype(np.float32))
    length = int(min(mrg_f.shape) / 8) * 8
    mark_f = np.zeros((size ** 2))
    # set replaces index and others
    replaces, others = list(), list()
    for r in place[:pls_len]:
        replaces.append(table.index(r))
    for o in place[pls_len:]:
        others.append(table.index(o))
    for h in range(0, length, 8):
        for w in range(0, length, 8):
            block = mrg_f[h:h + 8, w:w + 8]
            avg = list()
            for o in others:
                avg.append(block[int(o / 8)][o % 8])
            avg = np.average(avg)
            for r in replaces:
                x, y = int(r / 8), r % 8
                mark_f[cnt] = mrg_f[h + x, w + y] - avg
                cnt += 1
                if cnt == len(mark_f):
                    break
            if cnt == len(mark_f):
                break
        if cnt == len(mark_f):
            break

    mark_f = np.resize(mark_f, (size, size))
    mark = idct(mark_f) / alpha
    avg = np.average(mark)
    mark[mark > avg] = 255
    mark[mark <= avg] = 0

    plt.imshow(mark)
    plt.show()

    return mark


def detecting(merge_path, mk_size, color_type, alpha, pls_num):
    need = int(np.ceil((mk_size * mk_size / pls_num) ** 0.5) * 8)
    if color_type == 'L':
        merge = cv2.imread(merge_path, cv2.IMREAD_GRAYSCALE)
        # find embed part
        ph, pw = place_define(merge, need)
        print(ph, pw)
        sub_mrg = merge[ph:ph + need, pw:pw + need]
        detect = watermark_detect(sub_mrg, mk_size, alpha, pls_num)
    elif color_type == 'RGB':
        merge = cv2.imread(merge_path, cv2.IMREAD_COLOR)
        b, _, _ = cv2.split(merge)
        # find embed part
        ph, pw = place_define(b, need)
        print(ph, pw)
        sub_b = b[ph:ph + need, pw:pw + need]
        detect = watermark_detect(sub_b, mk_size, alpha, pls_num)
    else:
        detect = None

    return detect


def main(merge_path, size, color_type, alpha, pls_num, detect_path):
    detect = detecting(merge_path, size, color_type, alpha, pls_num)
    cv2.imwrite(detect_path, detect)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--color', type=str, default='RGB', help='color type of image, ex: L, RGB, RGBA')
    parser.add_argument('--merge', type=str, default='images/outputs/merge.png', help='merge image path')
    parser.add_argument('--size', type=int, default=100, help='mark size')
    parser.add_argument('--alpha', type=float, default=0.05, help='merge alpha')
    parser.add_argument('--pls-num', type=int, default=10, help='merge alpha')
    parser.add_argument('--project', default='images/outputs', help='save to project/name')
    parser.add_argument('--name', default='detect.png', help='save to project/name')
    opt = parser.parse_args()

    main(opt.merge, opt.size, opt.color, opt.alpha, opt.pls_num, '{}/{}'.format(opt.project, opt.name))
