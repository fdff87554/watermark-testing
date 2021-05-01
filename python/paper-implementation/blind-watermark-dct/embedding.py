import argparse

import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils.transfer import place_define, dct, idct


place = [27, 15, 14, 6, 26, 16, 13, 7, 25, 17, 12, 8, 24, 18, 11, 9, 23, 19, 10, 22, 20, 21]
table = [0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11, 18, 24, 31, 40,
         44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34, 37, 47, 50, 56, 59, 61, 35,
         36, 48, 49, 57, 58, 62, 63]


def watermark_merge(cvr, mk, pls_len):
    cnt = 0
    # change domain
    cvr_f = dct(cvr.astype(np.float32))
    mark_f = dct(mk.astype(np.float32)).flatten()
    length = int(min(cvr.shape) / 8) * 8
    # set replaces index and others
    replaces, others = list(), list()
    for r in place[:pls_len]:
        replaces.append(table.index(r))
    for o in place[pls_len:]:
        others.append(table.index(o))
    for h in range(0, length, 8):
        for w in range(0, length, 8):
            block = cvr_f[h:h+8, w:w+8]
            avg = list()
            for o in others:
                avg.append(block[int(o / 8)][o % 8])
            avg = np.average(avg)
            for r in replaces:
                x, y = int(r / 8), r % 8
                cvr_f[h + x, w + y] = mark_f[cnt] + avg
                cnt += 1
                if cnt == len(mark_f):
                    break
            if cnt == len(mark_f):
                break
        if cnt == len(mark_f):
            break

    merge = idct(cvr_f)
    plt.imshow(merge)
    plt.show()

    return merge


def img_embed(cvr, mark, mk_size, alpha, pls_num):
    need = int(np.ceil((mk_size * mk_size / pls_num) ** 0.5) * 8)

    # find embed part
    ph, pw = place_define(cvr, need)
    print(ph, pw)
    sub_cvr = cvr[ph:ph+need, pw:pw+need]
    merge = watermark_merge(sub_cvr, mark * alpha, pls_num)
    merge_cvr = cvr
    merge_cvr[ph:ph+need, pw:pw+need] = merge
    plt.imshow(merge_cvr)
    plt.show()

    return merge_cvr


def embedding(cover_path, mark_path, mark_size, color_type, alpha, pls_num):
    mark = cv2.imread(mark_path, cv2.IMREAD_GRAYSCALE)
    if color_type == 'L':
        cover = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
        merge = img_embed(cover, mark, mark_size, alpha, pls_num)
    elif color_type == 'RGB':
        cover = cv2.imread(cover_path, cv2.IMREAD_COLOR)
        b, g, r = cv2.split(cover)
        b_m = img_embed(b, mark, mark_size, alpha, pls_num)
        merge = cv2.merge((b_m.astype(np.float32), g.astype(np.float32), r.astype(np.float32)))
    else:
        merge = None

    return merge


def main(user_input, cover_path, mark_path, mark_size, color_type, alpha, pls_num, merge_path):
    merge = embedding(cover_path, mark_path, mark_size, color_type, alpha, pls_num)
    cv2.imwrite(merge_path, merge)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--user-input', type=str, default='', help='user inputs')
    parser.add_argument('--cover', type=str, default='images/inputs/cover.png', help='cover image path')
    parser.add_argument('--mark', type=str, default='images/inputs/mark.png', help='mark image path')
    parser.add_argument('--size', type=int, default=100, help='mark size')
    parser.add_argument('--color', type=str, default='RGB', help='color type of image, ex: L, RGB, RGBA')
    parser.add_argument('--alpha', type=float, default=0.05, help='merge alpha')
    parser.add_argument('--pls-num', type=int, default=10, help='merge alpha')
    parser.add_argument('--project', default='images/outputs', help='save to project/name')
    parser.add_argument('--name', default='merge.png', help='save to project/name')
    opt = parser.parse_args()

    main(opt.user_input, opt.cover, opt.mark, opt.size, opt.color, opt.alpha, opt.pls_num, '{}/{}'.format(opt.project, opt.name))
