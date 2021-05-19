import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.algo import fft, ifft, svd, isvd


# def mark_encrypt(mark, mu, x_0):
#     maps = np.zeros(mark.flatten().shape)
#     p = np.zeros(mark.flatten().shape)
#     maps[0] = x_0
#     for i in range(1, len(maps)):
#         maps[i] = mu * maps[i - 1] * (1 - maps[i - 1])
#     for i in range(len(maps)):
#         p[i] = int((10**8)*maps[i]) % len(maps) + 1
#
#     print(p)
#     return p


def embed(cvr, mark):
    # mark part
    # mark = mark_encrypt(mark, mu, x_0)
    mark_freq = fft(mark)
    mark_u, mark_s, mark_vh = svd(mark_freq)

    # cover part
    m_hei, m_wei = mark.shape
    hei, wei = cvr.shape
    blocks = []
    for h in range(0, hei, m_hei):
        for w in range(0, wei, m_wei):
            if h+m_hei > hei or w+m_wei > wei:
                continue
            blocks.append(cvr[h:h + m_hei, w:w + m_wei])

    em_blk = []
    for block in blocks:
        # plt.imshow(block)
        # plt.show()
        blk_freq = fft(block)
        blk_u, blk_s, blk_vh = svd(blk_freq)
        blk_s += mark_s * 0.1
        blk_freq = isvd(blk_u, blk_s, blk_vh)
        blk = ifft(blk_freq)
        # plt.imshow(blk)
        # plt.show()
        em_blk.append(blk)

    cnt = 0
    for h in range(0, hei, m_hei):
        for w in range(0, wei, m_wei):
            if h+m_hei > hei or w+m_wei > wei:
                continue
            cvr[h:h + m_hei, w:w + m_wei] = em_blk[cnt]
            cnt += 1

    plt.imshow(cvr)
    plt.show()


def detect(img):



def main():
    cover_path = './images/inputs/lena_gray.png'
    watermark_path = './images/inputs/mark.png'
    cover = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    embed(cover, watermark)


if __name__ == '__main__':
    main()
