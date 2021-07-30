import cv2
import matplotlib.pyplot as plt
import numpy as np
import pywt


def place_count(hei, wei):
    arr = []
    w, h, bons = 0, 0, 1
    while (w + h) < (wei + hei - 2):
        # if running iterator is even, go up
        if (w + h) % 2 == 0:
            h += 1
            if w > 0:
                w -= 1
        else:
            w += 1
            if h > 0:
                h -= 1
        if h == hei:
            h -= 1
            w += bons
            bons = 2
        if w == wei:
            w -= 1
            h += bons
            bons = 2
        arr.append((w, h))

    return arr


# arr is an np array type
def zigzag(arr):
    hei, wei = arr.shape
    output = [arr[0][0]]
    places = place_count(hei, wei)
    for i in range(len(places)):
        w, h = places[i]
        output.append(arr[w][h])

    return np.array(output)


def izigzag(arr):
    hei, wei = int(len(arr) ** 0.5), int(len(arr) ** 0.5)
    output = np.zeros((hei, wei))
    output[0][0] = arr[0]
    places = place_count(hei, wei)
    for i in range(len(places)):
        w, h = places[i]
        output[w][h] = arr[i + 1]

    return output


def dwt(img, level, wavelet):
    coeffs = pywt.wavedec2(img, level=level, wavelet=wavelet)
    arr, coeffs_slices = pywt.coeffs_to_array(coeffs)

    return arr, coeffs_slices


def idwt(arr, coeffs_slices, wavelet):
    coeffs = pywt.array_to_coeffs(arr, coeffs_slices, output_format='wavedec2')
    img = pywt.waverec2(coeffs, wavelet=wavelet)

    return img


# def


def embedded(cvr, mark, level, wave):
    # for level one embed
    c_hei, c_wei = cvr.shape
    c_arr, c_cs = dwt(cvr, level, wave)
    c_lh, c_hl = c_arr[:c_hei // 2, c_wei // 2:], c_arr[c_hei // 2:, :c_wei // 2]
    c_lh_freq, c_hl_freq = cv2.dct(c_lh.astype(np.float32)), cv2.dct(c_hl.astype(np.float32))
    mark_freq = cv2.dct(mark.astype(np.float32)).flatten()
    # count need len for each block
    need_len = round(mark_freq.shape[0] / ((min(c_hl.shape) / 8) ** 2))
    print(need_len)
    # embed
    cnt = 0
    for h in range(0, c_lh_freq.shape[0], 8):
        for w in range(0, c_lh_freq.shape[1], 8):
            # get embed 8 * 8 block
            lh_block, hl_block = c_lh_freq[h:h + 8, w:w + 8], c_hl_freq[h:h + 8, w:w + 8]
            # get zigzag array
            lh_block_list, hl_block_list = zigzag(lh_block), zigzag(hl_block)
            for i in range(need_len):
                lh_block_list[i + 64 - need_len] = mark_freq[cnt] * 0.1
                hl_block_list[i + 64 - need_len] = mark_freq[cnt] * 0.1
                cnt += 1
                if cnt >= mark_freq.shape[0]:
                    break
            # return zigzag array to 8 * 8 block
            lh_block, hl_block = izigzag(lh_block_list), izigzag(hl_block_list)
            c_lh_freq[h:h + 8, w:w + 8], c_hl_freq[h:h + 8, w:w + 8] = lh_block, hl_block
            if cnt >= mark_freq.shape[0]:
                break
        if cnt >= mark_freq.shape[0]:
            break
    # return to image
    c_lh, c_hl = cv2.idct(c_lh_freq), cv2.idct(c_hl_freq)
    c_arr[:c_hei // 2, c_wei // 2:], c_arr[c_hei // 2:, :c_wei // 2] = c_lh, c_hl
    merge = idwt(c_arr, c_cs, wave)
    plt.imshow(merge)
    plt.show()

    return merge


def detect(merge, level, wave, need_len, lengths):
    m_hei, m_wei = merge.shape
    m_arr, _ = dwt(merge, level, wave)
    m_lh, m_hl = m_arr[:m_hei // 2, m_wei // 2:], m_arr[m_hei // 2:, :m_wei // 2]
    m_lh_freq, m_hl_freq = cv2.dct(m_lh.astype(np.float32)), cv2.dct(m_hl.astype(np.float32))

    mark1_freq, mark2_freq = np.zeros(lengths), np.zeros(lengths)
    # detect
    cnt = 0
    for h in range(0, m_lh_freq.shape[0], 8):
        for w in range(0, m_lh_freq.shape[1], 8):
            # get embed 8 * 8 block
            lh_block, hl_block = m_lh_freq[h:h + 8, w:w + 8], m_hl_freq[h:h + 8, w:w + 8]
            # get zigzag array
            lh_block_list, hl_block_list = zigzag(lh_block), zigzag(hl_block)
            for i in range(need_len):
                mark1_freq[cnt] = lh_block_list[i + 64 - need_len] / 0.1
                mark2_freq[cnt] = hl_block_list[i + 64 - need_len] / 0.1
                cnt += 1
                if cnt >= lengths:
                    break
            if cnt >= lengths:
                break
        if cnt >= lengths:
            break

    mark1 = cv2.idct(np.resize(mark1_freq, (round(lengths ** 0.5), round(lengths ** 0.5))))
    mark2 = cv2.idct(np.resize(mark2_freq, (round(lengths ** 0.5), round(lengths ** 0.5))))
    mark = np.zeros(mark1.shape)
    hei, wei = mark1.shape

    for h in range(hei):
        for w in range(wei):
            if mark1[h][w] >= 255 // 2 or mark2[h][w] >= 255 // 2:
                mark[h][w] = 255
            else:
                mark[h][w] = 0

    plt.imshow(mark)
    plt.show()
    return mark


def main():
    # level and wavelet
    level = 1
    wavelet = 'db1'
    # # image open
    cover = cv2.imread('./images/inputs/lena_gray_256.png', cv2.IMREAD_COLOR)
    cover_b, cover_g, cover_r = cv2.split(cover)
    watermark = cv2.imread('./images/inputs/mark.png', cv2.IMREAD_GRAYSCALE)
    # # embed
    # watermarked_b = embedded(cover_b, watermark, level, wavelet)
    # watermarked = cv2.merge((watermarked_b.astype(np.float32), cover_g.astype(np.float32), cover_r.astype(np.float32)))
    # cv2.imwrite('./images/outputs/merge.png', watermarked)
    # # # detected
    watermarked = cv2.imread('./images/attacks/merge.png', cv2.IMREAD_COLOR)
    watermarked_b, _, _ = cv2.split(watermarked)
    watermark = detect(watermarked_b, level, wavelet, 19, 4900)
    cv2.imwrite('./images/outputs/none_attack.png', watermark)
    watermarked = cv2.imread('./images/attacks/colors.png', cv2.IMREAD_COLOR)
    watermarked_b, _, _ = cv2.split(watermarked)
    watermark = detect(watermarked_b, level, wavelet, 19, 4900)
    cv2.imwrite('./images/outputs/colors_or.png', watermark)
    watermarked = cv2.imread('./images/attacks/colors.png', cv2.IMREAD_COLOR)
    watermarked_b, _, _ = cv2.split(watermarked)
    watermark = detect(watermarked_b, level, wavelet, 19, 4900)
    cv2.imwrite('./images/outputs/colors_or.png', watermark)
    watermarked = cv2.imread('./images/attacks/colors.png', cv2.IMREAD_COLOR)
    watermarked_b, _, _ = cv2.split(watermarked)
    watermark = detect(watermarked_b, level, wavelet, 19, 4900)
    cv2.imwrite('./images/outputs/colors_or.png', watermark)
    watermarked = cv2.imread('./images/attacks/colors.png', cv2.IMREAD_COLOR)
    watermarked_b, _, _ = cv2.split(watermarked)
    watermark = detect(watermarked_b, level, wavelet, 19, 4900)
    cv2.imwrite('./images/outputs/colors_or.png', watermark)
    watermarked = cv2.imread('./images/attacks/colors.png', cv2.IMREAD_COLOR)
    watermarked_b, _, _ = cv2.split(watermarked)
    watermark = detect(watermarked_b, level, wavelet, 19, 4900)
    cv2.imwrite('./images/outputs/colors_or.png', watermark)


if __name__ == '__main__':
    main()

# no crop, image min 256 * 256, embed size 128 * 128 * 2.
