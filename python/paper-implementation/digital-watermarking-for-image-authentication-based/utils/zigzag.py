import numpy as np


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


def mapping(arr):
    zig = zigzag(arr)
    hei, wei = arr.shape
    block1, block2, block3, block4 = [], [], [], []
    for i in range(len(zig)):
        if i % 4 == 0:
            block1.append(zig[i])
        if i % 4 == 1:
            block2.append(zig[i])
        if i % 4 == 2:
            block3.append(zig[i])
        if i % 4 == 3:
            block4.append(zig[i])
    block1, block2, block3, block4 = np.array(block1), np.array(block2), np.array(block3), np.array(block4)
    block1, block2 = block1.reshape((int(hei / 2), int(wei / 2))), block2.reshape((int(hei / 2), int(wei / 2)))
    block3, block4 = block3.reshape((int(hei / 2), int(wei / 2))), block4.reshape((int(hei / 2), int(wei / 2)))

    return block1, block2, block3, block4


def imapping(block1, block2, block3, block4):
    zig = []
    block1, block2, block3, block4 = block1.flatten(), block2.flatten(), block3.flatten(), block4.flatten()
    for i in range(len(block1)):
        zig.append(block1[i])
        zig.append(block2[i])
        zig.append(block3[i])
        zig.append(block4[i])

    arr = izigzag(zig)

    return arr
