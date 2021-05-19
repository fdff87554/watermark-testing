import numpy as np

from utils.zigzag import zigzag, izigzag


def test():
    arr = [[0, 1, 2, 3],
           [4, 5, 6, 7],
           [8, 9, 10, 11],
           [12, 13, 14, 15]]
    print(zigzag(np.array(arr)))
    print(izigzag(zigzag(np.array(arr))))


if __name__ == '__main__':
    test()
