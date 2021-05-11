import cv2
import numpy as np


# 晚點看一個用偽隨機序列化進行加密的方法當 preprocessing


def embed_w1(cvr):



def embed_w2(cvr):



def main():
    # image open
    cover = cv2.imread('./images/inputs/lena_color.png', cv2.IMREAD_COLOR)
    mark = cv2.imread('./images/inputs/mark.png', cv2.IMREAD_GRAYSCALE)
    # mark = preprocessing(mark)
    b, g, r = cv2.split(cover)
    



if __name__ == '__main__':
    main()
