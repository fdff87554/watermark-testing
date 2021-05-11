import numpy as np


# svd input a should be an n*n matrix
def svd(a):
    u, s, vh = np.linalg.svd(a)

    return u, s, vh


def isvd(u, s, vh):
    a = np.dot(u[:, :len(s[0])] * s, vh)

    return a
