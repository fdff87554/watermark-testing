import numpy as np


def logistic_map(n, r, x_0):
    x = [x_0]
    for i in range(1, n):
        x.append(r * x[i - 1] * (1 - x[i - 1]))

    return x


def generator(n, r, x_0):
    lm = logistic_map(n, r, x_0)
    for i in range(len(lm)):
        lm[i] = np.ceil(lm[i] * 8)

    return lm


def main():
    n, r, x_0 = 100, 4, 0.3
    lm = generator(n, r, x_0)
    print(lm)


if __name__ == '__main__':
    main()
