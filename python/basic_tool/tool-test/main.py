from PIL import Image
import cv2 as cv
import numpy as np

# Open-cv imread, imwrite
cover = cv.imread('./images/input/cover_gray.png', 0)
dft = np.fft.fft2(cover)
idft = np.fft.ifft2(dft).real

cv.imwrite('./images/output/test_cv2.png', idft)

normal = cv.imread('./images/input/cover_gray.png', 0)
output = cv.imread('./images/output/test_cv2.png', 0)

print(normal - output)

# PIL Image.open & Image.save
with Image.open('./images/input/cover_gray.png') as cover:
    cover = cover.convert('L')

dft = np.fft.fft2(cover)
idft = np.fft.ifft2(dft).real

cover = Image.fromarray(idft.astype(np.uint8))
cover.save('./images/output/test_pil.png')

normal = Image.open('./images/input/cover_gray.png')
output = Image.open('./images/output/test_pil.png')
normal = np.asarray(normal)
output = np.asarray(output)
print(normal - output)

