import pywt


level = 1
wavelet = 'db1'

def dwt(img):
    coeffs = pywt.wavedec2(img, level=level, wavelet=wavelet)
    arr, coeffs_slices = pywt.coeffs_to_array(coeffs)

    return arr, coeffs_slices


def idwt(arr, coeffs_slices):
    coeffs = pywt.array_to_coeffs(arr, coeffs_slices, output_format='wavedec2')
    img = pywt.waverec2(coeffs, wavelet=wavelet)

    return img
