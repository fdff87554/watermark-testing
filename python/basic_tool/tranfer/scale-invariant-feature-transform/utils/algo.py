# import cv2
#
#
# def sift(img):
#     sifts = cv2.SIFT()
#     kp = sifts.detect(img, None)
#     print(kp.shape, len(kp))
#     pic = cv2.drawKeypoints(img, kp)
#
#     return pic
