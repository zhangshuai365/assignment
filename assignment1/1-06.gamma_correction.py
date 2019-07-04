# -*- coding: utf-8 -*-
import cv2
import numpy as np

image_address_1 = 'D:/Learning/CV/lesson-01-190630/assignment/Against the Light.jpg'
img = cv2.imread(image_address_1)


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append((i/255)**invGamma*255)
    table = np.array(table).astype('uint8')
    return cv2.LUT(image, table)


img_brighter = adjust_gamma(img, 1.5)
img_darker = adjust_gamma(img, 0.6)
cv2.imshow('origin', img)
cv2.imshow('brighter', img_brighter)
cv2.imshow('darker', img_darker)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()