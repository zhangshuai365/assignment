# -*- coding: utf-8 -*-
import cv2
import random

image_address_1 = 'D:/Learning/CV/lesson-01-190630/assignment/Against the Light.jpg'
img = cv2.imread(image_address_1)

angle = random.randint(-90,90)
scale_ratio = 1
M = cv2.getRotationMatrix2D((img.shape[1]*0.5,img.shape[0]*0.5),angle,scale_ratio)

img_rotate = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
cv2.imshow('rotate',img_rotate)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
M[0][1] = M[1][2] = 0
print (M)
img_rotate2 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow('rotate',img_rotate2)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
