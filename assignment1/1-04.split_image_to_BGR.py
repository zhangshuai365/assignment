# -*- coding: utf-8 -*-
import cv2

image_address_1 = 'D:/Learning/CV/lesson-01-190630/assignment/Against the Light.jpg'
img = cv2.imread(image_address_1)
B,G,R = cv2.split(img)
cv2.imshow('B', B)  # show image
cv2.imshow('G', G)  # show image
cv2.imshow('R', R)  # show image
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
print(B)  # show image data as 2D matrix
print(B.dtype)  # show the data's type=>uint8,means unsigned int
print(B.shape)  # h:shape[0], w:shape[1]
