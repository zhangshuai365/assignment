# -*- coding: utf-8 -*-
import cv2

image_address_1 = 'D:/Learning/CV/lesson-01-190630/assignment/Against the Light.jpg'
img_gray = cv2.imread(image_address_1, 0)
cv2.imshow('sunyanzi', img_gray)  # show image
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
print(img_gray)  # show image data as matrix
print(img_gray.dtype)  # show the data's type=>uint8,means unsigned int
print(img_gray.shape)  # h:shape[0], w:shape[1]
