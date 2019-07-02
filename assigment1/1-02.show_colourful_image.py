# -*- coding: utf-8 -*-
import cv2

image_address_1 = 'D:/Learning/CV/lesson-01-190630/assignment/Against the Light.jpg'
img = cv2.imread(image_address_1)
cv2.imshow('sunyanzi', img)  # show image
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
print(img)  # show image data as 3D matrix
print(img.dtype)  # show the data's type=>uint8,means unsigned int
print(img.shape)  # h:shape[0], w:shape[1], chanel:shape[2]
