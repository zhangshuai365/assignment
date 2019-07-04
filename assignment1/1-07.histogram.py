# -*- coding: utf-8 -*-
import cv2
from matplotlib import pyplot as plt

image_address_1 = 'D:/Learning/CV/lesson-01-190630/assignment/Against the Light.jpg'
img = cv2.imread(image_address_1)
#scale
scale_ratio = 0.8
img_smaller = cv2.resize(img,(int(img.shape[1]*scale_ratio),int(img.shape[0]*scale_ratio)))
plt.hist(img_smaller.flatten(),256,[100,256],color='red')
plt.show()

img_yuv = cv2.cvtColor(img_smaller,cv2.COLOR_BGR2YUV)
print(img_yuv)
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0]) # y: luminance(明亮度), u&v: 色度饱和度
print(img_yuv)
img_output = cv2.cvtColor(img_yuv,cv2.COLOR_YUV2BGR)

cv2.imshow('img_smaller',img_smaller)
cv2.imshow('img_output',img_output)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

