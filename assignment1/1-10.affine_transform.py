# -*- coding: utf-8 -*-
import cv2
import numpy as np
'''
 Affine transform 仿射变换
 Affine Transformation是一种二维坐标到二维坐标之间的线性变换，保持二维图形的“平直性”
 （译注：straightness，即变换后直线还是直线不会打弯，圆弧还是圆弧）和“平行性”（译注：
 parallelness，其实是指保二维图形间的相对位置关系不变，平行线还是平行线，相交直线的交
 角不变。）。仿射变换可以通过一系列的原子变换的复合来实现，包括：平移（Translation）、
 缩放（Scale）、翻转（Flip）、旋转（Rotation）和剪切（Shear）。
'''
image_address_1 = 'D:/Learning/CV/lesson-01-190630/assignment/Against the Light.jpg'
img = cv2.imread(image_address_1)

rows, cols, ch = img.shape
pts1 = np.float32([[0, 0], [cols - 1, 0], [cols - 1, rows - 1]])  # 变换前三个点的坐标
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.7, rows * 0.3], [cols * 0.8, rows * 0.9]])  # 这三个点变换后的坐标

M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))
cv2.imshow('origin', img)
cv2.imshow('affine', dst)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
