# -*- coding: utf-8 -*-
import cv2
import random
import numpy as np

# 读取原始图片
image_address_1 = 'D:/Learning/CV/lesson-01-190630/assignment/Against the Light.jpg'
img = cv2.imread(image_address_1)

def img_crop(img, crop_area=0):
    # 返回裁剪后图片
    height, width, channels = img.shape
    # 区域0：左上，区域1：右上，区域2：左下，区域3：右下，默认情况及其他输入为左上
    if crop_area == 0:
        h1, w1 = 0, 0
    elif crop_area == 1:
        h1, w1 = 0, width // 2
    elif crop_area == 2:
        h1, w1 = height // 2, 0
    elif crop_area == 3:
        h1, w1 = height // 2, width // 2
    else: h1, w1 = 0, 0
    h2 = h1 + int(height * 0.5)
    w2 = w1 + int(width * 0.5)
    img_crop = img[h1:h2, w1:w2, :]
    return img_crop


def img_color_shift(img,brightness=0):
    # 返回颜色变换后的图片
    B, G, R = cv2.split(img)  # 将图片的三个图层分开
    def change_brightness(layer):
        # 改变单个图层的亮度，1为变亮，-1为变暗，0或其他输入则亮度不变
        if brightness == 1:
            layer_rand = random.randint(30, 50)
        elif brightness == -1:
            layer_rand = random.randint(-50, -30)
        else:
            layer_rand = 0

        if layer_rand == 0:
            pass
        elif layer_rand > 0:
            lim = 255-layer_rand
            layer[layer > lim] = 255  # 布尔索引
            layer[layer <= lim] = (layer[layer <= lim] + layer_rand).astype(img.dtype)

        elif layer_rand < 0:
            lim = - layer_rand
            layer[layer < lim] = 0
            layer[layer >= lim] = (layer[layer >= lim] + layer_rand).astype(img.dtype)
    #  将三个图层均进行亮度变化
    for i in [B, G, R]:
        change_brightness(i)
    img_merge = cv2.merge((B, G, R))  # 将三个图层组合
    return img_merge


def img_rotation(img, angle=30):
    # 返回旋转后的图片，默认角度为30度
    M = cv2.getRotationMatrix2D((int(img.shape[1]*0.5), int(img.shape[0]*0.5)), angle, 1)  # 图片旋转矩阵
    img_rotate = cv2.warpAffine(img, M, (img.shape[1],img.shape[0])) # 旋转后的图片
    return img_rotate

def img_perspective(img):
    # 返回投影变换后的图片
    height, width, channels = img.shape
    random_margin = 30 # random_margin 用于调节变换幅度，越大则变换幅度越大
    # 随机生成变换前的四个点的坐标
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)
    # 随机生成变换后的四个点的坐标
    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])  # 变换前的四个点坐标
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])  # 这四个点变换后的坐标
    M_warp = cv2.getPerspectiveTransform(pts1, pts2) # 变换矩阵
    img_warp = cv2.warpPerspective(img, M_warp, (width, height)) # 变换后的图片
    return img_warp


def img_complex_transform(img, crop_area, angle, brightness):
    # 依次进行，剪切，旋转，颜色（亮度）变换，投影变换
    img = img_crop(img, crop_area)
    img = img_rotation(img, angle)
    img = img_color_shift(img, brightness)
    img = img_perspective(img)
    return img


img = img_complex_transform(img, 2, 60, 1)  # 生成变换后的图片
cv2.imshow('img complex transform', img)  # 将图片显示
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()