# -*- coding: utf-8 -*-
import cv2
import random
'''
change the brightness randomly
'''
image_address_1 = 'D:/Learning/CV/lesson-01-190630/assignment/Against the Light.jpg'
img = cv2.imread(image_address_1)


def change_color_randomly(img):
    B, G, R = cv2.split(img)
    def change_brightness(layer):
        layer_rand = random.randint(-50, 50)
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
    for i in [B, G, R]:
        change_brightness(i)
    img_merge = cv2.merge((B, G, R))
    return img_merge


img_random_color = change_color_randomly(img)
cv2.imshow('img', img)
cv2.imshow('img_random_color', img_random_color)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
