# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import sys


def medianBlur(img, kernel, padding_way):
    # img & kernel is List of List; padding_way a string:"REPLICA" or "ZERO"
    padding_fun = replicaPadding if padding_way.upper() == "REPLICA" else zerosPadding

    rows, cols, *_ = img.shape
    krows, kcols, *_ = kernel.shape
    ksize = krows * krows
    row_offset = -(krows // 2)
    col_offset = -(kcols // 2)

    kernel_list = np.zeros(ksize, dtype=img.dtype)
    result = np.zeros(img.shape, dtype=img.dtype)

    for row in range(rows):
        for col in range(cols):
            row_start = row + row_offset
            col_start = col + col_offset
            idx = 0
            for r in range(row_start, row_start + krows):
                for c in range(col_start, col_start + kcols):
                    kernel_list[idx] = padding_fun(img, r, c)
                    idx = idx + 1
            kernel_list.sort()
            result[row][col] = kernel_list[ksize // 2]
    return result


def replicaPadding(img, r, c):
    rows, cols, *_ = img.shape
    row = min(max(r, 0), rows - 1)
    col = min(max(c, 0), cols - 1)
    return img[row][col]

'''
def replicaPadding(img, r, c):
    rows, cols, *_ = img.shape
    if r < 0:
        row = 0
    elif r >= rows:
        row = rows - 1
    else:
        row = r
    if c < 0:
        col = 0
    elif c >= cols:
        col = cols - 1
    else:
        col = c
    return img[row][col]
'''


def zerosPadding(img, r, c):
    rows, cols, *_ = img.shape
    if r < 0 or c < 0 or r >= rows or c >= cols:
        return 0
    else:
        return img[r][c]


if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not file_path:
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, "Against the Light.jpg")
    img = cv2.imread(file_path, 0)
    kernel_shape = (3, 3)
    kernel = np.zeros(kernel_shape, )

    print("img:{},kernel:{}".format(img.shape, kernel.shape))
    cv2.imshow('original', img)

    blured_img = medianBlur(img, kernel, 'REPLICA')
    cv2.imshow('blured REPLICA', blured_img)
    blured_img = medianBlur(img, kernel, 'ZERO')
    cv2.imshow('blured ZERO', blured_img)

    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()
