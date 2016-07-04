# -*- coding: utf-8 -*-
"""
Created on Mon Jul 04 15:25:30 2016

@author: WZ5040
"""

from common import get_data_set, show_image, save_image
import numpy as np

data_set = get_data_set(0)
# get data
left = data_set['left']
right = data_set['right']
result = data_set['result']
my_result = np.zeros(left.shape)
# 必须为奇数
window_size = 13

sad_size = [window_size, ]
sad_size.extend(left.shape)
# sad 计算结果
sad_result = np.zeros(sad_size)
# sad 检测标志
sad_flag = np.zeros(left.shape, dtype=np.bool)

(row_length, pixel_length) = left.shape


def make_border(image):
    new_image = image

    top = image[0]
    bottum = image[-1]
    top = np.tile(top, (window_size / 2, 1))
    bottum = np.tile(bottum, (window_size / 2, 1))
    new_image = np.row_stack((top, new_image))
    new_image = np.row_stack((new_image, bottum))

    left = new_image[:, 0:1]
    right = new_image[:, -2:-1]

    left = np.tile(left, (1, window_size + window_size / 2))
    right = np.tile(right, (1, window_size + window_size / 2))
    # left = np.tile(left, (1, window_size / 2))
    # right = np.tile(right, (1, window_size / 2))

    new_image = np.column_stack((left, new_image))
    new_image = np.column_stack((new_image, right))

    return new_image


left_new = make_border(left)
right_new = make_border(right)


def get_sad(left, right, x, y, d):
    if x > 0 and y > 0:
        pass
    window_left = left[x - window_size / 2:x + window_size / 2 + 1,
                  y - window_size / 2:y + window_size / 2 + 1]
    window_right = right[x - window_size / 2:x + window_size / 2 + 1,
                   y - window_size / 2 - d:y + window_size / 2 + 1 - d]
    return np.sum(np.absolute(window_left - window_right))


if __name__ == '__main__':
    left_new = make_border(left)
    right_new = make_border(right)
    for d in np.arange(window_size):
        for x in np.arange(row_length):
            for y in np.arange(pixel_length):
                sad = get_sad(left_new, right_new, x + window_size / 2, y + window_size + window_size / 2, d)
                sad_result[d][x][y] = sad

    for x in np.arange(row_length):
        for y in np.arange(pixel_length):
            min_sad = 0
            for d in np.arange(1, window_size):
                if sad_result[min_sad][x][y] > sad_result[d][x][y]:
                    min_sad = d
            my_result[x][y] = min_sad * 255 / 13
    data_set['my_result_6'] = my_result
    save_image(my_result, 'window method 6')
    show_image(data_set)
