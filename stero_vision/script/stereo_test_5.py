# -*- coding: utf-8 -*-
"""
Created on Fri Jul 01 00:04:05 2016

@author: lvniqi
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
(row_length, pixel_length) = left.shape


def window_diff(window1, window2):
    (row_length_1, pixel_length_1) = window1.shape
    (row_length_2, pixel_length_2) = window2.shape

    diff = 0
    for row_pos in range(min(row_length_1, row_length_2)):
        for pixel_pos in range(min(pixel_length_1, pixel_length_2)):
            diff += abs(window1[row_pos][pixel_pos] - window2[row_pos][pixel_pos])
    return diff


def calculate_diff_window(left_window, right_rows, pixel_pos, d_max=10):
    start_pos = (pixel_pos - d_max) if (pixel_pos - d_max) > 0 else 0
    diff = []
    for pos in range(start_pos, pixel_pos):
        right_window = get_window(right_rows, pos, window_size)
        d = window_diff(right_window, left_window)
        diff.append(d)
    diff = diff[::-1]
    data_min = 0
    for depth in range(len(diff)):
        if diff[data_min] == 0:
            break
        if diff[depth] < diff[data_min]:
            data_min = depth
    return data_min


def get_rows(image, row_pos, window_size):
    row_start = row_pos - window_size / 2 if row_pos > window_size / 2 else 0
    row_end = window_size / 2 + row_pos + 1 if window_size / 2 + row_pos + 1 <= row_length else row_length
    return image[row_start:row_end]


def get_window(rows, pixel_pos, window_size):
    pixel_start = pixel_pos - window_size / 2 if pixel_pos > window_size / 2 else 0
    pixel_end = window_size / 2 + pixel_pos + 1 if window_size / 2 + pixel_pos + 1 <= pixel_length else pixel_length
    return rows[:, pixel_start:pixel_end]


for row_pos in range(row_length):
    left_rows = get_rows(left, row_pos, window_size)
    right_rows = get_rows(right, row_pos, window_size)
    # print 'row_start', row_start, 'row_end', row_end, 'row', row_pos
    for pixel_pos in range(pixel_length):
        pixel_start = pixel_pos - window_size / 2 if pixel_pos > window_size / 2 else 0
        pixel_end = window_size / 2 + pixel_pos + 1 if window_size / 2 + pixel_pos + 1 <= pixel_length else pixel_length
        #    print 'pixel_start', pixel_start, 'pixel_end', pixel_end, 'pixel', pixel_pos
        # 得到窗口
        left_window = get_window(left_rows, pixel_pos, window_size)
        d_max = 10
        depth = calculate_diff_window(left_window, right_rows, pixel_pos,d_max)
        my_result[row_pos][pixel_pos] = depth * 255 / 48

data_set['my_result_4'] = my_result
save_image(my_result, 'window method')
show_image(data_set)
