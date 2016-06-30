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
window_size = 3
(row_length, pixel_length) = left.shape

for row_pos in range(row_length):
    row_start = row_pos - window_size / 2 if row_pos > window_size / 2 else 0
    row_end = window_size / 2 + row_pos + 1 if window_size / 2 + row_pos + 1 <= row_length else row_length
    window_row = left[row_start:row_end]
    # print 'row_start', row_start, 'row_end', row_end, 'row', row_pos
    for pixel_pos in range(pixel_length):
        pixel_start = pixel_pos - window_size / 2 if pixel_pos > window_size / 2 else 0
        pixel_end = window_size / 2 + pixel_pos + 1 if window_size / 2 + pixel_pos + 1 <= pixel_length else pixel_length
        #    print 'pixel_start', pixel_start, 'pixel_end', pixel_end, 'pixel', pixel_pos

        # 得到窗口
        window = window_row[:, pixel_start:pixel_end]
