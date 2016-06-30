# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 17:59:37 2016

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


def calculate_min_max(image):
    """
    根据bt算法 获取临近区域max、min
    :param image:输入图像
    :return:(min,max)
    """
    size = image.shape
    row_length = size[0]
    pixel_length = size[1]
    image_max = np.zeros(size)
    image_min = np.zeros(size)
    for row_pos in range(row_length):
        row = image[row_pos]
        min_row = image_min[row_pos]
        max_row = image_max[row_pos]
        pixel_cache_1 = row[0]
        pixel_cache_2 = row[1]
        for pixel_pos in range(pixel_length):
            pixel = row[pixel_pos]
            # 边界检查
            if pixel_pos < pixel_length - 1:
                pixel_cache_2 = (pixel + row[pixel_pos + 1]) / 2
            else:
                pixel_cache_2 = pixel
            min_row[pixel_pos] = min(pixel_cache_1, pixel, pixel_cache_2)
            max_row[pixel_pos] = max(pixel_cache_1, pixel, pixel_cache_2)
            pixel_cache_1 = pixel_cache_2
    return image_min, image_max


def calculate_diff_bt(left, right, pixel_pos, d_max=10):
    (row_left, left_min, left_max) = left
    (row_right, right_min, right_max) = right
    start_pos = (pixel_pos - d_max) if (pixel_pos - d_max) > 0 else 0
    diff = []
    for pos in range(start_pos, pixel_pos):
        diff_l = max(0, row_right[pos] - left_max[pixel_pos], left_min[pixel_pos] - row_right[pos])
        diff_r = max(0, row_left[pixel_pos] - right_max[pos], right_min[pos] - row_left[pixel_pos])
        diff.append(min(diff_l, diff_r))
    diff = diff[::-1]
    data_min = 0
    for depth in range(len(diff)):
        if diff[data_min] == 0:
            break
        if diff[depth] < diff[data_min]:
            data_min = depth
    return data_min


(left_min, left_max) = calculate_min_max(left)
(right_min, right_max) = calculate_min_max(right)

# 扫描像素
for row_pos in range(len(left)):
    # 读入图像
    row_left = left[row_pos]
    row_right = right[row_pos]
    # 读入行
    row_left_min = left_min[row_pos]
    row_left_max = left_max[row_pos]
    row_right_min = right_min[row_pos]
    row_right_max = right_max[row_pos]
    for pixel_pos in range(len(row_left)):
        depth = calculate_diff_bt((row_left, row_left_min, row_left_max), (row_right, row_right_min, row_right_max),
                                  pixel_pos)
        my_result[row_pos][pixel_pos] = depth * 255 / 10

data_set['my_result_4'] = my_result
save_image(my_result, 'pixel bt method')
show_image(data_set)
if __name__ == '__main__':
    # calculate_min_max(left)
    pass
