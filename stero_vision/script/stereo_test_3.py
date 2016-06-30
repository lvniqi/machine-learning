# -*- coding: utf-8 -*-

from PIL import Image
from common import get_data_set, show_image
import numpy as np
import matplotlib.pyplot as plt

data_set = get_data_set(0)
# get data
left = data_set['left']
right = data_set['right']
result = data_set['result']
my_result = np.zeros(left.shape)


def calculate_diff_naive(pixel_value, row_right, pixel_pos, d_max=10):
    """
    :param pixel_value:左视图像素点
    :param row_right:右视图的这一行
    显而易见 右视图中的点在左视图左侧
    :param pixel_pos:左视图像素点位置
    :param d_max:最大深度
    :return:视差值
    """
    start_pos = (pixel_pos - d_max) if (pixel_pos - d_max) > 0 else 0
    row_right = row_right[start_pos:pixel_pos]
    diff = map(lambda value: abs(value * 1.0 - pixel_value), row_right)
    diff = diff[::-1]  # 逆序 否则是反的
    data_min = 0
    for depth in range(len(diff)):
        if diff[depth] < diff[data_min]:
            data_min = depth
    return data_min


# 扫描左图像素
for row_pos in range(len(left)):
    row_left = left[row_pos]
    row_right = right[row_pos]
    for pixel_pos in range(len(row_left)):
        pixel = row_left[pixel_pos]
        depth = calculate_diff_naive(pixel,row_right,pixel_pos)
        my_result[row_pos][pixel_pos] = depth*255/10
        # pixel = 0
        # print pixel_count

data_set['my_result'] = my_result
show_image(data_set)
if __name__ == '__main__':
    pass