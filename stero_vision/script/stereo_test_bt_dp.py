# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 15:44:01 2016

@author: lvniqi
"""
from common import get_data_set, show_image, save_image, get_compute_cost_bt_d_cpp_func
from stereo_test_dp import StereoVisionBM_DP
from stereo_test_bm import StereoVisionBM2
import numpy as np
from scipy.ndimage import filters


class StereoVisionBM_BT(StereoVisionBM2):
    def __init__(self, left, right, window_size=13, d_max=10, is_color=False):
        StereoVisionBM2.__init__(self, left, right, window_size, d_max, is_color)
        self.compute_cost_d_cpp_func = get_compute_cost_bt_d_cpp_func(self.dll)


class StereoVisionBM_DP_BT(StereoVisionBM_DP):
    def __init__(self, left, right, window_size=13, d_max=10, is_color=False):
        StereoVisionBM_DP.__init__(self, left, right, window_size, d_max, is_color)
        self.compute_cost_d_cpp_func = get_compute_cost_bt_d_cpp_func(self.dll)

    '''def aggregate_cost(self):
        for d in range(self.d_max):
            diff = self.diff[d]
            for row in np.arange(self.row_length):
                top = (row - self.window_size / 2) if (row - self.window_size / 2) > 0 else 0
                bottom = (row + self.window_size / 2 + 1) if \
                    (row + self.window_size / 2 + 1) <= self.row_length else self.row_length
                for column in np.arange(self.column_length):
                    left = (column - self.window_size / 2) if (column - self.window_size / 2) > 0 else 0
                    right = (column + self.window_size / 2 + 1) if \
                        (column + self.window_size / 2 + 1) <= self.column_length else self.column_length

                    diff_window = diff[top:bottom, left:right]
                    # 加入权值窗口
                    window = self.left[top:bottom, left:right]

                    sad = np.sum(diff_window * self.get_window(window, window_size))
                    self.sad_result[row][column][d] = sad / (window_size / 2)
        return self.sad_result
    '''

    # 使用python编写的BT代价计算程序 低速但算法相同
    def compute_cost_bt_d_python(self, d):
        cost = np.zeros(self.left.shape)
        for row in np.arange(self.row_length):
            for column in np.arange(self.column_length):
                left_pixel = self.left[row][column]
                right_pixel_l = self.right_extend[row][column + self.d_max - d - 1]
                right_pixel = self.right_extend[row][column + self.d_max - d]
                right_pixel_r = self.right_extend[row][column + self.d_max - d + 1]

                right_pixel_l = (right_pixel_l + right_pixel) / 2
                right_pixel_r = (right_pixel_r + right_pixel) / 2
                if not self.is_color:
                    right_pixel_min = np.min((right_pixel_l, right_pixel, right_pixel_r))
                    right_pixel_max = np.max((right_pixel_l, right_pixel, right_pixel_r))
                    cost[row][column] = np.max((0, left_pixel - right_pixel_max, right_pixel_min - left_pixel))
                else:
                    raise AttributeError
        return cost


if __name__ == '__main__':
    data_set = get_data_set(0)
    # get data
    left = data_set['left']
    right = data_set['right']
    result = data_set['result']
    import time

    window_size = 5
    d_max = 20
    tt = time.time()
    stereo = StereoVisionBM_DP_BT(left, right, window_size, d_max)
    stereo.compute_cost()
    stereo.aggregate_cost()
    my_result = stereo.get_result()
    my_result = my_result * 255 / d_max
    data_set['my_result_bt_dp'] = my_result
    show_image(data_set)
    print time.time() - tt
    save_image(my_result, 'window method BT DP')
