# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 00:06:27 2016

@author: lvniqi
"""

from common import get_data_set, show_image, save_image
from stereo_test_7 import StereoVisionBM2
import numpy as np


class StereoVisionBM_DP(StereoVisionBM2):
    def get_result_forward(self):
        for row in np.arange(self.row_length):
            sad_row = self.sad_result[row]
            last_disparity = 0
            # 首先 得到第一个像素的深度值
            for d in np.arange(1, self.d_max):
                if sad_row[0][last_disparity] > sad_row[0][d]:
                    last_disparity = d
            self.my_result[row][0] = last_disparity
            # 遍历剩下的像素值
            for column in np.arange(1, self.column_length):
                # 初始能量值为无穷大
                last_ed_es = np.inf
                result_d = 0
                for d in np.arange(self.d_max):
                    E_d = sad_row[column][d]
                    # 不同深度 平滑度惩罚 系数3
                    p = 2
                    E_s = p * np.abs(d - last_disparity)
                    # 上方的视差
                    E_s_2 = 0 if row == 0 else p * np.abs(d - self.my_result[row - 1][column])
                    if E_d + E_s + E_s_2 < last_ed_es:
                        last_ed_es = E_d + E_s + E_s_2
                        result_d = d
                last_disparity = result_d
                self.my_result[row][column] = result_d
        return self.my_result.copy()

    def get_result_reverse(self):
        for row in np.arange(self.row_length - 1, -1, -1):
            sad_row = self.sad_result[row]
            last_disparity = 0
            # 首先 得到第一个像素的深度值
            for d in np.arange(1, self.d_max):
                if sad_row[-1][last_disparity] > sad_row[-1][d]:
                    last_disparity = d
            self.my_result[row][-1] = last_disparity
            # 遍历剩下的像素值
            for column in np.arange(self.column_length - 1, 0, -1):
                # 初始能量值为无穷大
                last_ed_es = np.inf
                result_d = 0
                for d in np.arange(self.d_max):
                    e_d = sad_row[column][d]
                    # 不同深度 平滑度惩罚 系数3
                    p = 2
                    e_s = p * np.abs(d - last_disparity)
                    # 下侧的视差
                    e_s_2 = 0 if row == self.row_length - 1 else p * np.abs(d - self.my_result[row + 1][column])
                    if e_d + e_s + e_s_2 < last_ed_es:
                        last_ed_es = e_d + e_s + e_s_2
                        result_d = d
                last_disparity = result_d
                self.my_result[row][column] = result_d
        return self.my_result.copy()

    def get_result(self):
        reverse = self.get_result_reverse()
        forward = self.get_result_forward()
        return (reverse + forward) / 2


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
    stereo = StereoVisionBM_DP(left, right, window_size, d_max)
    stereo.compute_cost()
    stereo.aggregate_cost()
    my_result = stereo.get_result()
    my_result = my_result * 255 / d_max
    data_set['my_result_dp'] = my_result
    show_image(data_set)
    print time.time() - tt
    save_image(my_result, 'window method DP')
