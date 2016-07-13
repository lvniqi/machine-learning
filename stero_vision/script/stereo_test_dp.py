# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 00:06:27 2016

@author: lvniqi
"""

from common import get_data_set, show_image, save_image
from stereo_test_bm import StereoVisionBM2
import numpy as np


class StereoVisionBM_DP(StereoVisionBM2):
    def get_result_forward(self):

        # 代价矩阵
        cost = np.zeros((self.column_length, self.d_max), dtype=np.float32)
        # 结果矩阵
        result_array = np.zeros((self.column_length, self.d_max), dtype=np.int16)

        # 代价计算函数
        def cost_func(sad_row, column):
            # 第一个数据
            if column == 0:
                for d in range(self.d_max):
                    # 代价等于 数据项
                    cost_t = sad_row[column][d]
                    cost[column][d] = cost_t
            # 其余
            else:
                for d in range(self.d_max):
                    # 代价等于 数据项 + 约束项
                    # 约束系数
                    p = 5
                    # 上次的最佳视差
                    min_last = 0
                    # 总体代价结果
                    cost_result = np.inf
                    for last_cost in range(self.d_max):
                        # 上次代价 + 视差偏差*约束系数 + 这次代价
                        cost_t = cost[column - 1][last_cost] + np.abs(d - last_cost) * p + sad_row[column][d]
                        if cost_t < cost_result:
                            cost_result = cost_t
                            min_last = last_cost
                    cost[column][d] = cost_t
                    result_array[column][d] = min_last

        for row in np.arange(self.row_length):
            print "row: ", row
            sad_row = self.sad_result[row]
            for column in range(self.column_length):
                cost_func(sad_row, column)

            # 统计结果 最后一个最小的视差
            min_d_last = np.argmin(cost[-1])
            self.my_result[row][-1] = min_d_last
            for column in range(self.column_length - 1, 0, -1):
                self.my_result[row][column - 1] = result_array[column][min_d_last]
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
            for column in np.arange(self.column_length - 2, -1, -1):
                # 初始能量值为无穷大
                last_ed_es = np.inf
                result_d = 0
                for d in np.arange(self.d_max):
                    e_d = sad_row[column][d]
                    # 不同深度 平滑度惩罚 系数3
                    p = 0.1 * self.window_size * self.window_size
                    e_s = p * np.abs(d - last_disparity)
                    if e_d + e_s < last_ed_es:
                        last_ed_es = e_d + e_s
                        result_d = d
                last_disparity = result_d
                self.my_result[row][column] = result_d
        return self.my_result.copy()

    def get_result_down(self):
        for column in np.arange(self.column_length):
            last_disparity = 0
            # 首先 得到第一个像素的深度值
            for d in np.arange(1, self.d_max):
                if self.sad_result[0][column][last_disparity] > self.sad_result[0][column][d]:
                    last_disparity = d
            self.my_result[0][column] = last_disparity
            # 遍历剩下的像素值
            for row in np.arange(1, self.row_length):
                # 初始能量值为无穷大
                last_ed_es = np.inf
                result_d = 0
                for d in np.arange(self.d_max):
                    e_d = self.sad_result[row][column][d]
                    # 不同深度 平滑度惩罚 系数3
                    p = 0.1 * self.window_size * self.window_size
                    e_s = p * np.abs(d - last_disparity)
                    if e_d + e_s < last_ed_es:
                        last_ed_es = e_d + e_s
                        result_d = d
                last_disparity = result_d
                self.my_result[row][column] = result_d
        return self.my_result.copy()

    def get_result_up(self):
        for column in np.arange(self.column_length - 1, -1, -1):
            last_disparity = 0
            # 首先 得到第一个像素的深度值
            for d in np.arange(1, self.d_max):
                if self.sad_result[-1][column][last_disparity] > self.sad_result[-1][column][d]:
                    last_disparity = d
            self.my_result[-1][column] = last_disparity
            # 遍历剩下的像素值
            for row in np.arange(self.row_length - 2, -1, -1):
                # 初始能量值为无穷大
                last_ed_es = np.inf
                result_d = 0
                for d in np.arange(self.d_max):
                    e_d = self.sad_result[row][column][d]
                    # 不同深度 平滑度惩罚 系数3
                    p = 0.1 * self.window_size * self.window_size
                    e_s = p * np.abs(d - last_disparity)
                    if e_d + e_s < last_ed_es:
                        last_ed_es = e_d + e_s
                        result_d = d
                last_disparity = result_d
                self.my_result[row][column] = result_d
        return self.my_result.copy()

    def get_result(self):
        forward = self.get_result_forward()
        reverse = self.get_result_reverse()
        up = self.get_result_up()
        down = self.get_result_down()
        for row in np.arange(self.row_length):
            for column in np.arange(self.column_length):
                self.my_result[row][column] = np.median(
                    (reverse[row][column], forward[row][column], down[row][column], up[row][column]))
        return self.my_result.copy()


if __name__ == '__main__':
    data_set = get_data_set(0)
    # get data
    left = data_set['left']
    right = data_set['right']
    result = data_set['result']
    import time

    window_size = 5
    d_max = 15
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
