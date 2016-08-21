# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 23:22:14 2016

@author: lvniqi
"""

from common import get_data_set, show_image, save_image, get_census_cpp_func, \
    get_census_cpp, get_compute_cost_census_d_cpp_func, compute_cost_d_cpp
from stereo_test_dp import StereoVisionBM_DP
from stereo_test_bm import StereoVisionBM2
import numpy as np


class StereoVisionCensus_BT(StereoVisionBM2):
    def __init__(self, left, right, window_size=13, d_max=10, census_size=3, is_color=False):
        StereoVisionBM2.__init__(self, left, right, window_size, d_max, is_color)
        self.census_size = census_size
        self.compute_cost_d_cpp_func = get_compute_cost_census_d_cpp_func(self.dll)

    def compute_cost(self):
        self.get_census_cpp_func = get_census_cpp_func(self.dll)

        # print 'calcute left census'
        self.left_census = get_census_cpp(self.get_census_cpp_func, self.left, self.census_size)
        # self.left_census = self.get_census(self.left, self.census_size)
        # print 'calcute right census'
        right_census = get_census_cpp(self.get_census_cpp_func, self.right, self.census_size)
        # right_census = self.get_census(self.right, self.census_size)
        self.right_census_extend = self.make_border_rgb(right_census, self.d_max)

        for d in range(self.d_max):
            # print 'compute_cost_d:', d
            self.left_diff[d] = self.gaussian_filter(self.compute_cost_d(d))

        return self.left_diff.copy()

    def compute_cost_d(self, d):
        return compute_cost_d_cpp(self.compute_cost_d_cpp_func, self.left_census,
                                  self.right_census_extend[:,
                                  self.d_max - d:self.column_length + self.d_max - d].copy())
        # python版本
        """result = np.zeros((self.row_length, self.column_length), dtype=np.int16)
        right_census = self.right_census_extend[:, self.d_max - d:self.column_length + self.d_max - d, :]
        for row in np.arange(self.row_length):
            for column in np.arange(self.column_length):
                left = self.left_census[row][column]
                right = right_census[row][column]
                result[row][column] = self.get_hamming_distance(left, right)
        return result
        """

    @staticmethod
    def get_census(image, window_size):
        """
        python版本census计算
        :param image:图像
        :param window_size:census窗口
        :return:census结果
        """
        (row_length, column_length) = image.shape
        # 计算结果
        result = np.zeros((row_length, column_length, window_size * window_size), dtype=np.bool)
        for row in np.arange(row_length):
            for column in np.arange(column_length):
                # census
                census_array = np.zeros((window_size * window_size), dtype=np.bool)
                # 被测值
                mid_value = image[row][column]
                left_pos = column - window_size / 2 if column - window_size / 2 >= 0 else 0
                right_pos = column + window_size / 2 + 1 if column + window_size / 2 + 1 <= column_length else column_length
                top_pos = row - window_size / 2 if row - window_size / 2 >= 0 else 0
                bottom_pos = row + window_size / 2 + 1 if row + window_size / 2 + 1 <= row_length else row_length
                window = image[top_pos:bottom_pos, left_pos:right_pos]
                (row_length_t, column_length_t) = window.shape
                for row_t in np.arange(row_length_t):
                    for column_t in np.arange(column_length_t):
                        if window[row_t][column_t] > mid_value:
                            census_array[row_t * column_length_t + column_t] = True
                        else:
                            census_array[row_t * column_length_t + column_t] = False
                result[row][column] = census_array
        return result

    @staticmethod
    def get_hamming_distance(census1, census2):
        """
        python版本 得到汉明距离
        :param census1:census序列1
        :param census2:census序列2
        :return:
        """
        diff = census1 - census2
        diff_len = np.nonzero(diff)[0].shape[0]
        return diff_len


if __name__ == '__main__':
    data_set = get_data_set(0)
    # get data
    left = data_set['left']
    right = data_set['right']
    result = data_set['result']
    import time

    window_size = 7
    d_max = 20
    tt = time.time()
    stereo = StereoVisionCensus_BT(left, right, window_size, d_max)
    low_texture = stereo.low_texture_detection()
    stereo.compute_cost()
    stereo.aggregate_cost()
    my_result = stereo.get_result()

    diff_result = stereo.left_right_check()

    post_result = stereo.post_processing()
    #post_result2 = stereo.fix_low_texture()

    print "use time:", time.time() - tt
    low_texture = stereo.low_texture_detection()
    data_set['low_texture'] = low_texture

    my_result = my_result * (255.0 / d_max / 16)

    data_set['my_result_ensus_bm'] = my_result

    diff_result = diff_result * (255.0 / d_max / 16)
    data_set['diff_result'] = diff_result

    post_result = post_result * (255.0 / d_max / 16)
    data_set['post_result'] = post_result

    #post_result2 = post_result2 * (255.0 / d_max / 16)
    #data_set['post_result2'] = post_result2

    show_image(data_set)
    save_image(my_result, 'window method Census BM')
    save_image(diff_result, 'window method Census BM diff_result')
    save_image(post_result, 'window method Census BM post_result')
    #save_image(post_result2, 'window method Census BM post_result2')
