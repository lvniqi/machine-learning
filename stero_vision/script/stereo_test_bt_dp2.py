# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 15:44:01 2016

@author: lvniqi
"""
from common import get_data_set, show_image, save_image, get_compute_cost_bt_d_cpp_func, get_dp_forward_cpp_func_fast, \
    dp_forward_cpp
from stereo_test_dp import StereoVisionBM_DP
from stereo_test_bt_dp import StereoVisionBM_DP_BT
from stereo_test_bm import StereoVisionBM2
import numpy as np
from scipy.ndimage import filters


class StereoVisionBM_DP_BT2(StereoVisionBM_DP_BT):
    def __init__(self, left, right, window_size=13, d_max=10, is_color=False):
        StereoVisionBM_DP.__init__(self, left, right, window_size, d_max, is_color)
        self.compute_cost_d_cpp_func = get_compute_cost_bt_d_cpp_func(self.dll)

    def get_result(self):
        self.dp_forward_cpp_func = get_dp_forward_cpp_func_fast(self.dll)

        # 代价矩阵
        cost = np.zeros((self.column_length, self.d_max), dtype=np.float32)
        # 结果矩阵
        result_array = np.zeros((self.column_length, self.d_max), dtype=np.int16)

        for row in np.arange(self.row_length):
            print "row: ", row
            sad_row = self.sad_left_result[row]
            sad_row = np.array(sad_row, dtype=np.int16)

            for column in range(self.column_length):
                dp_forward_cpp(self.dp_forward_cpp_func, result_array, cost, sad_row, self.column_length,
                               self.d_max, 5)
                # cost_func(sad_row, column)

            # 统计结果 最后一个最小的视差
            min_d_last = np.argmin(cost[-1])
            self.my_result[row][-1] = min_d_last
            min_d = min_d_last
            for column in range(self.column_length - 1, 0, -1):
                self.my_result[row][column - 1] = result_array[column][min_d]
                min_d = result_array[column][min_d]
        return self.my_result.copy()


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
    stereo = StereoVisionBM_DP_BT2(left, right, window_size, d_max)
    stereo.compute_cost()
    stereo.aggregate_cost()
    my_result = stereo.get_result()
    my_result = my_result * 255 / d_max
    data_set['my_result_bt_dp2'] = my_result
    print time.time() - tt
    show_image(data_set)
    save_image(my_result, 'window method BT DP 2')
