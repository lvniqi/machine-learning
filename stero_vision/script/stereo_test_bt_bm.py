# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:58:40 2016

@author: lvniqi
"""

from common import get_data_set, show_image, save_image, get_compute_cost_bt_d_cpp_func
from stereo_test_dp import StereoVisionBM_DP
from stereo_test_bm import StereoVisionBM2
import numpy as np


class StereoVisionBM_BT(StereoVisionBM2):
    def __init__(self, left, right, window_size=13, d_max=10, is_color=False):
        StereoVisionBM2.__init__(self, left, right, window_size, d_max, is_color)
        self.compute_cost_d_cpp_func = get_compute_cost_bt_d_cpp_func(self.dll)


if __name__ == '__main__':
    data_set = get_data_set(0)
    # get data
    left = data_set['left']
    right = data_set['right']
    result = data_set['result']
    import time

    window_size = 11
    d_max = 20
    tt = time.time()
    stereo = StereoVisionBM_BT(left, right, window_size, d_max)
    stereo.compute_cost()
    stereo.aggregate_cost()
    my_result = stereo.get_result()
    my_result = my_result * (255.0 / d_max / 16)
    data_set['my_result_bt_bm'] = my_result

    diff_result = stereo.left_right_check()
    diff_result = diff_result * (255.0 / d_max / 16)
    data_set['diff_result'] = diff_result

    post_result = stereo.post_processing()
    post_result = post_result * (255.0 / d_max / 16)
    data_set['post_result'] = post_result

    print time.time() - tt

    show_image(data_set)
    save_image(my_result, 'window method BT BM')
    save_image(diff_result, 'window method BT BM diff_result')
    save_image(post_result, 'window method BT BM post_result')
