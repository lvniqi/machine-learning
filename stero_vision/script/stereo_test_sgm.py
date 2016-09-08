# -*- coding: utf-8 -*-
"""
Created on Wed Sept 7 15:44:01 2016

@author: lvniqi
"""
from common import get_data_set, show_image, save_image, get_compute_cost_bt_d_cpp_func, get_dp_forward_cpp_func_fast, \
    dp_forward_cpp
from stereo_test_bt_bm import StereoVisionBM_BT
import numpy as np
import ctypes


class StereoVisionBM_SGM(StereoVisionBM_BT):
    def __init__(self, left, right, window_size=13, d_max=10, is_color=False):
        StereoVisionBM_BT.__init__(self, left, right, window_size, d_max, is_color)
        self.compute_cost_d_cpp_func = get_compute_cost_bt_d_cpp_func(self.dll)
        self.cost_result = np.zeros(self.sad_left_result.shape, dtype=np.int32)

    def aggregate_cost(self, is_python=False):
        StereoVisionBM_BT.aggregate_cost(self, is_python)

        sgm_search = self.dll.SGM_search

        sgm_search.restype = ctypes.c_void_p
        sgm_search.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=3),
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=3),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
        ]
        sgm_search(self.cost_result, self.sad_left_result, self.sad_left_result.shape[0],
                    self.sad_left_result.shape[1], self.sad_left_result.shape[2], 0.5)



        self.sad_left_result = self.cost_result.copy()

        '''
        sgm_search(self.cost_result, self.sad_right_result, self.sad_right_result.shape[0],
                   self.sad_right_result.shape[1], self.sad_right_result.shape[2], 0.5)

        self.sad_right_result = self.cost_result.copy()
        '''
        return (self.sad_left_result, self.sad_right_result)


if __name__ == '__main__':
    data_set = get_data_set(0)
    # get data
    left = data_set['left']
    right = data_set['right']
    result = data_set['result']
    import time

    window_size = 7
    d_max = 32
    tt = time.time()
    stereo = StereoVisionBM_SGM(left, right, window_size, d_max)
    stereo.low_texture_detection()
    stereo.compute_cost()
    stereo.aggregate_cost()
    my_result = stereo.get_result()

    diff_result = stereo.left_right_check()
    diff_result = diff_result * (255.0 / d_max / 16)
    data_set['diff_result'] = diff_result

    post_result = stereo.post_processing()
    post_result = post_result * (255.0 / d_max / 16)
    data_set['post_result'] = post_result

    print time.time() - tt

    my_result = my_result * (255.0 / d_max / 16)
    data_set['my_result_sgm'] = my_result

    show_image(data_set)
    save_image(my_result, 'window method sgm')
    save_image(post_result, 'window method sgm post')
