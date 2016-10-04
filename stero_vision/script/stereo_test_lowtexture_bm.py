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


class StereoVisionLowTexture_BT(StereoVisionBM2):
    def __init__(self, left, right, window_size=13, d_max=10, census_size=3, is_color=False):
        StereoVisionBM2.__init__(self, left, right, window_size, d_max, is_color)

    def get_low_texture_cost_l_func(self):
        import ctypes
        get_low_texture_cost_l = self.dll.get_low_texture_cost_l

        get_low_texture_cost_l.restype = ctypes.c_void_p
        get_low_texture_cost_l.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.int16, ndim=2),
            np.ctypeslib.ndpointer(dtype=np.int16, ndim=2),
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
        ]
        return get_low_texture_cost_l

    def get_low_texture_cost_r_func(self):
        import ctypes
        get_low_texture_cost_r = self.dll.get_low_texture_cost_r

        get_low_texture_cost_r.restype = ctypes.c_void_p
        get_low_texture_cost_r.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.int16, ndim=2),
            np.ctypeslib.ndpointer(dtype=np.int16, ndim=2),
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
        ]
        return get_low_texture_cost_r

    def get_low_cost_l(self):
        self.get_low_texture_cost_l_func_t = self.get_low_texture_cost_l_func()
        shapes = np.array(self.left.shape, dtype=np.int32)
        self.low_texture_cost_ll = np.zeros(self.left.shape, dtype=np.int16)
        self.low_texture_cost_lr = np.zeros(self.left.shape, dtype=np.int16)
        self.get_low_texture_cost_l_func_t(self.low_texture_cost_ll, self.low_texture_column, shapes)
        self.get_low_texture_cost_l_func_t(self.low_texture_cost_lr, self.low_texture_column_r, shapes)
        return (self.low_texture_cost_ll.copy(), self.low_texture_cost_lr.copy())

    def compute_cost(self):
        '''self.get_census_cpp_func = get_census_cpp_func(self.dll)

        # print 'calcute left census'
        self.left_census = get_census_cpp(self.get_census_cpp_func, self.left, self.census_size)
        # self.left_census = self.get_census(self.left, self.census_size)
        # print 'calcute right census'
        right_census = get_census_cpp(self.get_census_cpp_func, self.right, self.census_size)
        # right_census = self.get_census(self.right, self.census_size)
        self.right_census_extend = self.make_border_rgb(right_census, self.d_max)
        '''
        self.get_low_cost_l()
        self.right_low_cost_extend = self.make_border(self.low_texture_cost_lr, self.d_max)
        for d in range(self.d_max):
            # print 'compute_cost_d:', d
            self.left_diff[d] = self.gaussian_filter(self.compute_cost_d(d))

        return self.left_diff.copy()

    def compute_cost_d(self, d):
        return compute_cost_d_cpp(self.compute_cost_d_cpp_func, self.low_texture_cost_ll,
                                  self.right_low_cost_extend[:,
                                  self.d_max - d:self.column_length + self.d_max - d].copy())+StereoVisionBM2.compute_cost_d(self,d)
        #return StereoVisionBM2.compute_cost_d(self,d)


if __name__ == '__main__':
    data_set = get_data_set(8)
    # get data
    left = data_set['left']
    right = data_set['right']
    result = data_set['result']
    import time

    window_size = 21
    d_max = 64
    tt = time.time()
    stereo = StereoVisionLowTexture_BT(left, right, window_size, d_max)
    low_texture = stereo.low_texture_detection(0.2)[0]

    (left_c, right_c) = stereo.get_low_cost_l()

    stereo.compute_cost()
    stereo.aggregate_cost()
    my_result = stereo.get_result()

    diff_result = stereo.left_right_check()

    post_result = stereo.post_processing()
    # post_result2 = stereo.fix_low_texture()

    print "use time:", time.time() - tt
    data_set['low_texture'] = low_texture
    data_set['left_c'] = left_c
    data_set['right_c'] = right_c
    my_result = my_result * (255.0 / d_max / 16)

    data_set['my_result_ensus_bm'] = my_result

    diff_result = diff_result * (255.0 / d_max / 16)
    data_set['diff_result'] = diff_result

    post_result = post_result * (255.0 / d_max / 16)
    data_set['post_result'] = post_result

    # post_result2 = post_result2 * (255.0 / d_max / 16)
    # data_set['post_result2'] = post_result2

    show_image(data_set)
    save_image(left_c,'window method low texture BM left_c')
    save_image(right_c,'window method low texture BM right_c')
    save_image(my_result, 'window method low texture BM')
    save_image(diff_result, 'window method low texture BM diff_result')
    save_image(post_result, 'window method low texture BM post_result')
    # save_image(post_result2, 'window method Census BM post_result2')
