# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 17:51:18 2016

@author: lvniqi
"""

from common import get_data_set, show_image, save_image, get_dll, get_aggregate_cost_cpp_func, compute_cost_d_cpp, \
    aggregate_cost_cpp, get_compute_cost_d_cpp_func
import numpy as np
from scipy.ndimage import filters


class StereoVisionBM2:
    def __init__(self, left, right, window_size=13, d_max=10, is_color=False):
        self.left = left
        self.right = right
        self.my_result = np.zeros(left.shape, dtype=np.int16)
        if is_color:
            (self.row_length, self.column_length, temp) = left.shape
            self.left_extend = self.make_border_rgb(self.left, window_size, d_max)
            self.right_extend = self.make_border_rgb(self.right, window_size, d_max)
        else:

            (self.row_length, self.column_length) = left.shape
            self.left_extend = self.make_border(self.left, window_size, d_max)
            self.right_extend = self.make_border(self.right, window_size, d_max)
        self.window_size = window_size
        self.d_max = d_max

        sad_size = [self.row_length, self.column_length, d_max, ]
        # sad 计算结果
        self.sad_left_result = np.zeros(sad_size, dtype=np.int32)
        self.sad_right_result = np.zeros(sad_size, dtype=np.int32)
        # 代价计算
        diff_size = [d_max, ]
        diff_size.extend(self.left.shape)
        self.left_diff = np.zeros(diff_size, dtype=np.int16)
        self.is_color = is_color
        self.dll = get_dll()
        self.compute_cost_d_cpp_func = get_compute_cost_d_cpp_func(self.dll)
        self.aggregate_cost_cpp_func = get_aggregate_cost_cpp_func(self.dll)

    @staticmethod
    def make_border(image, window_size, d_max):
        new_image = image

        left = new_image[:, 0:1]
        right = new_image[:, -2:-1]

        left = np.tile(left, (1, d_max))
        right = np.tile(right, (1, d_max))

        new_image = np.column_stack((left, new_image))
        new_image = np.column_stack((new_image, right))

        return new_image

    @staticmethod
    def make_border_rgb(image, window_size, d_max):
        new_image = image

        left = new_image[:, 0:1, :]
        right = new_image[:, -2:-1, :]

        left = np.tile(left, (1, d_max, 1))
        right = np.tile(right, (1, d_max, 1))

        new_image = np.column_stack((left, new_image))
        new_image = np.column_stack((new_image, right))

        return new_image

    def compute_cost_d(self, d):
        """
        Matching cost computation
        第一步 计算代价
        :param d: 深度d
        :return: 计算结果
        """
        return compute_cost_d_cpp(self.compute_cost_d_cpp_func, self.left,
                                  self.right_extend[:, self.d_max - d:self.column_length + self.d_max - d].copy())
        # python版本
        # return np.absolute(self.left - self.right_extend[:, self.d_max - d:self.column_length + self.d_max - d])

    # Matching cost computation
    def compute_cost(self):
        for d in range(self.d_max):
            self.left_diff[d] = self.gaussian_filter(self.compute_cost_d(d))
        return self.left_diff

    # Cost aggregation
    def aggregate_cost(self, is_python=False):
        if is_python:
            self.sad_left_result = self.aggregate_cost_python()
        else:
            self.sad_left_result = aggregate_cost_cpp(self.aggregate_cost_cpp_func, self.left_diff, self.window_size)

        for i in np.arange(self.d_max):
            left_sad = self.sad_left_result[:, :, i].copy()
            left_sad = left_sad[:, i:]
            for t in np.arange(i):
                left_sad = np.column_stack((left_sad, left_sad[:, -1]))
            right_sad = left_sad
            self.sad_right_result[:, :, i] = right_sad

        return (self.sad_left_result, self.sad_right_result)

    # 使用python编写的代价聚合程序 低速但算法相同
    def aggregate_cost_python(self):
        for d in range(self.d_max):
            diff = self.left_diff[d]
            for row in np.arange(self.row_length):
                top = (row - self.window_size / 2) if (row - self.window_size / 2) > 0 else 0
                bottom = (row + self.window_size / 2 + 1) if \
                    (row + self.window_size / 2 + 1) <= self.row_length else self.row_length
                for column in np.arange(self.column_length):
                    left = (column - self.window_size / 2) if (column - self.window_size / 2) > 0 else 0
                    right = (column + self.window_size / 2 + 1) if \
                        (column + self.window_size / 2 + 1) <= self.column_length else self.column_length

                    diff_window = diff[top:bottom, left:right]

                    sad = np.sum(diff_window)
                    # 归一化
                    sad_normal = sad * 100 / (bottom - top) / (right - left)
                    self.sad_left_result[row][column][d] = sad_normal
        return self.sad_left_result

    # Disparity computation
    def get_result(self, is_left=True):
        if is_left:
            used_sad_result = self.sad_left_result
        else:
            used_sad_result = self.sad_right_result
        for row in np.arange(self.row_length):
            for column in np.arange(self.column_length):
                min_sad = 0
                for d in np.arange(1, self.d_max):
                    if used_sad_result[row][column][min_sad] > used_sad_result[row][column][d]:
                        min_sad = d
                self.my_result[row][column] = min_sad
        # self.post_processing()
        return self.my_result.copy()

    def left_right_check(self):
        result = np.zeros((self.row_length, self.column_length), np.int16)
        print "left:"
        left = self.get_result(is_left=True)
        print "right:"
        right = self.get_result(is_left=False)
        for row in np.arange(self.row_length):
            left_row = left[row]
            right_row = right[row]
            result_row = result[row]
            for column in np.arange(self.column_length):
                # 左侧视差
                disparity_left = left_row[column]
                pos_right = column - disparity_left
                if pos_right < 0:
                    continue
                # 右侧视差
                disparity_right = right_row[pos_right]
                # 得到遮挡/不稳定点
                if disparity_left != disparity_right:
                    result_row[column] = d_max
        return result

    @staticmethod
    def gaussian_filter(image_in, sigma=1.5):
        return filters.gaussian_filter(image_in, sigma)

        # Disparity refinement

    @staticmethod
    def get_window(color_window, window_size):
        result_window = np.zeros(color_window.shape)
        row_start = color_window.shape[0] / 2
        column_start = color_window.shape[1] / 2
        pixel_mid = color_window[row_start][column_start]

        def get_window_line():
            # 记录行差异
            diff_row = np.absolute(row_pos - row_start)
            # 找到像素标记
            find_pixel = False
            # 向左搜索
            for column_pos in np.arange(column_start, -1, -1):
                if row[column_pos] == pixel_mid:
                    result_window[row_pos][column_pos] = window_size
                    find_pixel = True
                else:
                    break
            # 向右搜索
            for column_pos in np.arange(column_start, color_window.shape[1], 1):
                if row[column_pos] == pixel_mid:
                    result_window[row_pos][column_pos] = window_size
                    find_pixel = True
                else:
                    break
            return find_pixel

        # 向上搜索
        for row_pos in np.arange(row_start, -1, -1):
            row = color_window[row_pos]
            if not get_window_line():
                break
        # 向下搜索
        for row_pos in np.arange(row_start, color_window.shape[0], 1):
            row = color_window[row_pos]
            if not get_window_line():
                break
        # 其他区域赋值
        for row_pos in np.arange(color_window.shape[0]):
            # 记录行差异
            diff_row = np.absolute(row_pos - row_start)
            for column_pos in np.arange(color_window.shape[1]):
                if result_window[row_pos][column_pos] != 0:
                    continue
                diff_column = np.absolute(column_pos - column_start)
                result_window[row_pos][column_pos] = (window_size - 1) / (diff_column + diff_row)

        return result_window


if __name__ == '__main__':
    data_set = get_data_set(0)
    # get data
    left = data_set['left']
    right = data_set['right']
    result = data_set['result']
    import time

    window_size = 11
    d_max = 15
    tt = time.time()
    stereo = StereoVisionBM2(left, right, window_size, d_max)
    stereo.compute_cost()
    t_diff = stereo.aggregate_cost()
    my_result = stereo.get_result()
    my_result = my_result * 255 / d_max
    data_set['my_result_7'] = my_result

    diff_result = stereo.left_right_check()
    diff_result = diff_result * 255 / d_max
    data_set['diff_result'] = diff_result
    save_image(diff_result, 'diff_result')

    show_image(data_set)
    print time.time() - tt
    save_image(my_result, 'window method 7')
