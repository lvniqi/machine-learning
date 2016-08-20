# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 17:51:18 2016

@author: lvniqi
"""

from common import get_data_set, show_image, save_image, get_dll, get_aggregate_cost_cpp_func, compute_cost_d_cpp, \
    aggregate_cost_cpp, get_compute_cost_d_cpp_func, get_result_cpp_func, get_result_cpp, get_left_right_check_cpp_func, \
    left_right_check_cpp, low_texture_detection_cpp_func, low_texture_detection_cpp
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

        self.low_texture = None

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
        return self.left_diff.copy()

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
        result_cpp_func = get_result_cpp_func(self.dll)
        temp_result = None
        if is_left:
            temp_result = self.my_result
            used_sad_result = self.sad_left_result
        else:
            temp_result = self.my_result.copy()
            used_sad_result = self.sad_right_result

        get_result_cpp(result_cpp_func, temp_result, used_sad_result)
        '''for row in np.arange(self.row_length):
            for column in np.arange(self.column_length):
                min_sad = 0
                for d in np.arange(1, self.d_max):
                    if used_sad_result[row][column][min_sad] > used_sad_result[row][column][d]:
                        min_sad = d
                self.my_result[row][column] = min_sad'''
        return temp_result.copy()

    # 左右视差检查
    def left_right_check(self):
        self.left_right_result = np.zeros((self.row_length, self.column_length), np.int16)
        print "left:"
        left = self.get_result(is_left=True)
        print "right:"
        right = self.get_result(is_left=False)
        left_right_check_cpp_func = get_left_right_check_cpp_func(self.dll)
        left_right_check_cpp(left_right_check_cpp_func, self.left_right_result, left, right)
        '''
        for row in np.arange(self.row_length):
            left_row = left[row]
            right_row = right[row]
            result_row = self.left_right_result[row]
            for column in np.arange(self.column_length):
                # 左侧视差
                disparity_left = left_row[column]
                pos_right = column - disparity_left / 16
                if pos_right < 0:
                    continue
                # 右侧视差
                disparity_right = right_row[pos_right]
                # 得到遮挡/不稳定点
                diff = abs(disparity_left - disparity_right)
                if diff > 8:
                    result_row[column] = diff
        '''
        result = self.left_right_result.copy()
        result *= 255 / self.d_max

        return result

    def post_processing(self):
        """
        后处理 处理视差检后的结果
        :return:
        """
        '''for row in np.arange(self.row_length):
            for column in np.arange(self.column_length):
                if self.left_right_result[row][column]:
                    left_pos = column - self.window_size / 2 \
                        if column - self.window_size / 2 >= 0 \
                        else 0
                    right_pos = column + self.window_size / 2 + 1 \
                        if column + self.window_size / 2 + 1 <= self.column_length \
                        else self.column_length
                    top_pos = row - self.window_size / 2 \
                        if row - self.window_size / 2 >= 0 \
                        else 0
                    bottom_pos = row + self.window_size / 2 + 1 \
                        if row + self.window_size / 2 + 1 <= self.row_length \
                        else self.row_length
                    vote = np.zeros(self.d_max * 16, np.int16)
                    for i in np.arange(top_pos, bottom_pos):
                        for j in np.arange(left_pos, right_pos):
                            if self.left_right_result[i][j] <= 8:
                                vote[self.my_result[i][j]] += 1
                    self.my_result[row][column] = np.argmax(vote)'''
        post_processing_cpp = self.dll.post_processing
        import ctypes
        post_processing_cpp.restype = ctypes.c_void_p
        post_processing_cpp.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.int16, ndim=2),
            np.ctypeslib.ndpointer(dtype=np.int16, ndim=2),
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        strides = np.array(self.left.strides, dtype=np.int32)
        shapes = np.array(self.left.shape, dtype=np.int32)
        post_processing_cpp(self.my_result, self.left_right_result, strides, shapes, self.window_size, self.d_max)

        return self.my_result.copy()

    def fix_low_texture(self):
        for row in range(self.low_texture.shape[0]):
            for column in range(self.low_texture.shape[1]):
                if self.low_texture[row][column]:
                    (left, right, top, bottom) = (0, 0, 0, 0)
                    # find left
                    for left in range(column, -1, -1):
                        if not self.low_texture[row][left]:
                            break
                    # find right
                    for right in range(column, self.low_texture.shape[1]):
                        if not self.low_texture[row][right]:
                            break
                    # find top
                    for top in range(row, -1, -1):
                        if not self.low_texture[top][column]:
                            break
                    # find bottom
                    for bottom in range(row, self.low_texture.shape[0]):
                        if not self.low_texture[bottom][column]:
                            break
                    value_left = self.my_result[row][left]
                    value_right = self.my_result[row][right]
                    step = (value_right - value_left) * 1.0 / (right - left)
                    value = value_left + step * (column - left)

                    value_top = self.my_result[top][column]
                    value_bottom = self.my_result[bottom][column]
                    step_2 = (value_bottom - value_top) * 1.0 / (bottom - top)
                    value_2 = value_top + step * (bottom - top)
                    if value == value_2:
                        self.my_result[row][column] = value
                    elif np.abs(value - value_2) < 4 * 16:
                        self.my_result[row][column] = (value + value_2) / 2
                    else:
                        pass
                        #self.my_result[row][column] = 0
                        #self.my_result[row][column] = (value + value_2) / 2
        return self.my_result

    def low_texture_detection(self, texture_range=1):
        """
        低纹理区域检测
        :return: 检测结果
        """
        func = low_texture_detection_cpp_func(self.dll)
        result = np.zeros(self.left.shape, np.int16)
        low_texture_detection_cpp(func, result, self.left, self.window_size, int(self.window_size * texture_range))
        self.low_texture = filters.median_filter(result, 3)
        return self.low_texture.copy()

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
    d_max = 32
    tt = time.time()
    stereo = StereoVisionBM2(left, right, window_size, d_max)
    stereo.compute_cost()
    t_diff = stereo.aggregate_cost()
    my_result = stereo.get_result()
    my_result = my_result * (255.0 / d_max / 16)
    data_set['my_result_7'] = my_result

    low_texture = stereo.low_texture_detection()
    data_set['low_texture'] = low_texture

    diff_result = stereo.left_right_check()
    diff_result = diff_result * (255.0 / d_max / 16)
    data_set['diff_result'] = diff_result

    post_result = stereo.post_processing()
    post_result = post_result * (255.0 / d_max / 16)
    data_set['post_result'] = post_result

    post_result2 = stereo.fix_low_texture()
    post_result2 = post_result2 * (255.0 / d_max / 16)
    data_set['post_result2'] = post_result2

    show_image(data_set)
    print time.time() - tt
    save_image(diff_result, 'diff_result')
    save_image(post_result, 'post_result')
    save_image(post_result2, 'post2_result')
    save_image(low_texture, 'low_texture')
    save_image(my_result, 'window method 7')
