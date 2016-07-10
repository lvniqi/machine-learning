# -*- coding: utf-8 -*-
"""
Created on Mon Jul 04 15:25:30 2016

@author: WZ5040
"""

from common import get_data_set, show_image, save_image
import numpy as np


class StereoVisionBM1:
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

        sad_size = [d_max, ]
        sad_size.extend((self.row_length, self.column_length))
        # sad 计算结果
        self.sad_result = np.zeros(sad_size)

        self.used_window_compare_method = self.default_window_compare_method

        self.is_color = is_color

    @staticmethod
    def make_border(image, window_size, d_max):
        new_image = image

        top = image[0]
        bottum = image[-1]
        top = np.tile(top, (window_size / 2, 1))
        bottum = np.tile(bottum, (window_size / 2, 1))
        new_image = np.row_stack((top, new_image))
        new_image = np.row_stack((new_image, bottum))

        left = new_image[:, 0:1]
        right = new_image[:, -2:-1]

        left = np.tile(left, (1, d_max + window_size / 2))
        right = np.tile(right, (1, d_max + window_size / 2))

        new_image = np.column_stack((left, new_image))
        new_image = np.column_stack((new_image, right))

        return new_image

    @staticmethod
    def make_border_rgb(image, window_size, d_max):
        new_image = image
        row_lenth = image.shape
        top = image[0]
        bottum = image[-1]
        top = np.tile(top, (window_size / 2, 1, 1))
        bottum = np.tile(bottum, (window_size / 2, 1, 1))
        new_image = np.row_stack((top, new_image))
        new_image = np.row_stack((new_image, bottum))

        left = new_image[:, 0:1, :]
        right = new_image[:, -2:-1, :]

        left = np.tile(left, (1, d_max + window_size / 2, 1))
        right = np.tile(right, (1, d_max + window_size / 2, 1))

        new_image = np.column_stack((left, new_image))
        new_image = np.column_stack((new_image, right))

        return new_image

    def get_sad_window(self, row, column, d):
        # 由于向左右延展了window_size
        window_left = self.left_extend[row:row + self.window_size,
                      column + self.d_max: column + self.d_max + self.window_size]
        window_right = self.right_extend[row:row + self.window_size,
                       column + self.d_max - d: column + self.d_max + self.window_size - d]
        sad = 0
        # 加速并没有卵用 所以隐去
        '''if row > 0 and column > 0:
            sad_last = sad_result[d][row - 1][column]
            window_left_last = left[row - 1][column + d_max: column + d_max + self.window_size]
            window_right_last = right[row - 1][column + d_max - d: column + d_max + self.window_size - d]
            window_left_new = left[row + self.window_size-1][column + d_max: column + d_max + self.window_size ]
            window_right_new = right[row + self.window_size-1][column + d_max - d: column + d_max + self.window_size - d]
            sad = sad_last - np.sum(np.absolute(window_left_last - window_right_last)) + np.sum(np.absolute(
                window_left_new - window_right_new))
        else:
            sad = np.sum(np.absolute(window_left - window_right))
        '''
        sad = self.used_window_compare_method(window_left, window_right)
        self.sad_result[d][row][column] = sad
        return sad

    def get_sad_d(self, d):
        for row in np.arange(self.row_length):
            for column in np.arange(self.column_length):
                self.get_sad_window(row, column, d)
        return self.sad_result[d]

    def get_sad_all(self):
        for d in np.arange(self.d_max):
            self.get_sad_d(d)
        return self.sad_result

    def get_result(self):
        for row in np.arange(self.row_length):
            for column in np.arange(self.column_length):
                min_sad = 0
                for d in np.arange(1, self.d_max):
                    if self.sad_result[min_sad][row][column] > self.sad_result[d][row][column]:
                        min_sad = d
                self.my_result[row][column] = min_sad
        return self.my_result

    @staticmethod
    def default_window_compare_method(window_1, window_2):
        return np.sum(np.absolute(window_1 - window_2))

    def set_window_compare_method(self, method):
        self.used_window_compare_method = method


if __name__ == '__main__':
    data_set = get_data_set(0)
    # get data
    left = data_set['left']
    right = data_set['right']
    result = data_set['result']
    import time

    window_size = 13
    d_max = 15
    tt = time.time()
    stereo = StereoVisionBM1(left, right, window_size, d_max)
    '''
    stereo.get_sad_all()
    my_result = stereo.get_result()
    my_result = my_result * 255 / d_max
    data_set['my_result_6'] = my_result
    print time.time() - tt
    save_image(my_result, 'window method 6')
    show_image(data_set)'''
    data_set = get_data_set(0, is_color=True)
    left = data_set['left']
    right = data_set['right']
    result = data_set['result']
    stereo = StereoVisionBM1(left, right, window_size, d_max, is_color=True)
    stereo.get_sad_all()
    my_result = stereo.get_result()
    my_result = my_result * 255 / d_max
    data_set['my_result_6'] = my_result
    show_image(data_set,is_color=True)
