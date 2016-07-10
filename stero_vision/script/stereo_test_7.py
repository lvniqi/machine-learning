# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 17:51:18 2016

@author: lvniqi
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
        # 代价计算
        diff_size = [d_max, ]
        diff_size.extend(self.left.shape)
        self.diff = np.zeros(diff_size)

        self.is_color = is_color

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
        column_length = self.right.shape[1]
        return np.absolute(self.left - self.right_extend[:, d_max - d:column_length + d_max - d])

    # Matching cost computation
    def compute_cost(self):
        for d in range(self.d_max):
            self.diff[d] = self.compute_cost_d(d)
        return self.diff

    # Cost aggregation
    def aggregate_cost(self):
        for d in range(self.d_max):
            diff = self.diff[d]
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
                    self.sad_result[d][row][column] = sad
        return self.sad_result

    # Disparity computation
    def get_result(self):
        for row in np.arange(self.row_length):
            for column in np.arange(self.column_length):
                min_sad = 0
                for d in np.arange(1, self.d_max):
                    if self.sad_result[min_sad][row][column] > self.sad_result[d][row][column]:
                        min_sad = d
                self.my_result[row][column] = min_sad
        return self.my_result

    # Disparity refinement


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
    stereo.compute_cost()
    stereo.aggregate_cost()
    my_result = stereo.get_result()
    my_result = my_result * 255 / d_max
    data_set['my_result_7'] = my_result
    show_image(data_set)
