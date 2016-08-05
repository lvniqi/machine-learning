# -*- coding: utf-8 -*-
import PIL
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import ctypes

sub_folders = ['barn2', 'bull', 'cones', 'poster', 'sawtooth', 'teddy', 'tsukuba', 'venus','mydata_1']


def get_dll_folder():
    import sys
    this_dir = os.getcwd().replace('\\', '/')
    if '64-bit' in sys.version:
        return this_dir + '/cpp_speed_up/x64/Release/cpp_speed_up.dll'
    else:
        return this_dir + '/cpp_speed_up/Release/cpp_speed_up.dll'


def get_dll():
    return ctypes.windll.LoadLibrary(get_dll_folder())


def get_compute_cost_d_cpp_func(dll):
    """
    代价计算
    :param dll: dll文件
    :return: 代价计算函数
    """
    compute_cost_d = dll.compute_cost_d

    compute_cost_d.restype = ctypes.c_void_p
    compute_cost_d.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.int16, ndim=2),
        np.ctypeslib.ndpointer(dtype=np.int16, ndim=2),
        np.ctypeslib.ndpointer(dtype=np.int16, ndim=2),
        np.ctypeslib.ndpointer(dtype=np.int16, ndim=1),
        np.ctypeslib.ndpointer(dtype=np.int16, ndim=1),
    ]
    return compute_cost_d
    # compute_cost_d.


def get_compute_cost_bt_d_cpp_func(dll):
    """
    代价计算bt版本
    :param dll: dll文件
    :return: 代价计算函数bt版本
    """
    compute_cost_bt_d = dll.compute_cost_bt_d

    compute_cost_bt_d.restype = ctypes.c_void_p
    compute_cost_bt_d.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.int16, ndim=2),
        np.ctypeslib.ndpointer(dtype=np.int16, ndim=2),
        np.ctypeslib.ndpointer(dtype=np.int16, ndim=2),
        np.ctypeslib.ndpointer(dtype=np.int16, ndim=1),
        np.ctypeslib.ndpointer(dtype=np.int16, ndim=1),
    ]
    return compute_cost_bt_d


def get_compute_cost_census_d_cpp_func(dll):
    """
    代价计算census版本
    :param dll: dll文件
    :return: 代价计算函数census版本
    """
    compute_cost_census_d = dll.compute_cost_census_d

    compute_cost_census_d.restype = ctypes.c_void_p
    compute_cost_census_d.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.int16, ndim=2),
        np.ctypeslib.ndpointer(dtype=np.bool, ndim=3),
        np.ctypeslib.ndpointer(dtype=np.bool, ndim=3),
        np.ctypeslib.ndpointer(dtype=np.int16, ndim=1),
        np.ctypeslib.ndpointer(dtype=np.int16, ndim=1),
    ]
    return compute_cost_census_d


def compute_cost_d_cpp(func, left, right):
    """
    代价计算python包装
    :param func:代价计算函数
    :param left:左图
    :param right:右图
    :return:结果
    """
    strides = np.array(left.strides, dtype=np.int16)
    shapes = np.array(left.shape, dtype=np.int16)
    result = np.zeros(left.shape[:2], dtype=np.int16)
    func(result, left, right, strides, shapes)
    return result


def get_census_cpp_func(dll):
    """
        census序列计算
        :param dll: dll文件
        :return: census序列计算
    """
    get_census = dll.get_census

    get_census.restype = ctypes.c_void_p
    get_census.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.bool, ndim=3),
        np.ctypeslib.ndpointer(dtype=np.int16, ndim=2),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
        ctypes.c_int32
    ]
    return get_census


def get_census_cpp(func, image, window_size):
    """
    census计算python包装
    :param func: census计算函数
    :param image:待计算图像
    :param window_size:窗口大小
    :return:结果
    """
    strides = np.array(image.strides, dtype=np.int32)
    shapes = np.array(image.shape, dtype=np.int32)
    (row_length, column_length) = image.shape
    result = np.zeros((row_length, column_length, window_size * window_size), dtype=np.bool)
    func(result, image, strides, shapes, window_size)
    return result


def get_aggregate_cost_cpp_func(dll):
    """
    代价聚合函数
    :param dll: dll文件
    :return: 代价聚合函数
    """
    aggregate_cost = dll.aggregate_cost

    aggregate_cost.restype = ctypes.c_void_p
    aggregate_cost.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=3),
        np.ctypeslib.ndpointer(dtype=np.int16, ndim=3),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
        np.ctypeslib.ndpointer(dtype=np.int16, ndim=1),
        ctypes.c_int16,
    ]
    return aggregate_cost


def aggregate_cost_cpp(func, diff, window_size):
    """
    代价聚合python包装
    :param func: 代价聚合函数
    :param diff: 左右图差异
    :param window_size: 窗口大小
    :return: 聚合后的代价值
    """
    shapes = np.array(diff.shape, dtype=np.int16)
    diff_strides = np.array(diff.strides, dtype=np.int32)
    result = np.zeros((diff.shape[1], diff.shape[2], diff.shape[0]), dtype=np.int32)
    result_strides = np.array(result.strides, dtype=np.int32)
    func(result, diff, diff_strides, result_strides, shapes, window_size)
    return result


def get_dp_forward_cpp_func(dll):
    """
    动态规划搜索函数
    :param dll: dll文件
    :return: 动态规划搜索函数
    """
    dp_search_forward = dll.DP_search_forward

    dp_search_forward.restype = ctypes.c_void_p
    dp_search_forward.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.int16, ndim=2),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2),
        np.ctypeslib.ndpointer(dtype=np.int16, ndim=2),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
    ]
    return dp_search_forward


def get_dp_forward_cpp_func_fast(dll):
    """
    动态规划搜索函数
    :param dll: dll文件
    :return: 动态规划搜索函数
    """
    dp_search_forward2 = dll.DP_search_forward2

    dp_search_forward2.restype = ctypes.c_void_p
    dp_search_forward2.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.int16, ndim=2),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2),
        np.ctypeslib.ndpointer(dtype=np.int16, ndim=2),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
    ]
    return dp_search_forward2


def dp_forward_cpp(func, result, cost, sad_row, column_length, d_max, p):
    """
    动态规划向前python包装
    :param func: 动态规划搜索函数
    :param result: 结果矩阵
    :param cost:代价矩阵
    :param sad_row: 聚合后单行数据
    :param column_length:行宽度
    :param d_max:搜索最大视差
    :param p:约束参数
    """
    func(result, cost, sad_row, column_length, d_max, p)


def get_result_cpp_func(dll):
    """
    视差计算函数
    :param dll: dll文件
    :return: 视差计算函数
    """
    get_result = dll.get_result

    get_result.restype = ctypes.c_void_p
    get_result.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.int16, ndim=2),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=3),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
    ]
    return get_result


def get_result_cpp(func, result, sad_diff):
    """
    视察计算python包装
    :param func: 视差计算函数
    :param result: 视差结果
    :param sad_diff: 聚合后视差值
    """
    strides = np.array(sad_diff.strides, dtype=np.int32)
    shapes = np.array(sad_diff.shape, dtype=np.int32)
    func(result, sad_diff, strides, shapes)


def get_data_folder(sub_folder='barn2'):
    this_dir = os.getcwd().replace('\\', '/')
    folder = this_dir.split('/')
    folder[-1] = 'data/' + sub_folder
    data_dir = reduce(lambda result, f: result + '/' + f, folder) + '/'
    return data_dir


def get_data_set(pos=0, is_color=False):
    #print pos,sub_folders[pos]
    data_set = {}
    if is_color:
        img_l = np.array(Image.open(get_data_folder(sub_folders[pos]) + 'im2.png'), dtype=np.int16)
        img_r = np.array(Image.open(get_data_folder(sub_folders[pos]) + 'im6.png'), dtype=np.int16)
        img_result = np.array(Image.open(get_data_folder(sub_folders[pos]) + 'disp2.png'), dtype=np.int16)
    else:
        img_l = np.array(Image.open(get_data_folder(sub_folders[pos]) + 'im2.png').convert('L'), dtype=np.int16)
        img_r = np.array(Image.open(get_data_folder(sub_folders[pos]) + 'im6.png').convert('L'), dtype=np.int16)
        img_result = np.array(Image.open(get_data_folder(sub_folders[pos]) + 'disp2.png').convert('L'), dtype=np.int16)
    data_set['left'] = img_l
    data_set['right'] = img_r
    data_set['result'] = img_result
    return data_set


def show_image(image, title=None, is_color=False):
    plt.figure(num=u'测试')
    img_type = type(image)
    if img_type == dict:
        image_len = len(image)
        x_axis = (image_len + 1) / 2
        y_axis = 2
        pos = 1
        for key in image:
            plt.subplot(x_axis, y_axis, pos)  # 将窗口分为x_axis行y_axis列四个子图
            plt.title(key)
            if is_color:
                plt.imshow(np.array(image[key], dtype=np.uint8))
            else:
                plt.imshow(image[key], plt.cm.gray, norm=plt.Normalize(vmin=0, vmax=255))
            pos += 1
        plt.show()
    elif img_type == np.ndarray:
        if title is None:
            title = str(image.shape) + str(image.dtype)
        plt.title(title)
        if is_color:
            plt.imshow(np.array(image, dtype=np.uint8))
        else:
            plt.imshow(image, plt.cm.gray, norm=plt.Normalize(vmin=0, vmax=255))
        plt.show()


def save_image(image, name='test'):
    t = Image.fromarray(np.uint8(image))
    t.save(get_data_folder('result') + name + '.jpg')


if __name__ == '__main__':
    dll = get_dll()
    compute_cost_d = get_compute_cost_d_cpp_func(dll)
    # test cpp speed up
    left = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
    right = np.array([[-1, 2, 2], [4, 3, 3]], dtype=np.int16)
    result = compute_cost_d_cpp(compute_cost_d, left, right)
    print result
    # gray
    '''dataset_t = get_data_set()
    left_img = dataset_t['left']
    # save_image(left_img)
    show_image(dataset_t)
    # color
    dataset_t = get_data_set(is_color=True)
    left_img = dataset_t['left']
    show_image(dataset_t, is_color=True)'''
