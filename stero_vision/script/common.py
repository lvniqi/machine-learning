# -*- coding: utf-8 -*-
import PIL
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

sub_folders = ['barn2', 'bull', 'cones']


def get_data_folder(sub_folder='barn2'):
    this_dir = os.getcwd().replace('\\', '/')
    folder = this_dir.split('/')
    folder[-1] = 'data/' + sub_folder
    data_dir = reduce(lambda result, f: result + '/' + f, folder) + '/'
    return data_dir


def get_data_set(pos=0, is_color=False):
    data_set = {}
    if is_color:
        img_l = np.array(Image.open(get_data_folder(sub_folders[pos]) + 'im2.png'), dtype=np.int16)
        img_r = np.array(Image.open(get_data_folder(sub_folders[pos]) + 'im6.png'), dtype=np.int16)
        img_result = np.array(Image.open(get_data_folder(sub_folders[pos]) + 'disp6.png'), dtype=np.int16)
    else:
        img_l = np.array(Image.open(get_data_folder(sub_folders[pos]) + 'im2.png').convert('L'), dtype=np.int16)
        img_r = np.array(Image.open(get_data_folder(sub_folders[pos]) + 'im6.png').convert('L'), dtype=np.int16)
        img_result = np.array(Image.open(get_data_folder(sub_folders[pos]) + 'disp6.png').convert('L'), dtype=np.int16)
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
                plt.imshow(np.array(image[key],dtype = np.uint8))
            else:
                plt.imshow(image[key], plt.cm.gray, norm=plt.Normalize(vmin=0, vmax=255))
            pos += 1
        plt.show()
    elif img_type == np.ndarray:
        if title is None:
            title = str(image.shape) + str(image.dtype)
        plt.title(title)
        if is_color:
            plt.imshow(np.array(image,dtype = np.uint8))
        else:
            plt.imshow(image, plt.cm.gray, norm=plt.Normalize(vmin=0, vmax=255))
        plt.show()


def save_image(image, name='test'):
    t = Image.fromarray(np.uint8(image))
    t.save(get_data_folder('result') + name + '.jpg')


if __name__ == '__main__':
    # gray
    dataset_t = get_data_set()
    left_img = dataset_t['left']
    # save_image(left_img)
    show_image(dataset_t)
    # color
    dataset_t = get_data_set(is_color=True)
    left_img = dataset_t['left']
    show_image(dataset_t,is_color=True)
