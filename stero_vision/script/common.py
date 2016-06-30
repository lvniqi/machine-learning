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


def get_data_set(pos=0):
    dataset = {}
    img_L = np.array(Image.open(get_data_folder(sub_folders[pos]) + 'im2.png').convert('L'), 'f')
    img_R = np.array(Image.open(get_data_folder(sub_folders[pos]) + 'im6.png').convert('L'), 'f')
    img_result = np.array(Image.open(get_data_folder(sub_folders[pos]) + 'disp6.png').convert('L'), 'f')
    dataset['left'] = img_L
    dataset['right'] = img_R
    dataset['result'] = img_result
    return dataset


def show_image(image, title=None):
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
            plt.imshow(image[key], plt.cm.gray, norm=plt.Normalize(vmin=0, vmax=255))
            pos += 1
        plt.show()
    elif img_type == np.ndarray:
        if title is None:
            title = str(image.shape) + str(image.dtype)
        plt.title(title)
        plt.imshow(image, plt.cm.gray, norm=plt.Normalize(vmin=0, vmax=255))
        plt.show()


def save_image(image, name='test'):
    t = Image.fromarray(np.uint8(image))
    t.save(get_data_folder('result') + name + '.jpg')


if __name__ == '__main__':
    dataset = get_data_set()
    left_img = dataset['left']
    save_image(left_img)
    show_image(dataset)
