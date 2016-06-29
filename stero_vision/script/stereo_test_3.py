# -*- coding: utf-8 -*-

from PIL import Image
from common import get_data_set, show_image
import numpy as np
import matplotlib.pyplot as plt

data_set = get_data_set(0)
#get data
left = data_set['left']
right = data_set['right']
result = data_set['result']
my_result = np.zeros(left.shape)

#扫描左图像素
for row_pos in range(len(left)):
    row = left[row_pos]
    for pixel_pos in range(len(row)):
        pixel = left[row_pos][pixel_pos]
        #pixel = 0
    #print pixel_count

show_image(data_set)
show_image(right, 'right')
if __name__ == '__main__':
    pass
