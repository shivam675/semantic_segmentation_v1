import os
import pandas as pd
import numpy as np

import rospkg
import glob
import cv2
from sklearn.model_selection import train_test_split
pkg = rospkg.RosPack()



path = pkg.get_path('semantic_segmentor')
full_train_path_x = path + '/dataset/train/train_combined/images/*'
full_train_path_y = path + '/dataset/train/train_combined/masks/*'

train_x = glob.glob(full_train_path_x)
train_y = glob.glob(full_train_path_y)
# print(sorted(train_x)[1])
# print(sorted(train_y)[1])

############ local params #########
IMAGE_HEIGHT, IMAGE_WIDTH = 256, 256
CHANNEL = 3
BACK_BONE = 'resnet34'
# print(len(train_x) == len(train_y))



def load_data():
    x = np.zeros((len(train_x), IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL), dtype=np.float32)
    y = np.zeros((len(train_y), IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL), dtype=np.float32)
    for idx, file in enumerate(train_x):
        img = cv2.imread(file,1)
        mask = cv2.imread(train_y[idx], 1)
        try:
            img = cv2.resize(img, (256, 256))
            x[idx] = img
            mask = cv2.resize(mask, (256,256))
            y[idx] = mask
        except Exception as e:
            print(e)
    print(np.shape(x))
    print(np.shape(y))
    # return x, y
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size= 0.15, random_state=None)
    return x_train, y_train, x_val, y_val


if __name__ == '__main__':
    x, y = load_data()
    # x_t, y_t, x_v, x_val = split(x, y)
    # print(np.shape(x_t))