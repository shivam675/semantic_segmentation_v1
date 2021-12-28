import glob
from os import name
import rospkg
import numpy as np
import cv2

######## semantic segmentation related part ########
# from segmentation_models import Unet
# # from segmentation_models.backbones import get_preprocessing
# from segmentation_models.losses import bce_jaccard_loss
# from segmentation_models.metrics import iou_score


from keras.layers import Input, Conv2D
from keras.models import Model

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
    return x, y

def split(x,y):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size= 0.15, random_state=None)
    return x_train, y_train, x_val, y_val


# def preprocessing(back_bone, t_x, v_x):
#     preprocess_input = get_preprocessing(back_bone)
#     pre_x_train = preprocess_input(t_x)
#     pre_x_val = preprocess_input(v_x)


# def building_model():
#     base_model = Unet(backbone_name='inceptionv3', encoder_weights='imagenet')
#     inp = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL))
#     layer_one = Conv2D(3, strides=(1,1))(inp)
#     out = base_model(layer_one)
#     layer_two = Conv2D(3, strides=(1,1))(out)
#     model = Model(inp, layer_two, name=base_model.name)
#     model.summary()



if __name__ == '__main__':
    x , y = load_data()
    x_t, y_t, x_v, y_v = split(x,y)
    # preprocessing(BACK_BONE, x_t, x_v)
    # building_model()