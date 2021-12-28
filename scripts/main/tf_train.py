#!/usr/bin/python3

from numpy.lib.npyio import load
import tensorflow as tf
import tensorflow_datasets as tfds
from pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
import rospkg
from sklearn.model_selection import train_test_split
import glob
import cv2
import numpy as np


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
# print(len(train_x) == len(train_y))

class TrainSemanticSegmentor():
    def __init__(self) -> None:
        self.all_x_batchs = []
        self.all_y_batchs = []
        self.current_batch_x = None
        self.current_batch_y = None
        self.BATCH_SIZE = 64
        self.BUFFER_SIZE = 1000
        self.STEPS_PER_EPOCH = 65
        self.OUTPUT_CLASSES = 3
        self.EPOCHS = 20
        self.VAL_SUBSPLITS = 5
        self.VALIDATION_STEPS = 4

        pass

    def load_data(self, full_x, full_y):
        x_train, x_val, y_train, y_val = train_test_split(full_x, full_y, test_size= 0.15, random_state=None)
        return x_train, y_train, x_val, y_val


    def batch_data(self, x, y, batch_size = 64):
        x_batch, y_batch = [], []
        while x and y:
            self.all_x_batchs.append(train_x[:batch_size])
            self.all_y_batchs.append(train_y[:batch_size])
            x = x[batch_size:]
            y = y[batch_size:]


    def retrun_batch(self):
        self.model_defination()
        self.unet_model(self.OUTPUT_CLASSES)
        self.train_model()
        for x_b, y_b in zip(self.all_x_batchs, self.all_y_batchs):
            x = np.zeros((len(x_b), IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL), dtype=np.float32)
            y = np.zeros((len(y_b), IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL), dtype=np.float32)

            for idx, file in enumerate(x_b):
                img = cv2.imread(file,1)
                mask = cv2.imread(y_b[idx], 1)
                
                try:
                    img = cv2.resize(img, (256, 256))
                    x[idx] = img
                    mask = cv2.resize(mask, (256,256))
                    y[idx] = mask
                except Exception as e:
                    print(e)
            print(np.shape(x))
            print(np.shape(y))


            # self.current_batch_x, self.current_batch_y = x, y
            # run training function here 
            model_history = self.model.fit(x, y, epochs=self.EPOCHS,
                                    steps_per_epoch= self.STEPS_PER_EPOCH)





    def model_defination(self):
        base_model = tf.keras.applications.MobileNetV2(input_shape=[256, 256, 3], include_top=False)
        # Use the activations of these layers
        layer_names = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
        ]
        base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        self.down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

        self.down_stack.trainable = False

        self.up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
            ]
    
    def unet_model(self, output_channels:int):
        inputs = tf.keras.layers.Input(shape=[256, 256, 3])

        # Downsampling through the model
        skips = self.down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            filters=output_channels, kernel_size=3, strides=2,
            padding='same')  #64x64 -> 128x128

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)
    
    def train_model(self):
        
        self.model = self.unet_model(output_channels=self.OUTPUT_CLASSES)
        self.model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])


        

if __name__ == "__main__":
    k = TrainSemanticSegmentor()
    x_t, y_t, x_v, y_v = k.load_data(train_x, train_y)
    k.batch_data(x_t, y_t)
    k.retrun_batch()
