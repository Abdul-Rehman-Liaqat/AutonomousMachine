#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:43:00 2017

@author: abdulliaqat
"""

import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split 
from sklearn.cross_validation import train_test_split
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from keras.layers import Lambda,Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input, concatenate
import matplotlib.image as mpimg
np.random.seed(0)

def build_model_nvidia(args):
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)

    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem.
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=args["input_shape"]))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(args["keep_prob"]))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1)) 
    model.summary()

    return model

def build_model_nvidia_with_speed(args):
    print("Nvidia with speed input")
    image_input = Input(shape=(480, 640, 3), name = 'images')
    speed = Input(shape=(1,), names = "speed")
    x = Conv2D(24, 5, 5, activation='elu', subsample=(2, 2))(image_input)
    x = Conv2D(36, 5, 5, activation='elu', subsample=(2, 2))(x)
    x = Conv2D(48, 5, 5, activation='elu', subsample=(2, 2))(x)
    x = Conv2D(64, 3, 3, activation='elu')(x)
    x = Conv2D(64, 3, 3, activation='elu')(x)
    x = Dropout(args["keep_prob"])(x)
    x = Flatten()(x)
    x = Dense(100, activation = 'elu')(x)
    x = concatenate(x,speed)    
    x = Dense(50, activation = 'elu')(x)
    x = Dense(10, activation = 'elu')(x)
    steering_angle = Dense(1, name="steering_angle")(x)
    model = Model(input=[image_input,speed], output=steering_angle)
    model.summary()

    return model

def build_model_nvidia_multi_task(args):
    print("Nvidia Multi-Task")
    image_input = Input(shape=(480, 640, 3))
    x = Conv2D(24, 5, 5, activation='elu', subsample=(2, 2))(image_input)
    x = Conv2D(36, 5, 5, activation='elu', subsample=(2, 2))(x)
    x = Conv2D(48, 5, 5, activation='elu', subsample=(2, 2))(x)
    x = Conv2D(64, 3, 3, activation='elu')(x)
    x = Conv2D(64, 3, 3, activation='elu')(x)
    x = Dropout(args["keep_prob"])(x)
    x = Flatten()(x)
    x = Dense(100, activation = 'elu')(x)
    x = Dense(50, activation = 'elu')(x)
    x = Dense(10, activation = 'elu')(x)
    steering_angle = Dense(1, name="steering_angle")(x)
    speed = Dense(1, name="speed")(x)
    model = Model(input=image_input, output=[steering_angle,speed])
    model.summary()

    return model

def build_model_nvidia_multi_task2(args):
    print("Nvidia-Modified Multi-Task")
    image_input = Input(shape=(480, 640, 3))
    x = Conv2D(36, 5, 5, activation='elu', subsample=(2, 2))(image_input)
    x = Conv2D(48, 5, 5, activation='elu', subsample=(2, 2))(x)
    x = Conv2D(50, 5, 5, activation='elu', subsample=(2, 2))(x)
    x = Conv2D(64, 3, 3, activation='elu')(x)
    x = Conv2D(76, 3, 3, activation='elu')(x)
    x = Conv2D(76, 3, 3, activation='elu')(x)
    x = Conv2D(76, 3, 3, activation='elu')(x)
    x = Dropout(args["keep_prob"])(x)
    x = Flatten()(x)
    x = Dense(256, activation = 'elu')(x)
    x = Dense(128, activation = 'elu')(x)
    x = Dense(100, activation = 'elu')(x)
    x = Dense(50, activation = 'elu')(x)
    x = Dense(10, activation = 'elu')(x)
    steering_angle = Dense(1, name="steering_angle")(x)
    speed = Dense(1, name="speed")(x)
    model = Model(input=image_input, output=[steering_angle,speed])

    model.summary()

    return model


def build_model_resnet50_pre_trained(args):
    """
    RESNET50 pre-trained Transfer Learning
    Get the model. Remove final layers and attach my own fully
    connected layers.
    Flatten : 2048
    Fully connected: 1024
    Fully connected: 512
    Fully connected: 256
    Fully connected: 128
    Fully connected: 64
    Fully connected: 1
    """
    #IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 480, 640, 3
#    image_input = Input(shape=(224, 224, 3))
    image_input = Input(shape=(480, 640, 3))
    base_model = ResNet50(input_tensor=image_input, include_top=False,weights='imagenet')
    base_model.summary()
    last_layer = base_model.get_layer('avg_pool').output
    x= Flatten(name='flatten')(last_layer)
    x = Dense(1024, activation = "elu")(x)
    x = Dense(512, activation = "elu")(x)
    x = Dense(256, activation = "elu")(x)
    x = Dense(128, activation = "elu")(x)
    x = Dense(64, activation = "elu")(x)
    out = Dense(1, activation='softmax', name='output_layer')(x)
    model = Model(inputs=image_input,outputs= out)
#    model = Model(input=image_input,output= out)
    model.summary()
    for layer in model.layers[:-7]:
        layer.trainable = False

    return model


def build_model_resnet50_fully(args):
    """
    RESNET50 pre-trained Transfer Learning
    Get the model. Remove final layers and attach my own fully
    connected layers.
    Flatten : 2048
    Fully connected: 1024
    Fully connected: 512
    Fully connected: 256
    Fully connected: 128
    Fully connected: 64
    Fully connected: 1
    """
    
#    image_input = Input(shape=(224, 224, 3))    
    image_input = Input(shape=(480, 640, 3))
    base_model = ResNet50(input_tensor=image_input, include_top=False,weights='imagenet')
    base_model.summary()
    last_layer = base_model.get_layer('avg_pool').output
    x= Flatten(name='flatten')(last_layer)
    x = Dense(1024, activation = "elu")(x)
    x = Dense(512, activation = "elu")(x)
    x = Dense(256, activation = "elu")(x)
    x = Dense(128, activation = "elu")(x)
    x = Dense(64, activation = "elu")(x)
    out = Dense(1, activation='softmax', name='output_layer')(x)
    model = Model(inputs=image_input,outputs= out)
#    model = Model(input=image_input,output= out)
    model.summary()
    return model

