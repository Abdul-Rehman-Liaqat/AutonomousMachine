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
from keras.layers import Lambda,Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input, TimeDistributed, BatchNormalization, LSTM
import matplotlib.image as mpimg
np.random.seed(0)

def pop(model):
    '''Removes a layer instance on top of the layer stack.
    This code is thanks to @joelthchao https://github.com/fchollet/keras/issues/2371#issuecomment-211734276
    '''
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')
    else:
        model.layers.pop()
        if not model.layers:
            model.outputs = []
            model.inbound_nodes = []
            model.outbound_nodes = []
        else:
            model.layers[-1].outbound_nodes = []
            model.outputs = [model.layers[-1].output]
        model.built = False

    return model

def build_model_nvidia_multi_task_Final(args):
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
    y = Conv2D(64, 3, 3, activation='elu')(x)
    y = Conv2D(64, 3, 3, activation='elu')(y)
    y = Dropout(args["keep_prob"])(y)
    y = Flatten()(y)
    y = Dense(100, activation = 'elu')(y)
    y = Dense(50, activation = 'elu')(y)
    y = Dense(10, activation = 'elu')(y)
    speed = Dense(1, name="speed")(y)    
    model = Model(input=image_input, output=[steering_angle,speed])
    model.summary()

    return model

def build_model_nvidia_situation1(args):
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
    speed = Dense(1)(x)    
    model = Model(input=image_input, output=[speed])
    model.load_weights("model-047.h5")
    model.summary()

    return model

def build_model_nvidia_situation4(args):
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
    image_input = Input(shape=(480, 640, 3))
    x = Conv2D(24, 5, 5, activation='elu', subsample=(2, 2))(image_input)
    x = Conv2D(36, 5, 5, activation='elu', subsample=(2, 2))(x)
    x = Conv2D(48, 5, 5, activation='elu', subsample=(2, 2), name = 'test1')(x)
    x = Conv2D(64, 3, 3, activation='elu')(x)
    x = Conv2D(64, 3, 3, activation='elu')(x)
    x = Dropout(args["keep_prob"])(x)
    x = Flatten()(x)
    x = Dense(100, activation = 'elu')(x)
    x = Dense(50, activation = 'elu')(x)
    x = Dense(10, activation = 'elu')(x)
    speed = Dense(1)(x)    
    model = Model(input=image_input, output=[speed])
#    model.load_weights("nvidia_speed_transfer_leanring_for_final_multitask_situation1/model-nvidia-speed-transfer-learning.h5")
    
    
    pop(model)
    pop(model)
    pop(model)
    pop(model)
    pop(model)
    pop(model) # Dropout Layer
    pop(model)
    pop(model)
    
    cnn_transfer_layer = model.get_layer('test1').output
    x = Conv2D(64, 3, 3, activation='elu')(cnn_transfer_layer)
    x = Conv2D(64, 3, 3, activation='elu')(x)
    x = Dropout(args["keep_prob"])(x)
    x = Flatten()(x)
    x = Dense(100, activation = 'elu')(x)
    x = Dense(50, activation = 'elu')(x)
    x = Dense(10, activation = 'elu')(x)
    steering_angle = Dense(1,name = 'steering_angle')(x)    
    
    y = Conv2D(64, 3, 3, activation='elu')(cnn_transfer_layer)
    y = Conv2D(64, 3, 3, activation='elu')(y)
    y = Dropout(args["keep_prob"])(y)
    y = Flatten()(y)
    y = Dense(100, activation = 'elu')(y)
    y = Dense(50, activation = 'elu')(y)
    y = Dense(10, activation = 'elu')(y)
    speed = Dense(1, name = 'speed')(y)    
    model = Model(input=image_input, output=[steering_angle,speed])
    for layer in model.layers[1:4]:
        layer.trainable = False
    model.summary()
    return model



def build_model_nvidia_situation_Manish(args):
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
    image_input = Input(shape=(480, 640, 3))
    x = Conv2D(24, 5, 5, activation='elu', subsample=(2, 2))(image_input)
    x = Conv2D(36, 5, 5, activation='elu', subsample=(2, 2))(x)
    x = Conv2D(48, 5, 5, activation='elu', subsample=(2, 2), name = 'test1')(x)
    x = Conv2D(64, 3, 3, activation='elu')(x)
    x = Conv2D(64, 3, 3, activation='elu')(x)
    x = Dropout(args["keep_prob"])(x)
    x = Flatten()(x)
    x = Dense(100, activation = 'elu')(x)
    x = Dense(50, activation = 'elu')(x)
    x = Dense(10, activation = 'elu')(x)
    steering_angle = Dense(1, name = 'steering_angle')(x)    
    model1 = Model(input=image_input, output=[steering_angle])
    model1.load_weights("nvidia_no_preprocessing/udacity0/model-047.h5")
            
    cnn_transfer_layer = model1.get_layer('test1').output
    y = Conv2D(64, 3, 3, activation='elu')(cnn_transfer_layer)
    y = Conv2D(64, 3, 3, activation='elu')(y)
    y = Dropout(args["keep_prob"])(y)
    y = Flatten()(y)
    y = Dense(100, activation = 'elu')(y)
    y = Dense(50, activation = 'elu')(y)
    y = Dense(10, activation = 'elu')(y)
    speed = Dense(1, name = 'speed')(y)    
    model2 = Model(input=image_input, output=[speed])
    model2.load_weights("situation1/model-best.h5")

    model3 = Model(input=image_input, output=[steering_angle,speed])

    model3.summary()
    return model3


def build_model_nvidia_situation_Model_From_Scratch_With_Dropouts(args):
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
    image_input = Input(shape=(480, 640, 3))
    z = Conv2D(24, 5, 5, activation='elu', subsample=(2, 2))(image_input)
    z = Conv2D(36, 5, 5, activation='elu', subsample=(2, 2))(z)
    z = Conv2D(48, 5, 5, activation='elu', subsample=(2, 2), name = 'test1')(z)
    x = Conv2D(64, 3, 3, activation='elu')(z)
    x = Conv2D(64, 3, 3, activation='elu')(x)
    x = Dropout(args["keep_prob"])(x)
    x = Flatten()(x)
    x = Dense(100, activation = 'elu')(x)
    x = Dropout(args["keep_prob"])(x)
    x = Dense(50, activation = 'elu')(x)
    x = Dropout(args["keep_prob"])(x)
    x = Dense(10, activation = 'elu')(x)
    steering_angle = Dense(1, name = 'steering_angle')(x)    
            
    y = Conv2D(64, 3, 3, activation='elu')(z)
    y = Conv2D(64, 3, 3, activation='elu')(y)
    y = Dropout(args["keep_prob"])(y)
    y = Flatten()(y)
    y = Dense(100, activation = 'elu')(y)
    y = Dropout(args["keep_prob"])(y)
    y = Dense(50, activation = 'elu')(y)
    y = Dropout(args["keep_prob"])(y)
    y = Dense(10, activation = 'elu')(y)
    speed = Dense(1, name = 'speed')(y)    
    model3 = Model(input=image_input, output=[steering_angle,speed])
    model3.summary()
    return model3


def build_model_nvidia_lstm_mix(args):
    model = Sequential()
    model.add(TimeDistributed(Lambda(lambda x: x/127.5-1.0), input_shape=args["input_shape"]))
    model.add(TimeDistributed(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2))))
    model.add(TimeDistributed(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2))))
    model.add(TimeDistributed(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2))))
    model.add(TimeDistributed(Conv2D(64, 3, 3, activation='elu')))
    model.add(TimeDistributed(Conv2D(64, 3, 3, activation='elu')))
    model.add(Dropout(args["keep_prob"]))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(100, activation='elu')))
    model.add(TimeDistributed(Dense(50, activation='elu')))
    model.add(TimeDistributed(Dense(10, activation='elu')))
    model.add(TimeDistributed(Dense(1))) 
    # model.summary()

#     load weights
#     model.load_weights("/home/abdul/abdul/Final1/code/nvidia_no_preprocessing/udacity0/model-047.h5")
    ##Remove last 3 layers
    pop(model)
    pop(model)
    pop(model)
    ##Add new LSTM layers
#     model.add(LSTM(100, return_sequences=True, activation='elu'))
    model.add(BatchNormalization())
    model.add(LSTM(100, activation='elu'))
    model.add(Dropout(args["keep_prob"]))
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(args["keep_prob"]))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(args["keep_prob"]))
    model.add(Dense(10, activation='elu'))  
    model.add(Dense(1)) 
    model.summary()

    for layer in model.layers[:-9]:
        layer.trainable = False
    # for layer in model.layers:
    #     print(layer, ':' , layer.trainable) 
    return model
