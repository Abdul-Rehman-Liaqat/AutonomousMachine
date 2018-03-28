#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:11:46 2017

@author: abdulliaqat
"""

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Input
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import plot_model


#other
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
from os import listdir, rename, mkdir, remove
from os.path import isfile, join, isdir, exists
import image as image_utils #some helper functions
import pydot
import pickle
#import _pickle as cPickle
import re

#force CPU usage (instead of slow notebook GPU)
import os



# 1. load the pre-trained Inception V3 model (without top layers)
#base_model = InceptionV3(weights='imagenet', include_top=False)
base_model = ResNet50(weights = "imagenet", include_top = False)
plot_model(base_model, to_file='model.png')
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 480, 640, 3
#input_shape = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
input_shape = Input(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

image_input = Input(shape=(224, 224, 3))

model = ResNet50(input_tensor=image_input, include_top=True,weights='imagenet')
model.summary()
last_layer = model.get_layer('avg_pool').output
x= Flatten(name='flatten')(last_layer)
x = Dense(1024, activation = "elu")(x)
x = Dense(512, activation = "elu")(x)
x = Dense(256, activation = "elu")(x)
x = Dense(128, activation = "elu")(x)
x = Dense(64, activation = "elu")(x)
out = Dense(1, activation='softmax', name='output_layer')(x)
custom_resnet_model = Model(inputs=image_input,outputs= out)
custom_resnet_model.summary()
for layer in custom_resnet_model.layers[:-1]:
	layer.trainable = False

pre_trained = base_model.output
x = GlobalAveragePooling2D()(pre_trained)
#x = Dense(500, activation='elu')(x)
#x = Flatten()(pre_trained)
#x = Dense(500, activation="elu")(x)
#x = Dense(100, activation="elu")(x)
#x = Dense(50, activation="elu")(x)
#x = Dense(10, activation="elu")(x)
out = Dense(1, activation=None)(pre_trained)
my_model = Model(inputs=input_shape,outputs=out)

base_model.add(Dense(500, activation="elu"))
base_model.add(Dense(100, activation="elu"))
base_model.add(Dense(50, activation="elu"))
base_model.add(Dense(10, activation="elu"))
base_model.add(Dense(1, activation="elu"))
plot_model(base_model, to_file='model1.png')

y = Model(base_model,L)

# 2. add custom top-layers for multilabel problem
x = base_model.output
#x = GlobalAveragePooling2D()(x)
x = GlobalAveragePooling2D()(x)
x = Dense(500, activation='elu')(x)
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
predictions = Dense(num_classes, activation='sigmoid')(x)

# 3. combine to one model
model = Model(inputs=base_model.input, outputs=predictions)

# 4. we want to first train the top layers and keep the other weights frozen
for layer in base_model.layers:
    layer.trainable = False

# 5. compile the new model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])

#6. train model
epochs = 1 #usually around 15 
model.fit_generator(train_generator, validation_data=val_generator, steps_per_epoch = train_steps, 
                    epochs= epochs, class_weight = class_weights, validation_steps=val_steps)  

#7. save model
model.save("inception_paw.model")

deep_epochs = 1 #usually around 15

# 1. train the top 2 inception blocks, i.e. we will freeze the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True

# 2. recompile the model for these modifications to take effect
# use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])

# 3. Train the model again (this time fitting the last 2 inception blocks and the top layers)
model.fit_generator(train_generator, validation_data=val_generator, steps_per_epoch = train_steps, 
                    epochs=deep_epochs, class_weight = class_weights, 
                    validation_steps=val_steps)  
#save model
model.save("inception_paw_deeper.model")