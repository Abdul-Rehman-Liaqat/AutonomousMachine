#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:45:56 2017

@author: abdulliaqat
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 22:33:16 2017

@author: abdulliaqat
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 14:42:13 2017

@author: abdulliaqat
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 3 22:31:12 2017

@author: abdulliaqat
"""

import pandas as pd
import numpy as np
import os
#from sklearn.model_selection import train_test_split 
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.applications.resnet50 import ResNet50
from keras.layers import Lambda,Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input
import cv2
import matplotlib.image as mpimg
from all_models import *
from final_multitask_model_temp import *
np.random.seed(0)
import keras.backend as K

def customLoss(yTrue,yPred):
    steering_angle = (yPred[0]-yTrue[0])*(yPred[0]-yTrue[0])
    speed = (yPred[1]-yTrue[1])*(yPred[1]-yTrue[1])
    return steering_angle + 0.001*speed


### Load Udacity dataset.
def load_udacity(number):
    path = os.path.abspath(os.getcwd()+"/../data/udacity-"+str(number))
    data_df = pd.read_csv(path+"/interpolated.csv")
    data_df["left"] = data_df["filename"].apply(lambda x:path+"/"+x if x.split("/")[0]=="left" else None)
    data_df["center"] = data_df["filename"].apply(lambda x:path+"/"+x if x.split("/")[0]=="center" else None)
    data_df["right"] = data_df["filename"].apply(lambda x:path+"/"+x if x.split("/")[0]=="right" else None)
    data_df = data_df.rename(columns={"angle":"steering_angle"})
    return data_df

### Load Udacity dataset.
def load_udacity_normalized(number):
    path = os.path.abspath(os.getcwd()+"/../data/udacity-"+str(number))
    data_df = pd.read_csv(path+"/interpolated_normalized.csv")
    data_df["left"] = data_df["filename"].apply(lambda x:path+"/"+x if x.split("/")[0]=="left" else None)
    data_df["center"] = data_df["filename"].apply(lambda x:path+"/"+x if x.split("/")[0]=="center" else None)
    data_df["right"] = data_df["filename"].apply(lambda x:path+"/"+x if x.split("/")[0]=="right" else None)
    data_df = data_df.rename(columns={"angle":"steering_angle"})
    return data_df


### Load Udacity dataset.
def load_udacity_normalized2(number):
    path = os.path.abspath(os.getcwd()+"/../data/udacity-"+str(number))
    data_df = pd.read_csv(path+"/interpolated_normalized.csv")
    data_df["left"] = data_df["filename"].apply(lambda x:path+"/"+x if x.split("/")[0]=="left" else None)
    data_df["center"] = data_df["filename"].apply(lambda x:path+"/"+x if x.split("/")[0]=="center" else None)
    data_df["right"] = data_df["filename"].apply(lambda x:path+"/"+x if x.split("/")[0]=="right" else None)
    data_df = data_df.rename(columns={"angle":"steering_angle"})
    path = os.path.abspath(os.getcwd()+"/../data/udacity-"+str(number))
    data = pd.read_csv(path+"/interpolated.csv")
    data = data.rename(columns={"angle":"steering_angle"})
    data_df["steering_angle"] = data["steering_angle"]
    return data_df



### Load Unity3d dataset.
def load_unity3d(number):
    path = os.path.abspath(os.getcwd()+"/../data/unity3d-"+str(number))
    data_df = pd.read_csv(path+"/driving_log.csv", 
                                  names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    data_df["left"] = data_df["left"].apply(lambda x:path+"/IMG/"+x.split("/")[-1]) 
    data_df["right"] = data_df["right"].apply(lambda x:path+"/IMG/"+x.split("/")[-1])  
    data_df["center"] =  data_df["center"].apply(lambda x:path+"/IMG/"+x.split("/")[-1]) 
    def quick_transform(x):
        t = x.split("/")[-1].split("_")[1:]
        return "-".join(t[0:3])+" "+":".join(t[3:6])+"."+t[-1].split(".")[0]
    data_df["index"] = pd.to_datetime(data_df["center"].apply(lambda x:quick_transform(x)))
    data_df = data_df.rename(columns={"steering":"steering_angle"})
    return data_df

def load_data_center(args):
    if(args["data_type"] == "udacity"):
        data_df = load_udacity(args["data_number"])

    elif(args["data_type"] == "unity3d"):
        data_df = load_unity3d(args["data_number"])

    df = data_df[["center"]+args["target_variable"]].dropna()
    X = df['center'].values
    y = df[args["target_variable"]].values            
    print(df.head())
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args["test_size"], random_state=0)
    return X_train, X_valid, y_train, y_valid

def load_data(args):    
    if(args["data_type"] == "udacity"):
        data_df = load_udacity(args["data_number"])

    elif(args["data_type"] == "unity3d"):
        data_df = load_unity3d(args["data_number"])

    df = data_df[["center","left","right"]+args["target_variable"]].dropna()
    X = df['center'].values
    y = df[args["target_variable"]].values            
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args["test_size"], random_state=0)
    return X_train, X_valid, y_train, y_valid


def preprocess(args,image):
#    image = image[60:-25, :, :]
    ##### args["input_shape"] is a tupple (height,width,channels)
#    image = cv2.resize(image, (args["input_shape"][1], args["input_shape"][0]), cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image

def batch_generator(args,data_dir, image_paths, target, batch_size, is_training, data_type = "unity3d"):
    images_batch = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    target_batch0 = np.empty(batch_size)
    target_batch1 = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            img_path = image_paths[index]
            relevant_target0 = target[index][0]
            relevant_target1 = target[index][1]
            image = mpimg.imread(img_path) 
            # add the image and steering angle to the batch
            images_batch[i] = preprocess(args,image)
            target_batch0[i] = relevant_target0
            target_batch1[i] = relevant_target1
            i += 1
            if i == batch_size:
                break
        yield images_batch, [target_batch0,target_batch1]
 

def train_model(model, args, X_train, X_valid, y_train, y_valid,data_type="unity3d"):
    checkpoint = ModelCheckpoint('model-best-final_multitask_transfer_learned_weighted_loss.h5', monitor='val_steering_angle_loss', verbose=1, save_best_only=args['save_best_only'], mode='auto')
    model.compile(loss=customLoss, optimizer=Adam(lr=args["learning_rate"]))
    csv_logger = CSVLogger('model-best-final_multitask__transfer_learned_weighted_loss.log', separator = ',', append = True)
    history_callback = model.fit_generator(batch_generator(args,args["data_dir"], X_train, y_train, args["batch_size"], True,data_type=data_type),
                        args["samples_per_epoch"],
                        args["nb_epoch"],
                        max_q_size=1,
                        validation_data=batch_generator(args,args["data_dir"], X_valid, y_valid, args["batch_size"], False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint,csv_logger],
                        verbose=1)
#for command line args
def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 480, 640, 3
input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

args = {"data_dir" : "data",
        "test_size" : 0.2,
        "keep_prob" : 0.5,
        "nb_epoch" : 50,
        "samples_per_epoch" : 20000,
        "batch_size" : 40,
        "save_best_only" : True,
        "learning_rate" : 1.0e-4,
        "data_type" : "udacity",
        "data_number" : 0,
        "input_shape" : input_shape,
        "target_variable" : ["steering_angle","speed"]
        }

#    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
#    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
#    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
#    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
#    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=2)
#    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
#    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
#    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
#    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
#    args = vars(parser.parse_args())

#print parameters
print('-' * 30)
print('Parameters')
print('-' * 30)
for key, value in args.items():
    print('{:<20} := {}'.format(key, value))
print('-' * 30)

data = load_data_center(args)
#model = build_model_nvidia(args)
#model = build_model_resnet50_pre_trained(args)
#model = build_model_nvidia_multi_task(args)
#model = build_model_nvidia_multi_task2(args)
#model = build_model_nvidia_situation1(args)
#model = build_model_nvidia_situation4(args)
model = build_model_nvidia_situation_Manish(args)
#model = build_model_nvidia_situation_Model_From_Scratch_With_Dropouts(args)
##train model on data, it saves as model.h5
train_model(model, args, *data,data_type=args["data_type"])