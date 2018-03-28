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
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from keras.layers import Lambda,Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input
import cv2
import matplotlib.image as mpimg
from all_models import *
from final_multitask_model_temp import *
np.random.seed(0)



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
    data_df = pd.read_csv(path+"/interpolated.csv")
    data_df["left"] = data_df["filename"].apply(lambda x:path+"/"+x if x.split("/")[0]=="left" else None)
    data_df["center"] = data_df["filename"].apply(lambda x:path+"/"+x if x.split("/")[0]=="center" else None)
    data_df["right"] = data_df["filename"].apply(lambda x:path+"/"+x if x.split("/")[0]=="right" else None)
    data_df = data_df.rename(columns={"angle":"steering_angle"})
    data_df["steering_angle"] = (data_df["steering_angle"] - data_df["steering_angle"].mean())/(data_df["steering_angle"].max()-data_df["steering_angle"].min())
    data_df["speed"] = (data_df["speed"] - data_df["speed"].mean())/(data_df["speed"].max()-data_df["speed"].min())
    return data_df

### Load Udacity dataset.
def load_test():
    path = os.path.abspath(os.getcwd()+"/../data/test")
    data_df = pd.read_csv(path+"/interpolated.csv")
    data_df = data_df.rename(columns={"frame_id":"center"})
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


#IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 480, 640, 3
#IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 224, 224, 3
input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

args = {"data_dir" : "data",
        "test_size" : 0.2,
        "keep_prob" : 0.5,
        "nb_epoch" : 10,
        "samples_per_epoch" : 2000,
        "batch_size" : 40,
        "save_best_only" : True,
        "learning_rate" : 1.0e-4,
        "data_type" : "udacity",
        "data_number" : 0,
        "input_shape" : input_shape,
        "target_variable" : ["steering_angle","speed"],
        "data_normalized" : True
        }

#print parameters
print('-' * 30)
print('Parameters')
print('-' * 30)
for key, value in args.items():
    print('{:<20} := {}'.format(key, value))
print('-' * 30)

#weights = "/nvidia_no_preprocessing/udacity0/model-047.h5" 
#weights = "/nvidia_multitask2_speed_normalized_only_50_epoch/udacity0/model-best.h5"
weights = "/model-best.h5"
cwd = os.getcwd()
path = os.path.abspath(cwd+weights)
data = load_test()
img_parent = os.path.abspath(cwd+"/../data/test/center")
#model = build_model_nvidia(args)
#model = build_model_resnet50_pre_trained(args)
#model = build_model_nvidia_multi_task2(args)
#model = build_model_nvidia_situation4(args)
#model = build_model_nvidia_multi_task(args)
#model = build_model_nvidia_situation_Manish(args)
model = build_model_nvidia_situation_Model_From_Scratch_With_Dropouts(args)
model.load_weights(path)
model.compile(loss='mean_squared_error', optimizer=Adam(lr=args["learning_rate"]))

prediction = []
diff = []
for i in range(len(data)):
    img_path = data.center.iloc[i]
    image = mpimg.imread(img_parent+"/"+str(img_path)+".jpg") 
#    image = preprocess(args,image)
    pred = model.predict(np.expand_dims(image, axis=0), batch_size = 1)
    prediction.append(pred)
    diff.append(pred[0][0][0]-data.steering_angle.iloc[i])
    print(diff[-1])
#    print(pred[0][0][0])
np.savetxt("situaion_from_scratch_with_dropouts.txt",diff, delimiter=",")
prediction =  pd.DataFrame(prediction, columns = ["prediction","actual"])
prediction.to_csv("situation_from_scratch_with_dropouts.csv", index = False)
new_diff =  np.array(diff)
mse = (sum(new_diff*new_diff)/len(new_diff))
rmse = np.sqrt(mse)
print(mse,rmse)
#score = model.evaluate(data[""], Y, verbose=0)
