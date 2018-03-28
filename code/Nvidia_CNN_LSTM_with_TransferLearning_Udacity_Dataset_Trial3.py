
# coding: utf-8

# #### Using Transfer Learning for Nvidia CNN +LSTM model by freezing all layers of CNN network and using the weights for the best trained model for Nvidia CNN net 

# In[1]:

import pandas as pd
import numpy as np
import os
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.applications.resnet50 import ResNet50
from keras.layers import Lambda,Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input, LSTM, BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras import regularizers
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# from all_models import *
np.random.seed(0)


# In[2]:

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 480, 640, 3
TIME_STEP = 5
image_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
input_shape = (TIME_STEP, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

args = {"data_dir" : "/home/abdul/abdul/Final1/data/udacity-0/data",
        "validation_size" : 0.2, 
        "keep_prob" : 0.5,
        "nb_epoch" : 20,
        "samples_per_epoch" : 10000,
        "batch_size" : 20,
        "save_best_only" : True,
        "learning_rate" : 1.0e-4,
        "data_type" : "udacity",
        "path" : '/home/abdul/abdul/Final1/data/udacity-0',
        "image_shape" : image_shape,
        'input_shape' : input_shape,
        "target_variable" : ["steering_angle"]
        }


# In[3]:

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


# In[16]:

def build_model_nvidia_wt_LSTM_TL2(args):
    
    ##Nvidia Network
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

    # load weights
    model.load_weights("/home/abdul/abdul/Final1/code/nvidia_no_preprocessing/udacity0/model-047.h5")
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

# In[8]:

def load_udacity(path):
    path = path          ##'/home/abdul/abdul/Final1/data/udacity-0'
    data_df = pd.read_csv(path+"/interpolated.csv")
    data_df["left"] = data_df["filename"].apply(lambda x:path+"/"+x if x.split("/")[0]=="left" else None)
    data_df["center"] = data_df["filename"].apply(lambda x:path+"/"+x if x.split("/")[0]=="center" else None)
    data_df["right"] = data_df["filename"].apply(lambda x:path+"/"+x if x.split("/")[0]=="right" else None)
    data_df = data_df.rename(columns={"angle":"steering_angle"})
    return data_df


# In[9]:

# change matrix X[NoOfSample, width,hight,channel] to X'[NoOfSample,timestep, width,hight,channel]
## X and Y in array format
def create_timestep_dataset(X,Y, timestep):
    dataX, dataY = [], []
    for i in range(len(X) -timestep+1 ):
        dataX.append(X[i:i+timestep,].tolist())
        dataY.append(Y[i+timestep-1,])
    return np.array(dataX), np.array(dataY)


# In[10]:

def load_data(args):    
    if(args["data_type"] == "udacity"):
        data_df = load_udacity(args["path"])

#     elif(args["data_type"] == "unity3d"):
#         data_df = load_unity3d(args["data_number"])

    df = data_df[["center"]+args["target_variable"]].dropna()
    X = df['center'].values
    y = df[args["target_variable"]].values.flatten()   
    
#    X_train = X[0:np.int(np.floor(len(X)*.8))]         ## 80-20 SPLIT FOR TEST AND TRAIN
#    X_test = X[np.int(np.floor(len(X)*.8)):]
#    y_train = y[0:np.int(np.floor(len(y)*.8))]
#    y_test = y[np.int(np.floor(len(y)*.8)):]
#    
#    X_train, y_train = create_timestep_dataset(X_train,y_train, 5) ##Creating timestep data with lag 10
#    X_test, y_test = create_timestep_dataset(X_test,y_test, 5)

    X_train, y_train = create_timestep_dataset(X,y, 5) ##Creating timestep data with lag 10
     
     
    X_truetrain, X_valid, y_truetrain, y_valid = train_test_split(X_train, y_train, test_size=args["validation_size"], random_state=0)
    return X_truetrain, y_truetrain, X_valid, y_valid


# In[11]:

def preprocess(args,image):
#    image = image[60:-25, :, :]
    ##### args["image_shape"] is a tupple (height,width,channels)
    image = cv2.resize(image, (args["image_shape"][1], args["image_shape"][0]), cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image


# In[12]:

def batch_generator(args,data_dir, image_paths, target, batch_size, is_training, data_type = "unity3d"):
    images_batch = np.empty([batch_size, TIME_STEP, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    target_batch = np.empty(batch_size)      
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            for t in range(image_paths.shape[1]):
                img_path = image_paths[index][t]
                relevant_target = target[index]
                image = mpimg.imread(img_path) 
                # add the image and steering angle to the batch
                images_batch[i,t] = preprocess(args,image)
                target_batch[i] = relevant_target
            i += 1
            if i == batch_size:
                break
        yield images_batch, target_batch        


# In[13]:

def train_model(model, args, X_train, y_train, X_valid,  y_valid,data_type):
    
    # checkpoint
    checkpoint_filepath="Nvidia_wt_LSTM/weights_TL3.hdf5"
    checkpoint = ModelCheckpoint(checkpoint_filepath,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=args["save_best_only"],
                                 mode='auto')
    ##CSV Logger
    csv_logger = CSVLogger('Nvidia_wt_LSTM/training_TL3.log',append=True)
    
    callbacks_list = [checkpoint, csv_logger]
    
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args["learning_rate"]))
    history_callback = model.fit_generator(batch_generator(args,args["data_dir"], X_train, y_train, args["batch_size"], True,data_type=data_type),
                        args["samples_per_epoch"],
                        args["nb_epoch"],
                        max_q_size=1,
                        validation_data=batch_generator(args,args["data_dir"], X_valid, y_valid, args["batch_size"], False),
                        nb_val_samples=len(X_valid),
                        callbacks=callbacks_list,
                        verbose=1)
    print("-"*10)
    np.save('Nvidia_wt_LSTM/historyhistory_TL2.npy', history_callback.history)


# #### Running the MOdel

# In[14]:

X_train, y_train, X_valid, y_valid = load_data(args)


# In[17]:

model = build_model_nvidia_wt_LSTM_TL2(args)


# In[19]:

train_model(model, args, X_train, y_train, X_valid, y_valid, data_type = args["data_type"])





