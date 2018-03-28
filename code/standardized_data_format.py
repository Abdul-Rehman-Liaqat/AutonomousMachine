import pandas as pd
import argparse
import os
import pdb

        #for command line args
def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'

#parser = argparse.ArgumentParser(description='End-to-End Learning')
#parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
#parser.add_argument('-y', '--main', help='main type',          dest='main_type',         type=str,   default='unity3d')
#parser.add_argument('-z', '--sub', help='sub type',            dest='sub_type',          type=str,    default='1')
#parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
#parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
#parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
#parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
#parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
#parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
#parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
#args = parser.parse_args()


#### add new data set!
#### 1- add in data_address
#### 2- add if describing data and names into load_data function
data_address = {"unity3d":[1,2],
                "udacity":[1,2,3,4,5,6]}


### Load Udacity dataset.
def load_udacity(number):
    path = os.path.abspath(os.getcwd()+"/../data/udacity-"+str(number))
    data_df = pd.read_csv(path+"/interpolated.csv")      
    data_df["filename"] = data_df["filename"].apply(lambda x:path+"/"+x) 
    print(data_df.head())
    return data_df

### Load Unity3d dataset.
def load_unity3d(number):
    print("here")
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
    
    # Selecting useful columns
    return data_df
### Same pre-processing on images




#### load data folder and extract useful information into final format
def load_data(args, data_address):
    keys = list(data_address.keys())
#    print(keys)
#    print(args.main_type)
    if(not(args.main_type in keys)):
        print("main folder not correct")
    elif(not(int(args.sub_type) in data_address[args.main_type])):
        print("sub folder not correct")        
    else:
        path = os.path.abspath(os.getcwd()+"/../data/"+args.main_type+"-"+args.sub_type)
        if(str(args.main_type) == keys[0]):
            data_df = pd.read_csv(path+"/driving_log.csv", 
                                  names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
            print(data_df.head())
        elif(str(args.main_type) == keys[1]):
            print("yet to do something")
            
            
