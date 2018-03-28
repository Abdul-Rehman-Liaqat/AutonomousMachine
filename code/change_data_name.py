# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 18:23:21 2017

@author: Abdul Rehman
"""

import pandas as pd





path = r"/home/abdul/abdul/end-to-end/data/unity3d/track1"

data_df = pd.read_csv(path+"/driving_log.csv", names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

data_df["center"] = data_df["center"].apply(lambda x:path+"/IMG/"+x.split("/")[-1])
data_df["left"] = data_df["left"].apply(lambda x:path+"/IMG/"+x.split("/")[-1])
data_df["right"] = data_df["right"].apply(lambda x:path+"/IMG/"+x.split("/")[-1])

data_df.to_csv(path+"/driving_log.csv", header = None ,index = False)
