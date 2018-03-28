#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 22:09:40 2017

@author: abdulliaqat
"""

import os
import numpy as np
import pandas as pd
### Load Udacity dataset.
def load_udacity(number):
    path = os.path.abspath(os.getcwd()+"/../data/udacity-"+str(number))
    data_df = pd.read_csv(path+"/interpolated.csv")
    target = "angle"
    data_df[target] = (data_df[target] - data_df[target].mean()) / (data_df[target].max() - data_df[target].min())
    target = "speed"
    data_df[target] = (data_df[target] - data_df[target].mean()) / (data_df[target].max() - data_df[target].min())
    data_df.to_csv(path+"/interpolated_normalized.csv", index = False)
    