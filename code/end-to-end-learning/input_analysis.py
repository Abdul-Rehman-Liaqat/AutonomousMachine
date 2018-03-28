#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 23:17:23 2017

@author: abdulliaqat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

test = pd.read_csv('../data/udacity-0/interpolated.csv')
test= test.rename(columns={'angle':'steering_angle'})


test['ind'] = test.index
test.sort_values('steering_angle', inplace = True)
test.reset_index(inplace = True, drop = True)
plt.figure(0)
plt.plot(test.index,test.steering_angle)
plt.show()
plt.figure(1)
plt.plot(test.index[15000:len(test)-15000],test.steering_angle[15000:len(test)-15000])
plt.show()


lower_limit = 15000
upper_limit = 15000

one = test.iloc[range(0,lower_limit)]
two = test.iloc[range(len(test)-upper_limit,len(test))]
three = test.iloc[range(lower_limit,len(test)-upper_limit)].sample(frac = 0.3).reset_index(drop = True)

final = pd.concat([one,three,two])
final.sort_values('steering_angle', inplace = True)
plt.figure(2)
plt.plot(range(len(final)),final.steering_angle)
plt.show()