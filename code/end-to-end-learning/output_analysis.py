#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 23:59:39 2017

@author: abdulliaqat
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 23:17:23 2017

@author: abdulliaqat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

path = 'nvidia_multitask2_speed_normalized_only_50_epoch_trial2/multitask2_speed_only_normalized_diff.txt'
diff = pd.DataFrame()
diff['val'] = np.loadtxt(path)
diff['ind'] = range(len(diff.val))
diff.sort_values('val', inplace = True)
diff.reset_index(inplace = True, drop = True)

plt.plot(diff.index, diff.val)
plt.show()



img_parent = '../data/test/'
test_interpolated = pd.read_csv(img_parent + 'interpolated.csv')

culprit_images = test_interpolated['frame_id'].iloc[diff.ind.iloc[-10:-1]]

for img_path in culprit_images:
    img = mpimg.imread(img_parent+'center/'+str(img_path)+'.jpg')
    plt.figure(img_path)
    plt.imshow(img)
    plt.show()