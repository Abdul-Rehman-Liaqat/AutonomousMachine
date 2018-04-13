# -*- coding: utf-8 -*-
"""
Created on Tue Oct 1 22:12:11 2017

@author: Abdul Rehman
"""

import cv2
import numpy as np

#path = r"C:\Users\Abdul Rehman\Desktop\Research Project Second Presentation\Data\udacity\Ch2_001\center\1479425464536723116.jpg"

# =============================================================================
# Image Read
# =============================================================================
#img1 = cv2.imread(path)
path = r"test.jpeg"
img1 = cv2.imread(path)
height , width , layers =  img1.shape


def display_image(img):
    cv2.namedWindow('image_windows', cv2.WINDOW_NORMAL)
    cv2.imshow('image_windows', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
display_image(img1)
# =============================================================================
# Extract useful colors
# =============================================================================
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
'''
light_y = np.array([20, 100, 100], dtype = np.int8)
dark_y = np.array([30, 255, 255], dtype = np.int8)

hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
yellow = cv2.inRange(hsv, light_y, dark_y)
white = cv2.inRange(gray, 230, 255)
yw = cv2.bitwise_or(white, yellow)
#mask = yellow + white
mask = cv2.bitwise_or(white, yellow)
masked = gray.copy()
masked[np.argwhere(white == 0)] = 0
display_image(masked)
'''

# =============================================================================
# Detect Edges
# =============================================================================
edged = cv2.Canny(gray,100, 200)
display_image(edged)

# =============================================================================
# Cropping road from images
# =============================================================================
    
polygon =np.array( [[(0, height), (width / 3, height/2 )
                    , (width / 1.5, height /2 ),(width, height)]], np.int32)

mask = np.zeros_like(gray)
match_mask_color = 255
cv2.fillPoly(mask, polygon, match_mask_color)
crop = cv2.bitwise_and(edged, mask)
display_image(crop)

# =============================================================================
# Extrapolate Lines and combine them
# =============================================================================
extrapolated = cv2.HoughLinesP(
        crop,rho = 6, theta = np.pi/60, threshold = 160, lines = np.array([]),
        minLineLength = 40, maxLineGap = 25)


#def draw_lines(img1, extrapolated, color=[0, 255 , 0], thickness=1):
color=[0, 255 , 0]
temp = np.copy(img1)
line_img = np.zeros((temp.shape[0], temp.shape[1],3),dtype=np.uint8)
for line in extrapolated:
    for x1, y1, x2, y2 in line:
        cv2.line(line_img, (x1, y1), (x2, y2), color, 2)
temp = cv2.addWeighted(temp, 0.4 , line_img, 1.0, 0.0)
display_image(temp)