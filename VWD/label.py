#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:17:38 2017

@author: AlfredoLucas
"""

#%% 
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
os.chdir('/Users/AlfredoLucas/Documents/Trabajos/University/Cabrales/PIV/piv-repo')
import piv_functions as pivf
#%% Predefined functions
def load_video(location, filename, start_frame, n_frames, x_length, y_length):
    import numpy as np
    import cv2
    img = np.zeros((x_length[1]-x_length[0],y_length[1]-y_length[0],n_frames),dtype='uint8')
    eq = np.zeros((x_length[1]-x_length[0],y_length[1]-y_length[0],n_frames),dtype='uint8')
    for i in range(start_frame,start_frame+n_frames):
        frame = cv2.imread(location+'/'+filename+str(i)+'.tif')
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY )
        img[:,:,i-start_frame] = frame[x_length[0]:x_length[1],y_length[0]:y_length[1]]
        eq[:,:,i-start_frame] = cv2.equalizeHist(frame[x_length[0]:x_length[1],y_length[0]:y_length[1]])
    return img

#%% Define the cropping function for the region of interes
'''
Code taken from: http://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
'''
# import the necessary packages
import argparse
import cv2
training_data_folder = '/Users/AlfredoLucas/Documents/Trabajos/University/Cabrales/Videos/training_videos/training_data'
video_folder = '/Users/AlfredoLucas/Documents/Trabajos/University/Cabrales/Videos/training_videos/1001'
os.chdir(video_folder)

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []

#% Get the mouse location in the ROI
image = cv2.imread('img10001.jpg')
grid = pivf.get_optimal_grid(image.shape,20, 's')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
def click_rect(event, x, y, flags, param):
    	# grab references to the global variables
        global refPt
    	# if the left mouse button was clicked, record the starting
    	# (x, y) coordinates and indicate that cropping is being
    	# performed
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            cv2.rectangle(image, (refPt[0][0]-(grid),refPt[0][1]-(grid)), (refPt[0][0]+(grid),refPt[0][1]+(grid)), (0, 255, 0), 2)
            cv2.imshow("image", image)



cropped_region = np.zeros((grid*2,grid*2,1000))
set_vec = np.zeros((1,(grid*2)**2))
refPt = [(0,0)]
refPt_x = []
refPt_y = []
cv2.setMouseCallback("image", click_rect)
cv2.namedWindow("image")
clone = image.copy()
temp_loc = [(0,0)] 
count = 0

while True: 
	# display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    if refPt[0]!=temp_loc[0]:
            temp_loc = refPt
            refPt_x.append(refPt[0][0])
            refPt_y.append(refPt[0][1])
            cropped_region[:,:,count] = clone[(refPt[0][1]-(grid)):(refPt[0][1]+(grid)),(refPt[0][0]-(grid)):(refPt[0][0]+(grid))]
            set_vec = np.append(set_vec, np.reshape(cropped_region[:,:,count], (1,(grid*2)**2)), axis=0)
            count = count + 1
    
    	# if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()
        cropped_region = np.zeros((grid*2,grid*2,1000))
        refPt_vec = []
    	# if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
    		break

refPt_vec = np.concatenate(([refPt_x],[refPt_y]),axis=0)
set_vec = set_vec[1:,:]
# Save the vectorized image to a csv file. Append to the existing file
os.chdir(training_data_folder)
with open('training_data_1.csv', 'ab') as abc:
    np.savetxt(abc, set_vec, delimiter=",")
    
#%% Segmentize the rest of the image as not being part of the training set
def create_rest_set(img,grid_size, refPt_vec):
    grid_size = grid_size*2
    selected_region = np.zeros((grid_size,grid_size))
    rest_set_vec = np.zeros((1,(grid_size)**2))
    for i in range(int(img.shape[0]/grid_size)-1):
        for j in range(int(img.shape[1]/grid_size)-1):
            x_range = [i*grid_size, (i+1)*grid_size]
            y_range = [j*grid_size, (j+1)*grid_size]
            x_range_st = [np.matlib.repmat(i*grid_size,1,refPt_vec.shape[1])]
            x_range_ed = [np.matlib.repmat((i+1)*grid_size,1,refPt_vec.shape[1])]
            y_range_st = [np.matlib.repmat(j*grid_size,1,refPt_vec.shape[1])]
            y_range_ed = [np.matlib.repmat((j+1)*grid_size,1,refPt_vec.shape[1])]
            if not ((((y_range_st > (refPt_vec[0,:]-grid_size)).any() or (y_range_ed > (refPt_vec[0,:]-grid_size)).any()) and 
                     ((y_range_st < (refPt_vec[0,:]+grid_size)).any() or (y_range_ed < (refPt_vec[0,:]+grid_size)).any())) and 
                    (((x_range_st > (refPt_vec[1,:]-grid_size)).any() or (x_range_ed > (refPt_vec[1,:]-grid_size)).any()) and 
                     ((x_range_st < (refPt_vec[1,:]-grid_size)).any() or (x_range_ed < (refPt_vec[1,:]-grid_size)).any()))):
                selected_region = img[x_range[0]:x_range[1],y_range[0]:y_range[1]]  
                rest_set_vec = np.append(rest_set_vec, np.reshape(selected_region[:,:], (1,(grid_size)**2)), axis=0)
    return rest_set_vec

rest_set_vec = create_rest_set(clone.copy(),grid,refPt_vec=refPt_vec)
os.chdir(training_data_folder)
with open('training_data_0.csv', 'ab') as abc:
    np.savetxt(abc, rest_set_vec, delimiter=",")