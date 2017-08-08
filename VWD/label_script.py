#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 12:20:51 2017

@author: AlfredoLucas
"""

#%% Define the folder where the training data will be stored
global training_data_folder
training_data_folder = '/Users/AlfredoLucas/Documents/Trabajos/University/Cabrales/Videos/training_videos/training_data'

#%% Predefined functions

def create_rest_set(img,grid_size, refPt_vec):
        import numpy as np
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

def create_init_set(img,grid_size,refPt_vec):
    import numpy as np
    cropped_region = np.zeros(img.shape)
    set_vec = np.zeros((1,(grid*2)**2))
    for i in range(refPt_vec.shape[1]):
        cropped_region = img[(refPt_vec[1,i]-(grid_size)):(refPt_vec[1,i]+(grid_size)),(refPt_vec[0,i]-(grid_size)):(refPt_vec[0,i]+(grid_size))]
        set_vec = np.append(set_vec, np.reshape(cropped_region, (1,(grid_size*2)**2)), axis=0)
    return set_vec[1:,:]

def load_video(location):
    import os
    import numpy as np
    import cv2
    list_dir = os.listdir(location)
    list_dir = list_dir[1:]
    img_test = cv2.imread(location + '/' + list_dir[0])
    img = np.zeros((img_test.shape[0], img_test.shape[1],len(list_dir)),dtype='uint8')
    eq = np.zeros((img_test.shape[0], img_test.shape[1],len(list_dir)),dtype='uint8')
    count = 0
    for i in list_dir:
        frame = cv2.imread(location + '/' + i)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY )
        img[:,:,count] = frame
        eq[:,:,count] = cv2.equalizeHist(frame)
        count = count + 1
    return img

#%
def rest_label(img_vec, grid ,refPt_vec):
    import numpy as np
    import os
    for i in range(img_vec.shape[2]):
        set_vec = create_init_set(img_vec[:,:,i], grid,refPt_vec)
        rest_set_vec = create_rest_set(img_vec[:,:,i], grid, refPt_vec)
    os.chdir(training_data_folder)
    with open('training_data_1.csv', 'ab') as abc:
        np.savetxt(abc, set_vec, delimiter=",")
    with open('training_data_0.csv', 'ab') as abc:
        np.savetxt(abc, rest_set_vec, delimiter=",")

#%% Script

img_vec = load_video(location='/Users/AlfredoLucas/Documents/Trabajos/University/Cabrales/Videos/training_videos/1001')

#%%
import numpy as np
import cv2
import os
os.chdir('/Users/AlfredoLucas/Documents/Trabajos/University/Cabrales/PIV/piv-repo')
import piv_functions as pivf

refPt = []

#% Get the mouse location in the ROI
image = img_vec[:,:,0]
image = image.copy()
grid = pivf.get_optimal_grid(image.shape,20, 's')
def click_rect(event, x, y, flags, param):
    	# grab references to the global variables
        global refPt
    	# if the left mouse button was clicked, record the starting
    	# (x, y) coordinates and indicate that cropping is being
    	# performed
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            clone1 = cv2.rectangle(image, (refPt[0][0]-(grid),refPt[0][1]-(grid)), (refPt[0][0]+(grid),refPt[0][1]+(grid)), (0, 255, 0), 2).copy()
            cv2.imshow("image", clone1)


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
    
#% Segmentize the rest of the image as not being part of the training set
rest_set_vec = create_rest_set(clone.copy(),grid,refPt_vec=refPt_vec)
os.chdir(training_data_folder)
with open('training_data_0.csv', 'ab') as abc:
    np.savetxt(abc, rest_set_vec, delimiter=",")

for i in range(img_vec.shape[2]-1):
    set_vec = create_init_set(img_vec[:,:,i+1], grid, refPt_vec = refPt_vec)
    rest_set_vec = create_rest_set(img_vec[:,:,i+1])
    os.chdir(training_data_folder)
    with open('training_data_1.csv', 'ab') as abc:
        np.savetxt(abc, set_vec, delimiter=",")
    with open('training_data_0.csv', 'ab') as abc:
        np.savetxt(abc, rest_set_vec, delimiter=",")