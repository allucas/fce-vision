#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 21:53:44 2017

@author: AlfredoLucas
"""

#%% Import
from scipy.interpolate import interp1d
from scipy import signal
import scipy
import cv2
import numpy as np
import os

#%% Load functions

def load_video(location, filename, start_frame, n_frames, x_length, y_length):
    import numpy as np
    import cv2
    img = np.zeros((x_length[1]-x_length[0],y_length[1]-y_length[0],n_frames),dtype='uint8')
    eq = np.zeros((x_length[1]-x_length[0],y_length[1]-y_length[0],n_frames),dtype='uint8')
    for i in range(start_frame,start_frame+n_frames):
        frame = cv2.imread(location+'/'+filename+str(i)+'.jpg')
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY )
        img[:,:,i-start_frame] = frame[x_length[0]:x_length[1],y_length[0]:y_length[1]]
        eq[:,:,i-start_frame] = cv2.equalizeHist(frame[x_length[0]:x_length[1],y_length[0]:y_length[1]])
    return img

def get_init_cfl(img, step):
    import numpy as np
    import cv2
    # Find the difference between adjacent intesity points
    diff = np.zeros((img.shape[0],int(img.shape[1]-(step+1)),img.shape[2]))
    for k in range(img.shape[2]):
        frame = cv2.GaussianBlur(img[:,:,k],(11,11),5)
        diff[:,:,k] = frame[:,step:img.shape[1]-1].astype(float) - frame[:,0:img.shape[1]-(1+step)].astype(float)
    loc_cfl = np.zeros((diff.shape[0],2,diff.shape[2]))
    for i in range(diff.shape[2]):
       for j in range(diff.shape[0]):
           left = np.where(diff[j,:,i] == diff[j,:,i].min())
           left = left[0][0] # Define the left boundary of the cell free layer
           right = np.where(diff[j,:,i] == diff[j,:,i].max())
           right = right[0] 
           right = right[right>left]
           if len(right)>0:
               right = right[0] # Define the right boundary of the cell free layer
           elif len(right)==0:
               left= 0
               right = 0
           loc_cfl[j,0,i] = left
           loc_cfl[j,1,i] = right
    return loc_cfl




#%% Initial Parameters
cal_factor = int(input('Enter calibration factor (in m/px): '))
filename = input('filename: ') # Image names
n_frames = 2 # Number of frames to load and average over the PIV results
y_length = [0,1000] # Y limits of the image
x_length = [0,1000] # X limits of the image
start_frame = 10001 # Starting frame number on the filename
filling = 1 # Choose how to fill the missing values (1 is through interpolation, 2 is through mean weighted filter)

#%% Load the video
#img = load_video(location=location,filename=filename, n_frames=n_frames, x_length=x_length, y_length=y_length, start_frame=start_frame)
img_1 = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_RGB2GRAY )
img = np.zeros((img_1.shape[0],img_1.shape[1],2), dtype='uint8')
img[:,:,0] = img_1
img[:,:,1] = img_1

#img = cv2.medianBlur(img,5)
#%%
loc_cfl =  get_init_cfl(img,3)
hyst = 15
step = 4
cell_width = 30
mean_left = np.median(loc_cfl[:,0,:],axis=0)
mean_right = np.median(loc_cfl[:,1,:],axis=0)
diff = np.zeros((img.shape[0],int(img.shape[1]-(step+1)),img.shape[2]))
for k in range(img.shape[2]):
    frame = cv2.GaussianBlur(img[:,:,k],(11,11),5)
    diff[:,:,k] = frame[:,step:img.shape[1]-1].astype(float) - frame[:,0:img.shape[1]-(1+step)].astype(float)
loc_cfl = np.zeros((diff.shape[0],2,diff.shape[2]))

for i in range(diff.shape[2]):
   prev_left = mean_left[i]
   prev_right = mean_right[i]
   for j in range(diff.shape[0]):
       left = np.where(diff[j,:,i] <= diff[j,:,i].min()+hyst)[0]
       diff_val_left = diff[j,left,i]
       left_diff = np.abs(left-mean_left[i])
       loc_left_diff = np.where(left_diff<cell_width)[0]
       if len(loc_left_diff) == 0:  
           left = 0
           disp_left = 0
           temp_left = 0
       else:
           prev_left = left[loc_left_diff[np.where(diff[j,left[loc_left_diff],i]==diff[j,left[loc_left_diff],i].min())[0][0]]]
           temp_left = left[loc_left_diff[np.where(diff[j,left[loc_left_diff],i]==diff[j,left[loc_left_diff],i].min())[0][0]]]
       prev_left = temp_left
       left = temp_left
       right = np.where(diff[j,:,i] >= diff[j,:,i].max()-hyst)[0] 
       right = right[right>left]
       if len(right)==0:
           print('No value for right boundary was found, change hysteresis')
           break
       else:
           right_diff = np.abs(right-mean_right[i])
           loc_right_diff = np.where(right_diff<cell_width)[0]
           if len(loc_right_diff) == 0:  
               right = 0
               temp_right = 0
           else:
               prev_right = right[loc_right_diff[np.where(diff[j,right[loc_right_diff],i]==diff[j,right[loc_right_diff],i].max())[0][0]]]
               temp_right = right[loc_right_diff[np.where(diff[j,right[loc_right_diff],i]==diff[j,right[loc_right_diff],i].max())[0][0]]]
           prev_right = temp_right
           right = temp_right
       loc_cfl[j,0,i] = left
       loc_cfl[j,1,i] = right

#%% Fill in the missing values by using a cubic interpolation of defined step size
if filling == 1:
    step = 3
    loc_cfl_interp = np.zeros(loc_cfl.shape)
    for i in range(loc_cfl.shape[2]):
        loc_l = np.where(loc_cfl[:,0,i])[0]
        loc_r = np.where(loc_cfl[:,1,i])[0]
        for j in range(loc_cfl.shape[0]):
            if loc_cfl[j,0,i] == 0:
                above_l = loc_l[loc_l>j]
                below_l = loc_l[loc_l<j]
                if not ((len(below_l)==0) or (len(above_l)==0)):
                    x_l = np.append(below_l[(len(below_l)-(step)):len(below_l)],above_l[0:step])
                    y_l = loc_cfl[x_l,0,i]
                    f_l = interp1d(x_l,y_l, kind='cubic')
                    loc_cfl_interp[j,0,i] = f_l(j)
                else:
                    loc_cfl_interp[j,0,i] = 0
            else:
                loc_cfl_interp[j,0,i] = loc_cfl[j,0,i]
            if loc_cfl[j,1,i] == 0:
                above_r = loc_r[loc_r>j]
                below_r = loc_r[loc_r<j]
                if not ((len(below_r)==0) or (len(above_r)==0)):
                    x_r = np.append(below_r[(len(below_r)-(step)):len(below_r)],above_r[0:step])
                    y_r = loc_cfl[x_r,1,i]
                    f_r = interp1d(x_r,y_r, kind='cubic')
                    loc_cfl_interp[j,1,i] = f_r(j)
                else:
                    loc_cfl_interp[j,1,i] = 0
            else:
                loc_cfl_interp[j,1,i] = loc_cfl[j,1,i]
else:    
    #% Fill in the missing values using a mean weighted filter 
    step = 3
    loc_cfl_interp = np.zeros(loc_cfl.shape)
    for i in range(loc_cfl.shape[2]):
        loc_l = np.where(loc_cfl[:,0,i])[0]
        loc_r = np.where(loc_cfl[:,1,i])[0]
        for j in range(loc_cfl.shape[0]):
            if loc_cfl[j,0,i] == 0:
                above_l = loc_l[loc_l>j]
                below_l = loc_l[loc_l<j]
                if not ((len(below_l)==0) or (len(above_l)==0)):
                    x_l = np.append(below_l[(len(below_l)-(step)):len(below_l)],above_l[0:step])
                    loc_cfl_interp[j,0,i] = np.mean(x_l)
                else:
                    loc_cfl_interp[j,0,i] = 0
            else:
                loc_cfl_interp[j,0,i] = loc_cfl[j,0,i]
            if loc_cfl[j,1,i] == 0:
                above_r = loc_r[loc_r>j]
                below_r = loc_r[loc_r<j]
                if not ((len(below_r)==0) or (len(above_r)==0)):
                    x_r = np.append(below_r[(len(below_r)-(step)):len(below_r)],above_r[0:step])
                    loc_cfl_interp[j,1,i] = np.mean(x_r)
                else:
                    loc_cfl_interp[j,1,i] = 0
            else:
                loc_cfl_interp[j,1,i] = loc_cfl[j,1,i]
                
    ##%% Fill in the missing values using a mean weighted filter 
    #step = 5
    #loc_cfl_interp = np.zeros(loc_cfl.shape)
    #for i in range(loc_cfl.shape[2]):
    #    loc = np.where(loc_cfl[:,0,i])[0]
    #    for j in range(loc_cfl.shape[0]):
    #        if loc_cfl[j,0,i] == 0:
    #            above = loc[loc>j]
    #            below = loc[loc<j]
    #            if not ((len(below)==0) or (len(above)==0)):
    #                x = np.append(below[(len(below)-(step)):len(below)],above[0:step])
    #                y = np.mean(x)
    #                loc_cfl_interp[j,0,i] = y
    #            else:
    #                loc_cfl_interp[j,0,i] = 0
    #        else:
    #            loc_cfl_interp[j,0,i] = loc_cfl[j,0,i]
#%% Apply a 1D spatial gaussian filter to the array
sigma = 20
for i in range(loc_cfl_interp.shape[2]-1):
     loc_cfl_interp[:,0,i] = scipy.ndimage.filters.gaussian_filter1d(loc_cfl_interp[:,0,i],sigma)
     loc_cfl_interp[:,1,i] = scipy.ndimage.filters.gaussian_filter1d(loc_cfl_interp[:,1,i],sigma)
#%% Overlay two images
loc_cfl = loc_cfl.astype(int)
loc_cfl_interp = loc_cfl_interp.astype(int)
edges = np.zeros(img.shape, dtype='uint8')
wall = np.zeros(img.shape, dtype='uint8')
median_loc = np.zeros(img.shape, dtype='uint8')
for i in range(img.shape[2]):
    for j in range(img.shape[0]):
        edges[j,loc_cfl_interp[j,:,i],i] = 255
        median_loc[j,mean_left[i].astype(int),i] = 255
        #wall[j,loc_wall[j,1,i],i] = 0

#%% Save video with edges overlayed on top
dst = np.zeros(img.shape, dtype='uint8')
for i in range(img.shape[2]):
    img1 = img[:,:,i]
    img2 = edges[:,:,i]
    #img2 = median_loc[:,:,50] 
    dst[:,:,i] = cv2.addWeighted(img1,0.3,img2,0.7,0) 
    
#%% Export the results to a csv file
np.savetxt(filename[:-4]+'.csv',loc_cfl_interp[:,:,0]*cal_factor, delimiter=',')
cv2.imwrite(filename[:-4]+'_edges.jpg',dst[:,:,0])

#%% Print the median value for the diameter
med_diam = (np.median(loc_cfl_interp[:,1,0]) - np.median(loc_cfl_interp[:,0,0]))*cal_factor

print('\n The median diameter found is: ',med_diam)