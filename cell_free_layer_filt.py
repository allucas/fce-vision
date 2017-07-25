#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:09:57 2017

@author: AlfredoLucas
"""
#%% Import
from scipy.interpolate import interp1d
from scipy import signal
import scipy
import cv2
import numpy as np
from matplotlib import pyplot as plt
import piv_functions as pivf
import os

#%% Initial Parameters
location = '/Users/AlfredoLucas/Documents/Trabajos/University/Cabrales/PIV/flow_3' # Image stack location
os.chdir(location)
filename = 'flow' # Image names
n_frames = 100 # Number of frames to load and average over the PIV results
y_length = [0,500] # Y limits of the image
x_length = [0,500] # X limits of the image
start_frame = 1000 # Starting frame number on the filename

#%% Load the video
img = pivf.load_video(location=location,filename=filename, n_frames=n_frames, x_length=x_length, y_length=y_length, start_frame=start_frame)
#img = cv2.medianBlur(img,5)
#%%
loc_cfl =  pivf.get_init_cfl(img,3)
hyst = 15
step = 4
cell_width = 30
mean_left = np.median(loc_cfl[:,0,:],axis=0)
mean_right = np.median(loc_cfl[:,1,:],axis=0)
diff = np.zeros((img.shape[0],int(img.shape[1]-(step+1)),img.shape[2]))
for k in range(img.shape[2]):
    frame = cv2.GaussianBlur(img[:,:,k],(11,11),11)
    diff[:,:,k] = frame[:,step:img.shape[1]-1].astype(float) - frame[:,0:img.shape[1]-(1+step)].astype(float)
loc_cfl = np.zeros((diff.shape[0],2,diff.shape[2]))
for i in range(diff.shape[2]):
   prev_left = mean_left[i]
   prev_right = mean_right[i]
   for j in range(diff.shape[0]):
       left = np.where(diff[j,:,i] <= diff[j,:,i].min()+hyst)[0]
       diff_val_left = diff[j,left,i]
       disp_left = cell_width+1  
       while disp_left>cell_width:
           if len(diff_val_left) == 0:
               left = 0
               disp_left = 0
               temp_left = 0
           else:
               temp_left = np.where(diff[j,:,i] == diff_val_left.min())[0]
               temp_left = temp_left[np.where((temp_left-((mean_left[i]+prev_left)/2))==(temp_left-((mean_left[i]+prev_left)/2)).min())[0][0]]
               disp_left = np.abs(temp_left-mean_left[i])
               diff_val_left = np.delete(diff_val_left,np.where(diff_val_left == diff_val_left.min())[0][0])
       prev_left = temp_left
       left = temp_left
       right = np.where(diff[j,:,i] >= diff[j,:,i].max()-hyst)[0] 
       right = right[right>left]
       if len(right)==0:
           print('No value for right boundary was found, change hysteresis')
           break
       else:
           diff_val_right = diff[j,right,i]
           disp_right = cell_width+1
           while disp_right>cell_width: 
               if len(diff_val_right) == 0:
                   right = 0
                   disp_right = 0
                   temp_right = 0
               else:
                   temp_right = np.where(diff[j,:,i] == diff_val_right.min())[0]
                   temp_right = temp_right[np.where((temp_right-((mean_right[i]+prev_right)/2))==(temp_right-((mean_right[i]+prev_right)/2)).min())[0][0]]
                   disp_right = np.abs(temp_right-mean_right[i])
                   diff_val_right = np.delete(diff_val_right,np.where(diff_val_right == diff_val_right.min())[0][0])
           prev_right = temp_right
           right = temp_right
       loc_cfl[j,0,i] = left
       loc_cfl[j,1,i] = right

#%% Fill in the missing values by using a cubic interpolation of defined step size
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

#%% Fill in the missing values using a mean weighted filter 
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
sigma = 10
for i in range(loc_cfl_interp.shape[2]-1):
     loc_cfl_interp[:,0,i] = scipy.ndimage.filters.gaussian_filter1d(loc_cfl_interp[:,0,i],sigma)
     loc_cfl_interp[:,1,i] = scipy.ndimage.filters.gaussian_filter1d(loc_cfl_interp[:,1,i],sigma)
#%% Overlay two images
loc_cfl = loc_cfl.astype(int)
loc_cfl_interp = loc_cfl_interp.astype(int)
x_range = [50,450]
edges = np.zeros(img.shape, dtype='uint8')
wall = np.zeros(img.shape, dtype='uint8')
median_loc = np.zeros(img.shape, dtype='uint8')
for i in range(img.shape[2]):
    for j in range(x_range[0],x_range[1]):
        edges[j,loc_cfl_interp[j,:,i],i] = 255
        median_loc[j,mean_left[i].astype(int),i] = 255
        #wall[j,loc_wall[j,1,i],i] = 0

img1 = img[:,:,0]
img2 = edges[:,:,0]
#img2 = median_loc[:,:,50]
dst = cv2.addWeighted(img1,0.7,img2,0.3,0)
plt.imshow(dst,'gray')
