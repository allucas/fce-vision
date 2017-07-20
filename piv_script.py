 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 19:30:12 2017

@author: AlfredoLucas
"""

#%% Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from matplotlib import cm
import piv_functions as pivf
import os

#%% Initial Parameters
location = '/Users/AlfredoLucas/Documents/Trabajos/University/Cabrales/PIV/flow_5' # Image stack location
os.chdir(location)
filename = 'flow' # Image names
n_frames = 1000 # Number of frames to load and average over the PIV results
y_length = [100,350] # Y limits of the image
x_length = [0,500] # X limits of the image
start_frame = 1000 # Starting frame number on the filename

# PIV parameters
g_size = 25 # Size of the interrogation grid for PIV
step_size = 2  # Step size of PIV
#%% Load the video
img = pivf.load_video(location=location,filename=filename, n_frames=n_frames, x_length=x_length, y_length=y_length, start_frame=start_frame)

#%% Perform the background subtraction on the video
sub_vec = np.zeros((x_length[1]-x_length[0],y_length[1]-y_length[0],n_frames),dtype='uint8')
for i in range(n_frames-1):
    sub = np.abs(img[:,:,i+1]-img[:,:,i])
    sub_vec[:,:,i] = cv2.medianBlur(sub,11)

#%% Break the image into particles
blackhat_vec = np.zeros(sub_vec.shape)
kernel_bh = np.ones((5,5),np.uint8)
kernel_op = np.ones((3,3),np.uint8)
for i in range(sub_vec.shape[2]): 
    blackhat = cv2.morphologyEx(sub_vec[:,:,i], cv2.MORPH_TOPHAT, kernel_bh)
    blackhat_vec[:,:,i] = cv2.morphologyEx(blackhat, cv2.MORPH_OPEN, kernel_op)
    
#%% Test the size of the interrogation region for the PIV
fig,ax = plt.subplots(1)
ax.imshow(sub_vec[:,:,1],'gray')
#Create the rectangle
rect = matplotlib.patches.Rectangle((0,0),g_size,g_size,linewidth=1,edgecolor='r',
                                    facecolor='none')
ax.add_patch(rect)
plt.show()

#%% Get the split interrogation region
sub_vec = blackhat_vec
reg_vec_1 = pivf.grid_img(sub_vec[:,:,1],g_size)
reg_vec_2 = pivf.grid_img(sub_vec[:,:,2],g_size)

# Find the correlation between corresponding cells in the two frames
#g_num = 9
#corr = signal.correlate2d(reg_vec_1[:,:,g_num], reg_vec_2[:,:,g_num], boundary='symm', mode='same')
#corr = corr/255


# Plot the 3D surface plot of the correlation
#pivf.plot_corr_surf(reg_vec_1[:,:,g_num],reg_vec_2[:,:,g_num])
##% Heatmap
#fig = plt.figure()
#ax = fig.add_subplot(111)
#hm = plt.imshow(corr)
#fig.colorbar(hm, shrink=0.5, aspect=10)

#% Find the maximum peak
# Filter the correlation results
#blur = cv2.GaussianBlur(corr,(5,5),0)
#loc = np.where(corr == corr.max())
#plt.imshow(blur)
#pivf.plot_surf(g_size,blur)

#% Plot all the correlations
#corr_vec = pivf.get_corr_vec(reg_vec_1,reg_vec_2,5)
#x_num = int(sub_vec.shape[0]/g_size)
#y_num = int(sub_vec.shape[1]/g_size)
#fig, axs = plt.subplots(x_num, y_num, figsize=(3, 6), facecolor='w', edgecolor='k')
#axs = axs.ravel()
#
#for i in range(x_num*y_num):
#    axs[i].imshow(corr_vec[:,:,i])
#    axs[i].axis('off')
#    
#fig.subplots_adjust(hspace = 0, wspace=0)

#%% Get the average correlation vector along all the frames
reg_vec_1 = pivf.grid_img(sub_vec[:,:,1],g_size)
avg_corr =np.zeros(reg_vec_1.shape)
blur = 21
count = 0
for i in range(sub_vec.shape[2]-step_size):
    count = count + 1
    g_1 = pivf.grid_img(sub_vec[:,:,i],g_size)
    g_2 = pivf.grid_img(sub_vec[:,:,i+step_size],g_size)
    corr_vec = pivf.get_corr_vec(g_2,g_1,blur)
    for j in range(corr_vec.shape[2]):
        avg_corr[:,:,j] = avg_corr[:,:,j] + corr_vec[:,:,j]

for k in range(avg_corr.shape[2]):
    avg_corr[:,:,k] = avg_corr[:,:,k]/count

#%% Plot the average correlation
pivf.reconstruct_grid(int(sub_vec.shape[0]/g_size),int(sub_vec.shape[1]/g_size),avg_corr)

#%% Find the vectors in each grid space and plot them
x_disp = np.zeros(avg_corr.shape[2])
y_disp = np.zeros(avg_corr.shape[2])
center = np.floor(g_size/2)
for i in range(avg_corr.shape[2]):
    loc = np.where(avg_corr[:,:,i]==avg_corr[:,:,i].max())
    #dx, dy = pivf.gauss_interp(avg_corr[:,:,i],loc[0][0],loc[1][0])
    dx = 0
    dy = 0
    x_disp[i] = loc[0][0] + 1 - center + dx
    y_disp[i] = loc[1][0] + 1 - center + dy

#ax = plt.axes()
#g = 3
#ax.arrow(0.5, 0, y_disp[g]/(2*y_disp.min()), x_disp[g]/x_disp.min(), head_width=0.1,length_includes_head=True, fc='k', ec='k')
#plt.ylim([0,1])
#plt.show()

x_num = int(sub_vec.shape[0]/g_size)
y_num = int(sub_vec.shape[1]/g_size)
fig, axs = plt.subplots(x_num, y_num, figsize=(3, 6), facecolor='w', edgecolor='k')
axs = axs.ravel()
for i in range(x_num*y_num):
    axs[i].arrow(0.5, 0, y_disp[i]/g_size, -x_disp[i]/g_size, head_width=0.075,length_includes_head=True, fc='k', ec='k')
    axs[i].axis('off')
fig.subplots_adjust(hspace = 0, wspace=0)

#%% Plot x displacement as a function of distance from the wall
count = 0
for i in range(x_num-1):
    plt.plot(range(0,y_num),-x_disp[count:count+y_num],'--')
    count = count+y_num
plt.xlabel('Grid Distance From Wall')
plt.ylabel('Displacement (px)') 

#plt.xticks([1,2,3,4,5])
    
    
    