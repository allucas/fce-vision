#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 15:08:17 2017

@author: AlfredoLucas
"""
#%% Restart Kernel
#import os
#os._exit(00)

#%% Imports
import numpy as np
from matplotlib import pyplot as plt
import cv2
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
import piv_functions as pivf
grid = np.zeros([50,50,2],dtype='uint8')
density = 100
disp_x = 15
disp_y = 15
x = np.random.randint(0,30,density,dtype='uint8')
y = np.random.randint(0,30,density,dtype='uint8')
for i in range(len(x)):
    grid[x[i],y[i],0] = 255
    grid[x[i]+disp_x,y[i]+disp_y,1] = 255

plt.imshow(grid[:,:,0])
plt.figure()
plt.imshow(grid[:,:,1])

corr = signal.correlate2d(grid[:,:,1], grid[:,:,0], boundary='symm', mode='same')
corr = corr/255

#% Plot 3D Surface Plot
pivf.plot_corr_surf(grid[:,:,0], grid[:,:,1])

#% Plot heatmap
fig = plt.figure()
ax = fig.add_subplot(111)
hm = plt.imshow(corr)

# Show the center of the image
rect = matplotlib.patches.Circle((25,25),radius=0.7, facecolor='r')
ax.add_patch(rect)
fig.colorbar(hm, shrink=0.5, aspect=10)
plt.show()

#% Find the location of the maximum peak and calculate the displacement vector
loc_max = np.where(corr==corr.max())
center = grid.shape[0]/2
x_disp = (loc_max[0][0]- center) + 1
y_disp = (loc_max[1][0] - center) + 1

#%
print('X displacement: ', x_disp, ', Y Displacement: ', y_disp)
fig = plt.figure()
ax = fig.add_subplot(111)
img = ax.imshow(grid[:,:,1])
ax.arrow(25,0,y_disp,x_disp, width=1, head_width=5)
plt.show()