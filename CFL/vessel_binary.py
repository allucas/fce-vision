#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 09:36:27 2017

@author: AlfredoLucas
"""
#%%
import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
import scipy

thresh_val = 170
#img = cv2.imread('/Users/AlfredoLucas/Documents/Trabajos/University/Cabrales/fce-vision/CFL/Joyce/LR-13 Peptide_MMStack_Pos0.ome.tif')
#img = cv2.imread('/Users/AlfredoLucas/Documents/Trabajos/University/Cabrales/fce-vision/CFL/Joyce/3_10^-6 Phen after Pep Base_MMStack_Pos0.ome.tif')
#img = cv2.imread('/Users/AlfredoLucas/Documents/Trabajos/University/Cabrales/fce-vision/CFL/Joyce/AfterPepBaseline_MMStack_Pos0.ome.tif')
#img = cv2.imread('/Users/AlfredoLucas/Documents/Trabajos/University/Cabrales/fce-vision/CFL/Joyce/Baseline1_MMStack_Pos0.ome.tif')
#%% Inputs
cal_factor = float(input('Enter calibration factor (in px/mm): '))
filename = input('filename: ') # Image name
img = cv2.imread(filename)
#%%
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
img_gray = cv2.GaussianBlur(img_gray, (21,21),10)
plt.imshow(img_gray,'gray')
#img_thresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#            cv2.THRESH_BINARY,3,2)
ret , img_thresh = cv2.threshold(img_gray,thresh_val,255,cv2.THRESH_BINARY)

plt.figure()
plt.imshow(img_thresh,'binary')

#% Perform some morphological operations
kernel_1 = np.ones((21,21),'uint8')
kernel_2 = np.ones((21,21),'uint8')
kernel_3 = np.ones((19,19),'uint8')
closed = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel_1)
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_1)
gradient = cv2.morphologyEx(opened, cv2.MORPH_GRADIENT, kernel_2)
plt.figure()
plt.imshow(gradient,'binary')

#%% Define the center location of the image
cent_loc = 1000
image = img_gray
def click_rect(event, x, y, flags, param):
    	# grab references to the global variables
        global refPt
    	# if the left mouse button was clicked, record the starting
    	# (x, y) coordinates and indicate that cropping is being
    	# performed
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            clone1 = cv2.circle(image, (refPt[0][0],refPt[0][1]),10, (0, 0, 0), -1).copy()
            cv2.imshow("image", clone1)


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
        if not (((refPt[0][0]) < 0) or ((refPt[0][0])>image.shape[1]) or ((refPt[0][1])<0) or ((refPt[0][1])>image.shape[0])):
            temp_loc = refPt
            cent_loc = refPt[0][0]
    
    	# if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()

    	# if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
    		break


#%%
truncate = 100
edge_loc = np.zeros((gradient.shape[0]-truncate,2))

for i in range(gradient.shape[0]-truncate):
    idx = np.where(gradient[i,:]>=200)
    idx = idx[0]
    diff = cent_loc-idx
    left_locs = idx[np.where(diff == diff[diff>0].min())]
    right_locs = idx[np.where(diff == diff[diff<0].max())]
    right_locs = np.abs(right_locs)
    edge_loc[i,0] = left_locs.min()
    edge_loc[i,1] = right_locs.min()

sigma = 20
edge_loc[:,0] = scipy.ndimage.filters.gaussian_filter1d(edge_loc[:,0],sigma)
edge_loc[:,1] = scipy.ndimage.filters.gaussian_filter1d(edge_loc[:,1],sigma)

edges = np.zeros(gradient.shape, dtype='uint8')
for i in range(len(edge_loc)):
    edges[i,int(edge_loc[i,0])] = 255
    edges[i,int(edge_loc[i,1])] = 255

#% Overlay the two images

dst = np.zeros(img.shape, dtype='uint8')
img1 = img_gray
img2 = edges
#img2 = median_loc[:,:,50] 
dst = cv2.addWeighted(img1,0.3,img2,0.7,0) 

#%% Export the results to a csv file
np.savetxt(filename[:-4]+'.csv',edge_loc[:,:]/cal_factor, delimiter=',')
cv2.imwrite(filename[:-4]+'_edges.jpg',dst[:,:])

#%% Print the median value for the diameter
med_diam = (np.median(edge_loc[:,1]) - np.median(edge_loc[:,0]))/cal_factor

print('\n The median diameter found is: ',med_diam)