#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 11:47:23 2017

@author: AlfredoLucas
"""

#%%
import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy.signal
import scipy.ndimage
import os
#%% Define the functions
def load_video(location, filename, start_frame, n_frames):
    import numpy as np
    import cv2
    img_val = cv2.imread(location+'/'+filename+str(start_frame)+'.tif')
    img = np.zeros((img_val.shape[0],img_val.shape[1],n_frames),dtype='uint8')
    eq = np.zeros((img_val.shape[0],img_val.shape[1],n_frames),dtype='uint8')
    for i in range(start_frame,start_frame+n_frames):
        frame = cv2.imread(location+'/'+filename+str(i)+'.tif')
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY )
        img[:,:,i-start_frame] = frame
        #eq[:,:,i-start_frame] = cv2.equalizeHist(frame)
    return img


#%% Load the background
location = '/Users/AlfredoLucas/Documents/Trabajos/University/Cabrales/Videos/wss_videos/1002'
os.chdir(location)
filename = 'rbc'
seq = load_video(location=location, filename=filename, start_frame=1000, n_frames=500)
avg_bkg = np.median(seq,axis=2)
img = seq[:,:,1]
avg_bkg = avg_bkg.astype('uint8')
#% Modify the background to create a binary mask
filt_bkg = cv2.medianBlur(avg_bkg,9)
filt_bkg = cv2.GaussianBlur(filt_bkg,(5,5),5)
plt.figure()
plt.imshow(filt_bkg)
plt.figure()
count = 0
avg_y = np.zeros((filt_bkg.shape[1],filt_bkg.shape[0]),dtype='uint8')


#%%Find the edges of the vessel by using the peaks of the average intensity
for i in range(seq.shape[1]):
    plt.plot(filt_bkg[:,i])
    avg_y[count,:] = filt_bkg[:,i]
    count = count + 1
avg_y = np.mean(avg_y,axis=0)
#pks = scipy.signal.find_peaks_cwt(avg_y,np.arange(4,20))
avg_pks = scipy.signal.argrelmax(avg_y, order=10)
avg_pks = avg_pks[0]
avg_pks = [avg_pks[0], avg_pks[1]]
plt.figure()
plt.plot(avg_y)
plt.vlines(avg_pks,100,200)

#%
diff = np.zeros(seq.shape)
for i in range(seq.shape[2]):
    diff[:,:,i] = -seq[:,:,i]+avg_bkg

rang = 10
loc1 = np.zeros((avg_bkg.shape[1],rang*2))
loc1_pks = np.zeros((avg_bkg.shape[1]))
loc2 = np.zeros((avg_bkg.shape[1],rang*2))
loc2_pks = np.zeros((avg_bkg.shape[1]))
top_edge = np.zeros(avg_bkg.shape)
bot_edge = np.zeros(avg_bkg.shape)
for i in range(avg_bkg.shape[1]):
    loc1[i,:] = filt_bkg[(avg_pks[0]-rang):(avg_pks[0]+rang),i]
    loc1_pks[i] =  (avg_pks[0]-rang)  + int(np.median(np.where(loc1[i,:]==loc1[i,:].max())[0])) 
    loc2[i,:] = filt_bkg[(avg_pks[1]-rang):(avg_pks[1]+rang),i]
    loc2_pks[i] =  (avg_pks[1]-rang)  + int(np.median(np.where(loc2[i,:]==loc2[i,:].max())[0])) 
diff_thresh = rang/2
for i in range(avg_bkg.shape[1]-1):
    if np.abs(loc1_pks[i]-loc1_pks[i+1]) >= diff_thresh:
       loc1_pks[i+1] = loc1_pks[i]
    if np.abs(loc2_pks[i]-loc2_pks[i+1]) >= diff_thresh:
       loc2_pks[i+1] = loc2_pks[i]
sigma = 13
loc1_pks = scipy.ndimage.filters.gaussian_filter1d(loc1_pks, sigma)
loc2_pks = scipy.ndimage.filters.gaussian_filter1d(loc2_pks, sigma)
for i in range(avg_bkg.shape[1]):
    top_edge[int(loc1_pks[i]),i] = 255 
    bot_edge[int(loc2_pks[i]),i] = 255 
loc_pks = [loc1_pks, loc2_pks]

#% Show the overlayed results
dst_edges = cv2.addWeighted(bot_edge.astype('uint8'),0.5,top_edge.astype('uint8'),0.5,0) 
dst_bkg = cv2.addWeighted(dst_edges.astype('uint8'),0.5,img.astype('uint8'),0.5,0)
plt.figure()
plt.imshow(dst_bkg)

#%%
np.savetxt('vessel_edges.csv', loc_pks, delimiter=',')
np.savetxt('background.csv', avg_bkg, delimiter=',')