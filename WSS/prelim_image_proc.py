#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 09:24:25 2017

@author: AlfredoLucas
"""
#%%
import cv2
from matplotlib import pyplot as plt
import numpy as np
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

def play_video(frames):
    import cv2
    for i in range(frames.shape[2]-1):
        cv2.imshow('Video',frames[:,:,i])
        k = cv2.waitKey(0)
        if k == 27:
            break
    cv2.destroyAllWindows()
#%% Load the background
location = '/Users/AlfredoLucas/Documents/Trabajos/University/Cabrales/Videos/wss_videos/1000'
filename = 'rbc'
seq = load_video(location=location, filename=filename, start_frame=1000, n_frames=179)
avg_bkg = np.median(seq,axis=2)
img = seq[:,:,1]

#%%
ret , img_thresh = cv2.threshold(avg_bkg.astype('uint8'),120,255,cv2.THRESH_BINARY)
k_size = 7
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k_size,k_size))
filt = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel=kernel)
plt.imshow(filt,'gray')
#%% Make adjustments on the image

# Perform basic filtering
diff = img-(filt*255)

#%%
min_val = diff.min()
for i in range(diff.shape[0]):
    for j in range(diff.shape[1]):
        diff[i,j] = diff[i,j] - min_val
diff = diff.astype('uint8')
adjust = cv2.equalizeHist(diff)
blur = cv2.GaussianBlur(adjust, (7,7),5)
thresh_val = 150
blur = cv2.medianBlur(adjust,11)
ret , img_thresh = cv2.threshold(blur,thresh_val,255,cv2.THRESH_BINARY)
#% Perform Morphological operations
k_size = 5
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k_size,k_size))
filt = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel=kernel)
final = filt
plt.imshow(blur,'gray')
#%%

edges = cv2.Canny(blur,100,180)
plt.imshow(edges,'gray')
