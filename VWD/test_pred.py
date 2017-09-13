#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:50:27 2017

@author: AlfredoLucas
"""
#%%
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.externals import joblib

#%% Load the classifier of interest
clf = joblib.load('/Users/AlfredoLucas/Documents/Trabajos/University/Cabrales/Videos/training_videos/training_data/random_forest_2.pkl')

#%% Segmentize the image into different grids
def create_grid(img,g_size):
    import numpy as np
    g_vec = np.zeros((g_size,g_size,(int(img.shape[0]/g_size))*(int(img.shape[1]/g_size))), dtype='uint8')
    count = 0
    for i in range(int((img.shape[0])/g_size)-1):
        for j in range(int((img.shape[1])/g_size)-1):
            g_vec[:,:,count] = img[i*g_size:(i+1)*g_size,j*g_size:(j+1)*g_size]
            count = count + 1
    return g_vec
#%%
img = cv2.imread('/Users/AlfredoLucas/Documents/Trabajos/University/Cabrales/Videos/training_videos/1003/img10016.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
g_size = 32
grid_vec = create_grid(img=img,g_size=g_size)
features = np.zeros((grid_vec.shape[2], g_size*g_size),dtype='uint8')
for i in range(grid_vec.shape[2]):
    features[i,:] = np.reshape(grid_vec[:,:,i],(g_size**2))

pred = clf.predict(features[:,0:1023])

#%%
l_img = img.copy()
count = 0
for i in range(int(img.shape[0]/g_size)):
    for j in range(int(img.shape[1]/g_size)):
        if pred[count] == 1:
            l_img = cv2.circle(l_img,(int(i*g_size),int(j*g_size)),7, [0,255,0],-1)
        count = count + 1

plt.imshow(l_img,'gray')