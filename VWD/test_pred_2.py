#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 13:42:55 2017

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
img = cv2.imread('/Users/AlfredoLucas/Documents/Trabajos/University/Cabrales/Videos/test_videos/1000/img10016.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
g_size = 32
grid_vec = create_grid(img=img,g_size=g_size)
l_img = img.copy()
for i in range(int(img.shape[0]/g_size)-1):
    feature = np.zeros(int(img.shape[1]/g_size))
    feature_vec = np.zeros((int(img.shape[1]/g_size), int((img.shape[1]/g_size)*(img.shape[0]/g_size))))
    for j in range(int(img.shape[1]/g_size)-1):
        grid = img[i*g_size:(i+1)*g_size,(j*g_size):((j+1)*g_size)]
        feature = np.reshape(grid,g_size**2)
        if clf.predict(feature[:-1].reshape(1,-1))[0]==1:
            l_img = cv2.circle(l_img,(int(j*g_size + g_size/2),int(i*g_size + g_size/2)),7, [0,255,0],-1)

plt.imshow(l_img)
