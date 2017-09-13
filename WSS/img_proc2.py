#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 19:59:35 2017

@author: AlfredoLucas
"""

#%% Attempt to sharpen the image


def sharpen_img(img):
    adjust = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    cl1 = clahe.apply(img)
    adjust = cl1
    
    # Perform filtering
    blur = cv2.medianBlur(img,3)
    blur = cv2.bilateralFilter(img,10,40,40)
    plt.imshow(blur)
    g_mask = cv2.GaussianBlur(blur, (23,23),3)
    sharp = cv2.addWeighted(blur, 1.5, g_mask, -0.5,0)
    ret2,th2 = cv2.threshold(img,135,255,cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(th2, (3,3),1)
    k_size=5
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(k_size,k_size))
    morph = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel=kernel)
    k_size_2 = 11
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k_size_2,k_size_2))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel=kernel_2)
    th3 = cv2.adaptiveThreshold(sharp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,201,13)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    #sharp = cv2.filter2D(blur, -1, kernel)
    edges = cv2.Canny(morph,25,150)
    blur = cv2.GaussianBlur(img, (3,3),1)
    return morph

edges = np.zeros(rbc.shape)
for i in range(rbc.shape[2]):
    edges[:,:,i] = sharpen_img(rbc[:,:,i])
    
#plt.imshow(blur,'gray')
#plt.figure()
#plt.imshow(sharp,'gray')
#plt.figure()
plt.imshow(edges,'gray')
#%%
play_video(edges)
#%%
diff = np.zeros(seq.shape)
for i in range(seq.shape[2]):
    diff[:,:,i] = -seq[:,:,i]+avg_bkg