#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 09:31:53 2017

@author: AlfredoLucas
"""
#%%
import scipy.ndimage

#%%
def proc_img(img):
    adjust = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    cl1 = clahe.apply(img)
    adjust = cl1
    
    # Perform filtering
    blur = cv2.medianBlur(img,3)
    blur = cv2.bilateralFilter(img,10,40,40)
    #plt.imshow(blur)
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

#% Select the contours that children of the box only and with the maximum area
def select_contours(img):
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    temp = 0
    for i in range(len(contours)):
        if hierarchy[0][i][3] == 0:
            temp = i
    rbc_contour = contours[temp]
    sigma = 3
    #rbc_contour[:,0,0] = scipy.ndimage.filters.gaussian_filter(rbc_contour[:,0,0], sigma)
    #rbc_contour[:,0,1] = scipy.ndimage.filters.gaussian_filter(rbc_contour[:,0,1], sigma)
    filt_size = 3
    rbc_contour[:,0,0] = scipy.ndimage.filters.uniform_filter(rbc_contour[:,0,0],size=filt_size)
    rbc_contour[:,0,1] = scipy.ndimage.filters.uniform_filter(rbc_contour[:,0,1], size=filt_size)
    rbc_contour = rbc_contour.astype('int32')
    img_cont = cv2.drawContours(img, [rbc_contour], 0, (0,255,0), 1)
    img_cont = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_cont

#%
img_cont = np.zeros(rbc.shape,dtype='uint8')
for i in range(rbc.shape[2]):
    proc_image = proc_img(img=rbc[:,:,i])
    img_cont[:,:,i] = select_contours(proc_image)
    