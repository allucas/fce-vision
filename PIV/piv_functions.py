#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 11:39:27 2017

@author: AlfredoLucas
"""

#%% Functions for the PIV code

def grid_img(img,reg_size):
    import numpy as np
    if (img.shape[0]%reg_size == 0) & (img.shape[1]%reg_size == 0):
        num_x = int(img.shape[0]/reg_size)
        num_y = int(img.shape[1]/reg_size)
        num_reg = num_x*num_y
        reg_vec = np.zeros((reg_size, reg_size,num_reg),dtype='uint8')
        count = 0
        for i in range(num_x):
            for j in range(num_y):
                reg_vec[:,:,count] = img[i*reg_size:(i+1)*reg_size,j*reg_size:(j+1)*reg_size]
                count = count+1
        return reg_vec
    else:
        print('Image dimensions must be divisible by region size')
        return 0

def get_disp_vec(f1,f2):
    from scipy import signal
    corr = signal.correlate2d(f1, f2, boundary='symm', mode='same')
    loc_max = np.where(corr==corr.max())
    center = g_size/2
    x_disp = (loc_max[0][0]- center) + 1
    y_disp = (loc_max[1][0] - center) + 1
    return x_disp, y_disp

def plot_corr_surf(f1,f2):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib import pyplot as plt
    from scipy import signal
    import numpy as np
    corr = signal.correlate2d(f1, f2, boundary='symm', mode='same')
    g_size = f1.shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.zeros((g_size,1))
    X[:,0] = np.linspace(0,g_size-1, g_size)
    Y=X
    X, Y = np.meshgrid(X, Y)
    Z = corr
    surf = ax.plot_surface(X=X,Y=Y,Z=Z,cmap=cm.coolwarm)
    fig.colorbar(surf, shrink=0.5, aspect=10)

def plot_surf(g_size,z):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib import pyplot as plt
    from scipy import signal
    import numpy as np
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.zeros((g_size,1))
    X[:,0] = np.linspace(0,g_size-1, g_size)
    Y=X
    X, Y = np.meshgrid(X, Y)
    Z = z
    surf = ax.plot_surface(X=X,Y=Y,Z=Z,cmap=cm.coolwarm)
    fig.colorbar(surf, shrink=0.5, aspect=10)

def get_corr_vec(g1,g2, r_blur):
    # Gaussian blurring is included in the function
    from scipy import signal
    import numpy as np
    import cv2
    corr_vec = np.zeros(g1.shape)
    for i in range(g1.shape[2]):
        corr = signal.correlate2d(g1[:,:,i],g2[:,:,i],boundary='symm', mode='same')
        corr_vec[:,:,i] = cv2.GaussianBlur(corr,(r_blur,r_blur),0)
    return corr_vec

def reconstruct_grid(x_num, y_num, grid):
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(x_num, y_num, figsize=(3, 6), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    for i in range(x_num*y_num):
        axs[i].imshow(grid[:,:,i])
        axs[i].axis('off')
    fig.subplots_adjust(hspace = 0, wspace=0)

def save_video(img_vec,filename):
    import cv2
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case
    out = cv2.VideoWriter(filename,fourcc, 30.0, (1024,1024))
    for i in range(img_vec.shape[2]):
        frame = img_vec[:,:,i]
        frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
        out.write(frame)
    out.release()

def load_video(location, filename, start_frame, n_frames, x_length, y_length):
    import numpy as np
    import cv2
    img = np.zeros((x_length[1]-x_length[0],y_length[1]-y_length[0],n_frames),dtype='uint8')
    eq = np.zeros((x_length[1]-x_length[0],y_length[1]-y_length[0],n_frames),dtype='uint8')
    for i in range(start_frame,start_frame+n_frames):
        frame = cv2.imread(location+'/'+filename+str(i)+'.jpg')
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY )
        img[:,:,i-start_frame] = frame[x_length[0]:x_length[1],y_length[0]:y_length[1]]
        eq[:,:,i-start_frame] = cv2.equalizeHist(frame[x_length[0]:x_length[1],y_length[0]:y_length[1]])
    return img

def get_acorr_vec(g1,r_blur):
    from scipy import signal
    import numpy as np
    import cv2
    acorr_vec = np.zeros(g1.shape)
    for i in range(g1.shape[2]):
        corr = signal.correlate2d(g1[:,:,i],g1[:,:,i],boundary='symm', mode='same')
        acorr_vec[:,:,i] = cv2.GaussianBlur(corr,(r_blur,r_blur),0)
    return acorr_vec

def play_video(frames):
    import cv2
    for i in range(frames.shape[2]-1):
        cv2.imshow('Video',frames[:,:,i])
        k = cv2.waitKey(0)
        if k == 27:
            break
    cv2.destroyAllWindows()

def gauss_interp(corr,x_peak, y_peak):
    import numpy as np
    dx = (np.log(corr[x_peak+1,y_peak]) - np.log(corr[x_peak-1,y_peak]))/(2*(np.log(corr[x_peak+1,y_peak]) + np.log(corr[x_peak-1,y_peak]) - np.log(corr[x_peak,y_peak])))
    dy = (np.log(corr[x_peak,y_peak+1]) - np.log(corr[x_peak,y_peak-1]))/(2*(np.log(corr[x_peak,y_peak+1]) + np.log(corr[x_peak,y_peak-1]) - np.log(corr[x_peak,y_peak])))
    return dx, dy

def correct_corr(corr,g_size):
    import numpy as np
    correct_corr = np.zeros(corr.shape)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            correct_corr[i,j] = corr[i,j]/((1-(i/g_size))*(1-(j/g_size))) 
    return correct_corr

def play_overlay(img1,img2):
    import cv2
    for i in range(img1.shape[2]-1):
        cv2.destroyAllWindows()
        dst = cv2.addWeighted(img1[:,:,i],0.7,img2[:,:,i],0.3,0)
        cv2.imshow('Video',dst)
        k = cv2.waitKey(0)
        if k == 27:
            break
    
def get_init_cfl(img, step):
    import numpy as np
    import cv2
    # Find the difference between adjacent intesity points
    diff = np.zeros((img.shape[0],int(img.shape[1]-(step+1)),img.shape[2]))
    for k in range(img.shape[2]):
        frame = cv2.GaussianBlur(img[:,:,k],(11,11),5)
        diff[:,:,k] = frame[:,step:img.shape[1]-1].astype(float) - frame[:,0:img.shape[1]-(1+step)].astype(float)
    loc_cfl = np.zeros((diff.shape[0],2,diff.shape[2]))
    for i in range(diff.shape[2]):
       for j in range(diff.shape[0]):
           left = np.where(diff[j,:,i] == diff[j,:,i].min())
           left = left[0][0] # Define the left boundary of the cell free layer
           right = np.where(diff[j,:,i] == diff[j,:,i].max())
           right = right[0] 
           right = right[right>left]
           if len(right)>0:
               right = right[0] # Define the right boundary of the cell free layer
           elif len(right)==0:
               left= 0
               right = 0
           loc_cfl[j,0,i] = left
           loc_cfl[j,1,i] = right
    return loc_cfl

