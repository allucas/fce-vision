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

def load_video(location, filename, start_frame, n_frames, x_length, y_length):
    import numpy as np
    import cv2
    img = np.zeros((x_length[1]-x_length[0],y_length[1]-y_length[0],n_frames),dtype='uint8')
    eq = np.zeros((x_length[1]-x_length[0],y_length[1]-y_length[0],n_frames),dtype='uint8')
    for i in range(start_frame,start_frame+n_frames):
        frame = cv2.imread(location+'/'+filename+str(i)+'.tif')
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
            
        