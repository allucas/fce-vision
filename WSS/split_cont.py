#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 12:34:24 2017

@author: AlfredoLucas
"""
#%%

def split_image(img):
    M = cv2.moments(img)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    top = img[:,cx:].T
    bot = img[:,:cx].T
    
    M_top = cv2.moments(top)
    cx_top = int(M_top['m10']/M_top['m00'])
    cy_top = int(M_top['m01']/M_top['m00'])
    
    M_bot = cv2.moments(bot)
    cx_bot = int(M_bot['m10']/M_bot['m00'])
    cy_bot = int(M_bot['m01']/M_bot['m00'])
    
    right_top = top[:,cx_top:]
    left_top = top[:,:cx_top]
    
    right_bot = bot[:,cx_top:]
    left_bot = bot[:,:cx_top]
    
    return right_top, left_top, right_bot, left_bot
 
def get_param(img):
    temp_x = np.zeros((img.shape[0]))
    temp_y = np.zeros((img.shape[0]))    
    for i in range(img.shape[0]):
        temp_x[i] = i
        if (img[i,:]==255).any():
            temp_y[i] = np.where(img[i,:]==255)[0][0]
        else:
            temp_y[i] = np.inf
    for i in range(img.shape[0]):
        if (temp_y[i]==np.inf):
            x = temp_x[:i]
            y = temp_y[:i]
            break
        else:
            x = temp_x
            y = temp_y
    return x, y

def rotate(x,y, theta):
    x_r = np.zeros(len(x))
    y_r = np.zeros(len(y))
    for i in range(len(x)):
        x_r[i] = x[i]*np.cos(theta*np.pi/180) - y[i]*np.sin(theta*np.pi/180)
        y_r[i] = x[i]*np.sin(theta*np.pi/180) + y[i]*np.cos(theta*np.pi/180)
    return x_r, y_r

def straighten(x,y):
    temp = x[0]
    count = 0
    for i in range(len(x)):
        if x[i]>temp:
            temp = x[i]
            count = 0
#        elif x[i]==temp:
#            count = count+0.01
#            x[i] = temp+count
        else:
            x[i] = temp
    return x,y

def get_coeffs(img):
    x, y = get_param(img)
    #x_r, y_r = rotate(x,y,90)
    x_r, y_r = straighten(x,y)
    x_r = x_r-x_r[0]
    y_r = y_r - y_r[-1]
    p = np.polyfit(x_r, y_r, deg=4)
    return x_r, y_r, p

#def plt_whole_param(p_l,p_r)

#%% Convert the contour to an actual parametrization
hl, = plt.plot([], [])
hl2, = plt.plot([], [])

cont = img_cont[:,:125,i]

for i in range(cont.shape[0]):
    for j in range(cont.shape[1]):
        if cont[i,j] > 100:
            cont[i,j] = 255
        else:
            cont[i,j] = 0


#%
right_top, left_top, right_bot, left_bot = split_image(cont)



#%
x_r, y_r, p_right = get_coeffs(right_top)
x_l, y_l, p_left = get_coeffs(left_top)

#plt.plot(x_r, y_r)
#    plt.plot(np.polyval(p_right,x_r))
#    plt.figure()
#    plt.plot(x_l,y_l)
#    plt.plot(np.polyval(p_left,x_l))

#%
#    plt.plot(np.polyval(p_right,x_r))
#    plt.plot(np.polyval(p_left,x_l))

#%
