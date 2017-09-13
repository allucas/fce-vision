#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 10:53:47 2017

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

def play_video(frames):
    import cv2
    for i in range(frames.shape[2]-1):
        cv2.imshow('Video',frames[:,:,i])
        k = cv2.waitKey(0)
        if k == 27:
            break
    cv2.destroyAllWindows()

#%% Load the background
location = '/Users/AlfredoLucas/Documents/Trabajos/University/Cabrales/Videos/wss_videos/1001'
filename = 'rbc'
os.chdir(location)
seq = load_video(location=location, filename=filename, start_frame=1000, n_frames=200)
img = seq[:,:,0]
avg_bkg = np.loadtxt('background.csv', delimiter=',')
avg_bkg = avg_bkg.astype('uint8')
#%% Load the edges of the image 
locs = np.loadtxt('vessel_edges.csv',delimiter=',')
loc1_pks = locs[0]
loc2_pks = locs[1]
top_edge = np.zeros(avg_bkg.shape)
bot_edge = np.zeros(avg_bkg.shape)
for i in range(avg_bkg.shape[1]):
    top_edge[int(loc1_pks[i]),i] = 255 
    bot_edge[int(loc2_pks[i]),i] = 255 

#% Plot the edges to see if they match
dst_edges = cv2.addWeighted(bot_edge.astype('uint8'),0.5,top_edge.astype('uint8'),0.5,0) 
#dst_bkg = cv2.addWeighted(dst_edges.astype('uint8'),0.5,img.astype('uint8'),0.5,0)
plt.figure()
plt.imshow(dst_edges)

#% Subtract the background from the original frames
diff = np.zeros(seq.shape)
for i in range(seq.shape[2]):
    diff[:,:,i] = -seq[:,:,i]+avg_bkg

#% Define a function to remove the outer regions of an image based on the background edges
def crop_img(img, top, bot):
    cropped = img
    for i in range(img.shape[1]):
        cropped[:int(top[i]),i] = 255
        cropped[int(bot[i]):-1,i] = 255
    return cropped
cropped = np.zeros(seq.shape)
for i in range(seq.shape[2]):
    cropped[:,:,i] = crop_img(diff[:,:,i],loc1_pks,loc2_pks)


#%% Track the RBC throughout the frame
start_fr = 0
end_fr = 150
play_video(seq[:,:,start_fr:end_fr])
img_seq = seq[:,:,start_fr:end_fr]
img_seq_cropped = cropped[:,:,start_fr:end_fr]
loc = np.zeros(2,int)
while loc[0]==0 and loc[1]==0:
    for i in range(2):
        idx = int(i*(img_seq.shape[2]-1))
        image = img_seq[:,:,idx].copy()
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
                temp_loc = refPt
                loc[i] = refPt[0][0]
            	# if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                image = clone.copy()
            	# if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
            		break

#%% Calculate the velocity
fact = 1
vel = ((loc[1]-loc[0])/(end_fr-start_fr))*fact
#%% Crop the image based on the velocity calculated
rect_x = 100
rect_y = 140
rbc = np.zeros((rect_x,rect_y,int(end_fr-start_fr)),dtype='uint8')
if (rect_x>img_seq.shape[0]) or (rect_y>img_seq.shape[1]):
    print('Rectangle size is too big')
else:
    for i in range(end_fr-start_fr):
        rbc[:,:,i] = img_seq_cropped[:,int((loc[0] + vel*i)-(rect_y/2)):int((loc[0] + vel*i)+(rect_y/2)),i]

#%%
def proc_img(img):
    ret2,th2 = cv2.threshold(img,140,255,cv2.THRESH_BINARY)
    k_size=5
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(k_size,k_size))
    morph = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel=kernel)
    k_size_2 = 11
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k_size_2,k_size_2))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel=kernel_2)
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
    #rbc_contour[:,0,0] = scipy.ndimage.filters.uniform_filter(rbc_contour[:,0,0],size=filt_size)
    #rbc_contour[:,0,1] = scipy.ndimage.filters.uniform_filter(rbc_contour[:,0,1], size=filt_size)
    rbc_contour = rbc_contour.astype('int32')
    #rbc_contour = cv2.convexHull(rbc_contour)
    img_cont = cv2.drawContours(img, [rbc_contour], 0, (0,255,0), 1)
    img_cont = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_cont

#%
img_cont = np.zeros(rbc.shape,dtype='uint8')
for i in range(rbc.shape[2]):
    proc_image = proc_img(img=rbc[:,:,i])
    img_cont[:,:,i] = select_contours(proc_image)

#%
dst = np.zeros(rbc.shape, dtype='uint8')
for i in range(rbc.shape[2]):
    dst[:,:,i] = cv2.addWeighted(rbc[:,:,i],0.7,img_cont[:,:,i],0.3,0) 

play_video(dst)


#%%

#%%

def split_image(img):
    M = cv2.moments(img)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    top = img[:,cx:].T
    bot = img[:,:cx].T
    # circ = cv2.circle(cont.copy(),(cx,cy),4,255)
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

def get_coeffs(img,p_deg):
    x, y = get_param(img)
    #x_r, y_r = rotate(x,y,90)
    x_r, y_r = straighten(x,y)
    x_r = x_r-x_r[0]
    y_r = y_r - y_r[-1]
    p = np.polyfit(x_r, y_r, deg=p_deg)
    return x_r, y_r, p

def param_poly(p,x):
    r = x.max()
    y = np.zeros(len(x))
    for j in range(len(y)):
        for i in range(len(p)):
            y[j] = p[len(p)-1-i]*(x[j]**(i)) + y[j]
        #y[j] = (1-np.exp(-(1-(x[j]/r)**2)))*y[j]
    return y
    #def plt_whole_param(p_l,p_r)
    

#% Continuum Functions

def get_F(p, P, X):
    par_xX = 0
    for i in range(len(p)):
        par_xX = par_xX + i*(p[len(p)-1-i]-P[len(p)-1-i])*(X**2)
    F_xy = np.matrix([[1,0],[par_xX,1]])
    F_polar = np.matrix([[1,0,0],[0,1,0],[par_xX,0,1]])
    F =  F_polar
    return F

def get_E(F):
    E = (0.5)*(np.matmul(F.T,F)-np.eye(len(F)))
    return E

def plt_color_map(X, p, val):
    plt.scatter(X, np.polyval(p,X),alpha=0.7, c=val, cmap=plt.get_cmap("jet")), plt.colorbar()
#%% Convert the contour to an actual parametrization
plt.figure()
hl, = plt.plot([], [])
hl2, = plt.plot([], [])
p_deg = 4
n_frames = img_cont.shape[2]
x_r = np.zeros(n_frames)
x_l = np.zeros(n_frames)
y_r = np.zeros(n_frames)
y_l = np.zeros(n_frames)
p_right = np.zeros((n_frames,p_deg+1))
p_left = np.zeros((n_frames, p_deg+1))
for i in range(n_frames):
    cont = img_cont[:,:125,i]
    
    for j in range(cont.shape[0]):
        for k in range(cont.shape[1]):
            if cont[j,k] > 100:
                cont[j,k] = 255
            else:
                cont[j,k] = 0
    
    
    #%
    right_top, left_top, right_bot, left_bot = split_image(cont)
    
    
    #%
    
    if i == 0:
        X_r,Y_r,m = get_coeffs(right_top, p_deg)
        X_l,Y_l,m = get_coeffs(left_top, p_deg) 
    
    x_r, y_r, p_right[i,:] = get_coeffs(right_top, p_deg)
    #print(len(x_r))
    x_l, y_l, p_left[i,:] = get_coeffs(left_top, p_deg)
    
    #%
#    x_r[i], y_r[i], p_right[i] = get_coeffs(right_top)
#    x_l[i], y_l[i], p_left[i] = get_coeffs(left_top)
#    #plt.plot(x_r, y_r)
#    plt.plot(np.polyval(p_right,x_r))
#    plt.figure()
#    plt.plot(x_l,y_l)
#    plt.plot(np.polyval(p_left,x_l))
    
    #% Plot the contours
#    plt.plot(param_poly(p_right[i,:],x_r))
#    plt.plot(param_poly(p_left[i,:],x_l))
#    
    #%
    #plt.show()

#%% Calculate F and E
F_r = np.zeros((3,3,n_frames-1,len(X_r)))
E_r = np.zeros((3,3,n_frames-1,len(X_r)))
mean_E_r = np.zeros((n_frames-1,len(X_r)))
for j in range(n_frames-1):
    for k in range(len(X_r)):
        F_r[:,:,j,k] = get_F(p_right[j+1,:], p_right[0,:],X_r[k])
        E_r[:,:,j,k] = get_E(F_r[:,:,j,k])
        mean_E_r[j,k] = np.mean(np.linalg.eig(E_r[:,:,j,k])[0])

F_l = np.zeros((3,3,n_frames-1,len(X_l)))
E_l = np.zeros((3,3,n_frames-1,len(X_l)))
mean_E_l = np.zeros((n_frames-1,len(X_l)))
for j in range(n_frames-1):
    for k in range(len(X_l)):
        F_l[:,:,j,k] = get_F(p_left[j+1,:], p_left[0,:],X_l[k])
        E_l[:,:,j,k] = get_E(F_l[:,:,j,k])
        mean_E_l[j,k] = np.mean(np.linalg.eig(E_l[:,:,j,k])[0])
        
#%%

for loc in range(1,n_frames):
    plt.figure()
    plt_color_map(X_r, p_right[loc], mean_E_r[loc,:]), plt_color_map(X_l, p_left[loc], mean_E_l[loc,:])
    plt.show()