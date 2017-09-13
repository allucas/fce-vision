#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:16:12 2017

@author: AlfredoLucas
"""
#%%
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
diameter = 7.82

def find_quartic(pts,vals):
    x_mat = np.zeros((5,5))
    
    for i in range(5):
        for j in range(5):
                x_mat[i,j] = pts[i]**j
    x_mat = np.matrix(x_mat)
    vals = np.matrix(vals).T
    inv_mat = np.linalg.inv(x_mat)
    coeffs = np.matmul(inv_mat,vals)
    return coeffs

def z_sf(x, y,d0=diameter, a0=0.0518, a1=2.0026, a2=-4.491):
    xy = (x**2 + y**2)
    z_val_top = np.sqrt(1 - ((4*xy)/(d0**2)))*(a0 + a1*(xy/(d0**2)) + a2*((xy**2)/(d0**4)))
    z_val_bot = -np.sqrt(1 - ((4*xy)/(d0**2)))*(a0 + a1*(xy/(d0**2)) + a2*((xy**2)/(d0**4)))
    return z_val_top, z_val_bot 

def z(x, y,d0=diameter, a0=0.0518, a1=2.0026, a2=-4.491):
    xy = (x**2 + y**2)
    r_xy = np.sqrt(xy)
    r_d = xy/(d0**2)
    r = d0/2
    coeffs = find_quartic([0,r/2,r,r/4, -r/2],[-0.5,0.25,0,-0.4,0.25])
    a3 = float(coeffs[0])
    a4 = float(coeffs[1])
    a5 = float(coeffs[2])
    a6 = float(coeffs[3])
    a7 = float(coeffs[4])
    z_val_bot = (a3 + a4*r_xy + a5*(r_xy)**2 + a6*(r_xy)**3 + a7*(r_xy)**4)
    z_val_top = np.sqrt(1 - ((4*xy)/((d0)**2)))*(-4 + a1*(xy/((d0)**2)) + a2*((xy**2)/(d0**4)))
    #z_val_bot = -np.sqrt(1 - ((4*xy)/(d0**2)))*(2 + a1*(xy/(d0**2)) + a2*((xy**2)/(d0**4)))
    return z_val_top, z_val_bot 

def polar_grid(radius, pts):
    t = np.linspace(0,2*np.pi,pts)
    r = np.linspace(0,radius,pts)
    R,T = np.meshgrid(r,t)
    X,Y = R*np.cos(T), R*np.sin(T)
    return X,Y

# Surface Plot with Rectangular Grid
#X = np.arange(-4, 4, 0.1)
#Y = np.arange(-4, 4, 0.1)
#X, Y = np.meshgrid(X, Y)
#Z_top, Z_bot = z(x=X, y=Y)
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_wireframe(X, Y, Z_top, rstride=10, cstride=10)
#ax.plot_wireframe(X, Y, Z_bot, rstride=10, cstride=10)
#
#
#plt.show()

#z_val_top, z_val_bot = z(X, np.ones((len(X)))*2)
#plt.plot(X,z_val_top)
#plt.plot(X, z_val_bot)

#%% Surface Plot with Polar Grid
X,Y = polar_grid(diameter/2,300)
Z_top, Z_bot = z(x=X, y=Y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z_top, rstride=10, cstride=10)
ax.plot_wireframe(X, Y, Z_bot, rstride=10, cstride=10)
#ax.set_zlim(-0.25, 0.25)
plt.show()

#%%
r = diameter/2
pts = [0,r/2,r,r/4, -r/2]
vals= [-0.5,0.25,0,-0.4, 0.25]
coeffs = find_quartic(pts,vals)
x = np.arange(0,r,0.1)
y =  y = float(coeffs[0]) + float(coeffs[1])*x + float(coeffs[2])*x**2 + float(coeffs[3])*x**3 + float(coeffs[4])*x**4
plt.plot(x,y)
plt.plot(pts,vals,'.')



#def quadratic(x,coeffs):
#    y = float(coeffs[0]) + float(coeffs[0])*x + float(coeffs[1])*x**2 + float(coeffs[2])*x**3 + float(coeffs[3])*x**4
#    return y 
#
