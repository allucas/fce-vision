#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 16:19:55 2017

@author: viv5149
"""

from tkinter import filedialog
from tkinter import * 
import pandas as pd 
import numpy as np 
import scipy as sp
from scipy import signal
from scipy.integrate import simps 
import matplotlib.pyplot as plt
import math
import cmath 

#%%
def filechooser(): 
    root = Tk() 
    root.withdraw() 
    root.filename = filedialog.askopenfilename(initialdir = "/",title="Select File",filetypes = (("excel files","*.xlsx"),("all files","*.*")))
    root.destroy()
    return root.filename

def down_sample(indata,num): 
    x_in = indata[0]
    y_in = indata[1]
    x_output = [x_in[ii] for ii in range(len(x_in)) if ii%num==0]
    y_output = [y_in[ii] for ii in range(len(y_in)) if ii%num==0]    
    output = [x_output,y_output]
    return output 

def smooth(data,neighbor,poly): 
    y_in = data[1]
    
    if neighbor%2 == 0:
        neighbor = neighbor + 1
    
    y_out = sp.signal.savgol_filter(y_in,neighbor,poly)
    output = [data[0],y_out]
    return output 

def fourier_coefficient(data,n):
    x = data[0]
    y = data[1]
    N = x[len(x)-1]
    fourier_data =  np.zeros(shape=((n+1),3))
    a0 = (1/N)*simps(y,x)
    fourier_data[0] = [a0,0,0]
    
    for ii in range(1,(n+1)):
        cos = [math.cos(ii*math.pi*jj/N) for jj in x]
        integrand1 = [xx*yy for xx,yy in zip(y,cos)]
        aii = sp.integrate.simps(integrand1,x)
        aii = aii*2/N 
        sin = [math.cos(ii*math.pi*jj/N) for jj in x]
        integrand2 = [xx*yy for xx,yy in zip(y,sin)]
        bii = sp.integrate.simps(integrand2,x)
        bii = bii*2/N
        omega = ii*math.pi/N
        fourier_data[ii] = [aii,bii,omega]
    
    return fourier_data

def fourier_series(data,coefficients): 
    x = data[0]
    a0 = coefficients[0][0]
    terms = len(coefficients)
    outy = []
    for ii in x: 
        dummy = 0 
        for jj in range(1,terms):
            ajj = coefficients[jj][0]
            bjj = coefficients[jj][1]
            omega = coefficients[jj][2]
            num = ajj*math.cos(omega*ii)+bjj*math.sin(omega*ii)
            dummy += num 
        
        dummy += a0 
        outy.append(dummy)
    
    output = [x,outy]
    return output 

# filename = filechooser()
# df = pd.read_excel(filename)
df = pd.read_excel('Prony_Test.xlsx')
x_name = df.columns[0]
y_name = df.columns[1]
x_data = df[x_name]
y_data = df[y_name]
data = [x_data,y_data]
data1 = down_sample(data,5)
data2 = smooth(data1,30,2)
fit = math.floor(len(data2[0])/2)
coefficients = fourier_coefficient(data2,fit)
fourier_dat = fourier_series(data2,coefficients)

#%%
plt.figure(1)
plt.plot(data[0],data[1],'bo')
plt.xlabel('Frames')
plt.ylabel('Velocity (mm/s)')
plt.title('Original Input')

plt.figure(2)
plt.plot(data1[0],data1[1],'bo',data2[0],data2[1],'r--')
plt.title('Smoothened Curve')

plt.figure(3)
plt.plot(data2[0],data2[1],'bo',fourier_dat[0],fourier_dat[1],'r--')
plt.title('Fourier Fit')

#%% Prony Analysis  

def prony_coefficients(data,L):
    
    x = data[0]    
    scale = max(x)
    x = np.divide(x,scale)
    deltax = x[1]-x[0]
    
    if (x[0] !=0): 
        x_shift = x[0]
        x = np.subtract(x,x_shift)
    
    y = data[1]
    y_shift = np.mean(y)
    y = np.subtract(y,y_shift)
    n = len(y)
    yinit = y[0]
    y[0] = y[0]-yinit
    

    
    lpm = np.empty([n-L,L],dtype=complex)
    
    for ii in range(len(lpm)): 
        lpm[ii] = [y[jj] for jj in range(ii+L-1,ii-1,-1)]
    
    F = [y[kk] for kk in range(L,n)]
    
    lpm_inv = np.linalg.pinv(lpm)
    P = np.matmul(lpm_inv,F)
    P = np.multiply(P,-1)
    P = np.insert(P,0,1)
    
    mu = np.roots(P)
    
    pron = np.empty([n,L],dtype=complex)
    
    for ll in range(n):
        #factor = n-ll 
        #factor = factor/n
        factor = deltax
        dummy = mu**(x[ll]/factor)
        pron[ll] = dummy
    
    pron_inv = np.linalg.pinv(pron)

    B = np.matmul(pron_inv,y)
    
    parameters = np.empty([L,4])
    
    for aa in range(L): 
        dummy_mu = mu[aa]
        muRe = dummy_mu.real
        muIm = dummy_mu.imag
        dummy_B = B[aa]
        BRe = dummy_B.real
        BIm = dummy_B.imag 
        
        sigma = (muRe**2)+(muIm**2)
        sigma = sigma**0.5
        sigma = np.log(sigma)
        parameters[aa][0] = sigma
        
        
        if (muRe == 0): 
            omega = math.radians(90)
        else: 
            omega = muIm/muRe
            omega = math.atan(omega)
        
        parameters[aa][1] = omega
        
        amp = (BRe**2)+(BIm**2)
        amp= amp**0.5
        amp = amp*2
        parameters[aa][2] = amp
        
        if (BRe == 0):
            phase = math.radians(90)
        else: 
            phase = BIm/BRe
            phase = math.atan(phase)
        parameters[aa][3]= phase
    
    return parameters


def prony_plot(data,parameters): 
    x = data[0]
    scale = max(x)
    x = np.divide(x,scale)
    deltax = x[1]-x[0]
    y = data[1]
    y_shift = np.mean(y)
    y = np.subtract(y,y_shift)
    yinit = y[0]
    print(yinit)
    y[0] = y[0] - yinit
    
    if (x[0] !=0): 
        x_shift = x[0]
        x = np.subtract(x,x_shift)
    n = len(x)
    
    prony_out = np.empty(n)
    
    for ii in range(n): 
        summ = 0 
        
        for jj in range(len(parameters)): 
            #factor = n-jj 
            #factor = factor/n
            factor = deltax
            sigma = parameters[jj][0]
            omega = parameters[jj][1]
            A = parameters[jj][2]
            phi = parameters[jj][3]
            t = sigma*x[ii]/factor
            val = math.exp(t)
            val = val*A
            sinu = omega*x[ii]/factor
            sinu = sinu+phi
            sinu = math.cos(sinu)
            newval = val*sinu
            
            summ += newval
        
        prony_out[ii] = summ 
    
    prony_out[0] = prony_out[0] + yinit
    prony_out = np.add(prony_out,y_shift)
    return prony_out

coeff = prony_coefficients(data2,200)
prony_y = prony_plot(data2,coeff)

xplot = data2[0]
plt.figure(4)
plt.plot(data2[0],data2[1],'bo',xplot,prony_y,'r--')
plt.title('Prony Fit')

























