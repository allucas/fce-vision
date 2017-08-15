#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 17:39:53 2017

@author: AlfredoLucas
"""
#%%
import pandas as pd
from matplotlib import pyplot as plt
import scipy.signal as sp
import numpy as np
# Prony Fit of Data

df = pd.read_excel('Prony_Test.xlsx')
time = df['X'].values
data = df['Y'].values

# Preprocesing
ds_factor = 10
data_ds = sp.decimate(data,ds_factor)
data_ds = sp.medfilt(data_ds,3)
time_ds = sp.decimate(time,ds_factor)
plt.plot(time_ds[3:-1], data_ds[3:-1])

# Plot the downsampled, filtered signal and the original
plt.plot(time,data, alpha=0.5)

def prony_fit(n, data):
    N = len(data)
    # Create the autoregressive matrix and find the coefficients to the autoregressive polynomial
    auto_mat = np.zeros((n,n))
    auto_vec = np.zeros((n,1))
    for i in range(n):
        for j in range(n):
            auto_mat[-(i+1),j] = data[i+j]
        auto_vec[i] = data[i+n]
    coeffs = np.matmul(np.linalg.pinv(auto_mat),auto_vec)
    coeffs = np.append([1],-coeffs)
    # Find the roots (z) given the coefficients
    z_roots = np.roots(coeffs)
    
    # Calculate and find the amplitude matrix
    
    y_vec = np.zeros((N,1))
    z_mat = np.zeros((N,n),dtype='complex128')
    for i in range(n):
        for j in range(N):
            z_mat[j,i] = z_roots[i]**(j)
            y_vec[j] = data[j]
    
    A_vec = np.matmul(np.linalg.pinv(z_mat),y_vec)
    
    y_hat = np.matmul(z_mat,A_vec)
    
    return y_hat, A_vec, z_roots
    #return y_hat

#%% Prony Fit
start_pt = 5
N = len(data_ds[start_pt:-1])
rms_val = 10
n_val = 10
for n in range(int(np.floor(N/2)-10)):
    n = n +10
    y_hat, a, b = prony_fit(n = n,data=data_ds[start_pt:-1])
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    
    rms = sqrt(mean_squared_error(data_ds[start_pt:-1], y_hat))
    if rms < rms_val:
        rms_val = rms
        n_val = n

#%%
trim_data = data_ds[start_pt:-1]
fig = plt.figure()
ax = plt.subplot(1,1,1)
prony_bf, A_bf, z_bf = prony_fit(n_val,trim_data)
freq_bf = np.imag(np.log(z_bf))
for i in range(1):
    A_bf[np.argmax(A_bf)] = 0

max_freq_bf = freq_bf[np.argmax(A_bf)]
ax.plot(time_ds[start_pt:-1], trim_data,label='Preprocessed Data')
ax.plot(time_ds[start_pt:-1], prony_bf, label='Prony Fit')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles,labels)
print('The second most dominant frequency is: ', max_freq_bf)
#%% Predict the prony fit at every point
n_val_vec = np.ones(len(trim_data))*100
error_vec = np.ones(len(trim_data))*100000
for n in range(int(np.floor(N/2)-10)):
    n = n+10
    y_hat, a, b = prony_fit(n = n,data=trim_data)
    for i in range(len(y_hat)):
        error_val = np.abs(y_hat[i]-trim_data[i])
        if error_val < error_vec[i]:
            error_vec[i] = error_val
            n_val_vec[i] = n
count = 0
y_hat_vec = np.zeros(len(n_val_vec),dtype='complex128')
A_vec = np.zeros((len(n_val_vec), int(np.floor(N/2))), dtype='complex128')
z_roots = np.zeros((len(n_val_vec), int(np.floor(N/2))), dtype='complex128')
for n in n_val_vec:
    y_hat_val, A_vec_val, z_roots_val = prony_fit(n=int(n), data = trim_data)
    zeros_z = np.zeros(int(np.floor(N/2)-len(z_roots_val)))
    zeros_A = np.zeros(int(np.floor(N/2)-len(A_vec_val)))
    A_vec[count,:] = np.append(A_vec_val,zeros_A)
    z_roots[count,:]  = np.append(z_roots_val, zeros_z)
    y_hat_vec[count] = y_hat_val[count]
    count = count + 1
                           
#%%
fig = plt.figure()
ax = plt.subplot(1,1,1)
ax.plot(time_ds[start_pt:-1], sp.medfilt(data_ds[start_pt:-1]),label='Preprocessed Data')
ax.plot(time_ds[start_pt:-1], y_hat_vec, label='Prony Fit')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles,labels)

#%% Extract the frequency information from the coefficients
freq_choice = 0
lambda_vec = np.log(z_roots)
freq_vec = np.imag(lambda_vec)
freq_vec_max = np.zeros(len(freq_vec))
A_vals = np.abs(np.real(A_vec))
A_max = np.argmax(A_vals,1)

for j in range(freq_choice+1):
    for i in range(len(A_vals)):
        A_vals[i,A_max[i]] = 0
    A_max = np.argmax(A_vals,1)

for i in range(len(freq_vec)):
    freq_vec_max[i] = freq_vec[i,A_max[i]]
damp_vec = np.real(lambda_vec)
mean_freq = np.mean(np.abs(freq_vec_max))
plt.figure()
plt.plot(np.abs(freq_vec_max))
plt.plot([0, len(freq_vec_max)], [mean_freq, mean_freq], '--r')
plt.xlabel('Sample Point')
plt.ylabel('Frequency (rad/s)')
#%%

                           
                           