#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 12:59:41 2017

@author: AlfredoLucas
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir('/Users/AlfredoLucas/Documents/Trabajos/University/Cabrales/Videos/training_videos')
#%%  Import into dataframes

df_1 = pd.read_csv('training_data/training_data_1.csv')
df_0 = pd.read_csv('training_data/training_data_0.csv')

#%
df_1['label'] = 1
df_0['label'] = 0
df_1.columns = np.arange(1025)
df_0.columns = np.arange(1025)
df = pd.concat([df_1, df_0],ignore_index=True)
#%% Shuffle the data before separating the training and test set and then create the test and training set
X = df[df.columns[np.arange(1023)]]
y = df[df.columns[1024]]

shuffle_idx = np.random.permutation(len(y))

X.values[:] = X.values[shuffle_idx]
y.values[:] = y.values[shuffle_idx]
#%% Separate into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

#%% Attempt a random forest classifier in the data
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

#%% Perform a cross validation performance check
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf,X_train, y_train, cv=5)

#%% Create a confusion matrix to evaluate performance
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train, y_train_pred)

