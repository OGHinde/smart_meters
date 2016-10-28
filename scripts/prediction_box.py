# -*- coding: utf-8 -*-
"""Prediction module."""

# Author: Óscar G. Hinde

print '\nInit.'
import resources
import sys
import pandas as pd
pd.options.mode.chained_assignment = None  # Eliminate pandas' warnings. Default='warn'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcess
from sklearn.svm import SVR
print "Importing cPickle..."
# Use cPickle if possible. It's much faster!
try:
    import cPickle as pickle
    print "Success!"
except:
    import pickle
    print "Failed. Defaulted to pickle."


def toeplitz_SVR_validate(X, cluster_index_list, memory=96):
    
    print 'Calculating validation results...'
    K = len(cluster_index_list)
    val_scores = []
    tr_scores = []
    svr_val_hat = []
    
    for cluster_index in cluster_index_list:
        time_series = np.mean(X[cluster_index,:], axis=0)

        # Restrict time series length, 
        # define training and validation set sizes
        l = time_series.shape[0]-time_series.shape[0]%(memory+1) 
        time_series = time_series[:l]
        n_tr = int((l-memory)*0.4)
        n_val = int((l-memory)*0.3)
    
        # Generate and normalize (row-wise) Toeplitz 
        # matrix with the specified memory
        toeplitz = np.zeros((l-memory, memory+1,))     
        for i in range(l-memory):
            toeplitz[i, :] = time_series[i:i+memory+1]        
        mx = np.mean(toeplitz, axis=1)
        stdx = np.std(toeplitz, axis=1, dtype=np.float64)
        toeplitz = np.divide((toeplitz-np.array([mx,]*(memory+1)).transpose()), 
                              np.array([stdx,]*(memory+1)).transpose())

        # Create training and validation sets
        X_tr = toeplitz[:n_tr, :memory]
        y_tr = toeplitz[:n_tr, memory]
        X_val = toeplitz[n_tr:n_tr+n_val,:][:,:memory]
        y_val = toeplitz[n_tr:n_tr+n_val, memory]

        # Train SVR model
        svr = SVR()
        svr.fit(X_tr, y_tr)
        tr_scores.append(svr.score(X_tr, y_tr))
        val_scores.append(svr.score(X_val, y_val))
        svr_val_hat.append(svr.predict(X_val))

    return tr_scores, val_scores, svr_val_hat


def toeplitz_SVR_test(X, cluster_index_list, memory=96):

    K = len(cluster_index_list)
    tr_scores = []
    tst_scores = []
    svr_tst_hat = []

    for cluster_index in cluster_index_list:
        time_series = np.mean(X[cluster_index,:], axis=0)
    
        # Restrict time series length, 
        # define training and test set sizes
        l = time_series.shape[0]-time_series.shape[0]%(memory+1) 
        time_series = time_series[:l]
        n_tr = int((l-memory)*0.4)+int((l-memory)*0.3)
        n_tst = l-memory-n_tr
    
        # Generate and normalize (row-wise) Toeplitz 
        # matrix with the specified memory
        toeplitz = np.zeros((l-memory, memory+1,))     
        for i in range(l-memory):
            toeplitz[i, :] = time_series[i:i+memory+1]        
        mx = np.mean(toeplitz, axis=1)
        stdx = np.std(toeplitz, axis=1, dtype=np.float64)
        toeplitz = np.divide((toeplitz-np.array([mx,]*(memory+1)).transpose()), 
                              np.array([stdx,]*(memory+1)).transpose())

        # Create training and validation sets
        X_tr = toeplitz[:n_tr, :memory]
        y_tr = toeplitz[:n_tr, memory]
        X_tst = toeplitz[n_tr:, :memory]
        y_tst = toeplitz[n_tr:, memory]

        # Train SVR model
        svr = SVR()
        svr.fit(X_tr, y_tr)
        tr_scores.append(svr.score(X_tr, y_tr))
        tst_scores.append(svr.score(X_tst, y_tst))
        svr_tst_hat.append(svr.predict(X_tst))

    return tr_scores, tst_scores, svr_tst_hat