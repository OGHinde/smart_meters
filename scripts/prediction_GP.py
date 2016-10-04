# -*- coding: utf-8 -*-
"""
Prediction functions.

GP_pred(X, y, m): windowed Gaussian process prediction.

@author: oghinde
"""

# Initialise and import.
print '\nInit.'
import resources
import sys
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None  # Eliminate pandas' warnings. Default='warn'
import scipy.io
import numpy as np
from sklearn.gaussian_process import GaussianProcess
from sklearn.svm import SVR
print 'Importing cPickle...'
# Use cPickle if possible. It's much faster!
try:
    import cPickle as pickle
    print 'Success!'
except:
    import pickle
    print 'Failed. Defaulted to pickle.'

print('Python version ' + sys.version)

# Check that the paths to the data are correct!
print '\nInput path set to '+resources.base_matrix_1h

# Load preprocessed data respecting datetime index.
with open(resources.base_matrix_1h, 'r') as f:
    clustering_matrix, codes = pickle.load(f) 

dims = clustering_matrix.shape
m = 192 # predictor memory.

time_series_list = []
# Normalise clustering matrix.
mx = np.mean(clustering_matrix, axis=1)
stdx = np.std(clustering_matrix, axis=1, dtype=np.float64)
#clustering_matrix = clustering_matrix-np.array([mx,]*dims[1]).transpose()
clustering_matrix = np.divide((clustering_matrix-np.array([mx,]*dims[1]).transpose()), 
                               np.array([stdx,]*dims[1]).transpose())

mean_ts = np.mean(clustering_matrix, axis=0)
scipy.io.savemat('/Users/oghinde/Academia/Research_projects/smart_grids'+
                '/smart_meters/data/mean_ts.mat', mdict={'mean_ts': mean_ts})
time_series_list.append(mean_ts)
n_clusters = len(time_series_list)
val_scores = []
tst_scores = []
tr_scores = []

for time_series in time_series_list:
    l = time_series.shape[0]-time_series.shape[0]%(m+1) # time series length
    time_series = time_series[:l]
    
    # Generate working matrix
    full_matrix = np.reshape(time_series, (l/(m+1), (m+1)))
    
    # Generate train, val and test indexes
    n_tr = int(round(l*.4/(m+1)))
    n_val = int(round(l*.3/(m+1)))
    n_tst = int(full_matrix.shape[0]-n_tr-n_val)
    full_idx = np.arange(l/(m+1))
    tr_idx = np.random.choice(full_idx, n_tr, replace=False)
    full_idx = np.setxor1d(full_idx, tr_idx)
    val_idx = np.random.choice(full_idx, n_val, replace=False)
    tst_idx = np.setxor1d(full_idx, val_idx)
    
    # Create train, val and test matrixes
    X_tr = full_matrix[tr_idx, :m]
    y_tr = full_matrix[tr_idx, m]
    X_val = full_matrix[val_idx, :m]
    y_val = full_matrix[val_idx, m]
    X_tst = full_matrix[tst_idx, :m]
    y_tst = full_matrix[tst_idx, m]
    
    regressor = SVR()
    regressor.fit(X_tr, y_tr)
    tst_score = regressor.score(X_tst, y_tst)  
    val_score = regressor.score(X_val, y_val)
    tr_score = regressor.score(X_tr, y_tr)
    tst_scores.append(tst_score)
    val_scores.append(val_score)
    tr_scores.append(tr_score)

for k, score in enumerate(tr_scores):
    print "\nSVR train score for cluster {} = {}".format(k, score)
    
for k, score in enumerate(val_scores):
    print "\nSVR validation score for cluster {} = {}".format(k, score)
    
for k, score in enumerate(tst_scores):
    print "\nSVR test score for cluster {} = {}".format(k, score)
