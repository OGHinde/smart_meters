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
import pandas as pd
pd.options.mode.chained_assignment = None  # Eliminate pandas' warnings. Default='warn'
import numpy as np
from statsmodels.tsa.ar_model import AR
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

m = 5 # predictor memory.

time_series_list = []
time_series_list.append(np.mean(clustering_matrix, axis=0))
n_clusters = len(time_series_list)
tst_scores = []

for time_series in time_series_list:
    l = time_series.shape[0]-time_series.shape[0]%(m+1) # time series length
    time_series = time_series[:l]
    
    full_matrix = np.reshape(time_series, (l/(m+1), (m+1)))
    
    n_tr = int(round(l*.4/(m+1)))
    n_val = int(round(l*.3/(m+1)))
    n_tst = int(full_matrix.shape[0]-n_tr-n_val)
    
    full_idx = np.arange(l/(m+1))
    tr_idx = np.random.choice(full_idx, n_tr, replace=False)
    full_idx = np.setxor1d(full_idx, tr_idx)
    val_idx = np.random.choice(full_idx, n_val, replace=False)
    tst_idx = np.setxor1d(full_idx, val_idx)    
    
    X_tr = full_matrix[tr_idx, :m]
    y_tr = full_matrix[tr_idx, m]
    X_val = full_matrix[val_idx, :m]
    y_val = full_matrix[val_idx, m]
    X_tst = full_matrix[tst_idx, :m]
    y_tst = full_matrix[tst_idx, m]
    
    # Data normalization
    mx = np.mean(X_tr, axis=0, dtype=np.float64)
    stdx = np.std(X_tr, axis=0, dtype=np.float64)
    X_tr = np.divide((X_tr-np.tile(mx, [n_tr, 1])), np.tile(stdx, [n_tr, 1]))
    X_val = np.divide((X_val-np.tile(mx, [n_val, 1])), np.tile(stdx, [n_val, 1]))
    X_tst = np.divide((X_tst-np.tile(mx, [n_tst, 1])), np.tile(stdx, [n_tst, 1]))
    
    gp = GaussianProcess()
    gp.fit(X_tr, y_tr)
    tst_scores.append(gp.score(X_tst, y_tst))
#    val_score = gp.score(X_val, y_val)

for k, score in enumerate(tst_scores):
    print "\nGP test score for cluster %d = %d" % (k, score)
#print "GP validation score = ", val_score