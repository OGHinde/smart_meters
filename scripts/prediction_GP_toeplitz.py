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
import matplotlib.pyplot as plt
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

m = 96 # predictor memory.
dims = clustering_matrix.shape

time_series_list = []
mx = np.mean(clustering_matrix, axis=1)
stdx = np.std(clustering_matrix, axis=1, dtype=np.float64)
#clustering_matrix = clustering_matrix-np.array([mx,]*dims[1]).transpose()
clustering_matrix = np.divide((clustering_matrix-np.array([mx,]*dims[1]).transpose()), 
                               np.array([stdx,]*dims[1]).transpose())
                               
time_series_list.append(np.mean(clustering_matrix, axis=0))
n_clusters = len(time_series_list)
val_scores = []
tst_scores = []
tr_scores = []

for time_series in time_series_list:
    l = time_series.shape[0]-time_series.shape[0]%(m+1) # time series length
    time_series = time_series[:l]

    # Generate and normalize Toeplitz matrix
    toeplitz = np.zeros((l-m, m+1,))               
    for i in range(l-m):
        toeplitz[i, :] = time_series[i:i+m+1]        
#    mx = np.mean(toeplitz, axis=1)
#    stdx = np.std(toeplitz, axis=1, dtype=np.float64)
#    toeplitz = np.divide((toeplitz-np.array([mx,]*(m+1)).transpose()), 
#                            np.array([stdx,]*(m+1)).transpose())
    
    n_tr = int((l-m)*0.4)
    n_val = int((l-m)*0.3)
    n_tst = l-m-n_tr-n_val
    
    X_tr = toeplitz[:n_tr, :m]
    y_tr = toeplitz[:n_tr, m]
    X_val = toeplitz[n_tr:n_tr+n_val,:][:,:m]
    y_val = toeplitz[n_tr:n_tr+n_val, m]
    X_tst = toeplitz[n_tr+n_val:, :m]
    y_tst = toeplitz[n_tr+n_val:, m]

    gp = GaussianProcess(normalize=False)
    gp.fit(X_tr, y_tr)
    tst_score = gp.score(X_tst, y_tst)  
    val_score = gp.score(X_val, y_val)
    tr_score = gp.score(X_tr, y_tr)

    gp_tst_hat = gp.predict(X_tst)
    
    tst_scores.append(tst_score)
    val_scores.append(val_score)
    tr_scores.append(tr_score)

    for k, score in enumerate(tr_scores):
        print "\nGP train score for cluster {} = {}".format(k, score)
    for k, score in enumerate(val_scores):
        print "\nGP validation score for cluster {} = {}".format(k, score)
    for k, score in enumerate(tst_scores):
        print "\nGP test score for cluster {} = {}".format(k, score)
    
#%% Plotting block
    
fig = plt.figure()    
plt.plot(y_tst, 'k', label='Real target')
plt.plot(gp_tst_hat, 'r--', label='GP prediciton')
plt.legend()
plt.title('1 week prediction results.')
axes = plt.gca()
axes.set_xlim([2000,2168])
plt.show()