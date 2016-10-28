# -*- coding: utf-8 -*-
"""
Prediction functions.

GP_pred(X, y, m): windowed Gaussian process prediction.

@author: oghinde
"""

# Initialise and import.
print '\nInit.'
import resources
import ClusterWeeks
import ClusterBox
import sys
import pandas as pd
pd.options.mode.chained_assignment = None  # Eliminate pandas' warnings. Default='warn'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcess
from sklearn.svm import SVR


print("Python version " + sys.version)

# Check that the paths to the data are correct!
print "\nInput path set to "+resources.base_matrix_1h

print "\nLoading Data..."
# Load preprocessed data respecting datetime index.
with open(resources.base_matrix_1h, 'r') as f:
    clustering_matrix, codes = pickle.load(f) 

K = 0  # number of clusters.
m = 96  # predictor memory.
dims = clustering_matrix.shape

print "Number of clusters = {}".format(K)
print "Predictor memory = {}".format(m)

print "\nNormalizing data..."
mx = np.mean(clustering_matrix, axis=1)
stdx = np.std(clustering_matrix, axis=1, dtype=np.float64)
#clustering_matrix = clustering_matrix-np.array([mx,]*dims[1]).transpose()
clustering_matrix = np.divide((clustering_matrix-np.array([mx,]*dims[1]).transpose()), 
                               np.array([stdx,]*dims[1]).transpose())

time_series_list = []
if K == 0:
    print "Loading week clustering results..."
    cluster_index_list, weights = ClusterWeeks.week_based_clustering()
    for i in range(len(clustering_index_list)):
        time_series_list.append(np.mean(clustering_matrix[clustering_index_list[i],:], axis=0))
elif K == 1:       
    print "\nNo clustering specified."
    time_series_list.append(np.mean(clustering_matrix, axis=0))
else:
    print "\nClustering..."
    clustering_indexes = np.squeeze(ClusterBox.ClusterBox(clustering_matrix, K))
    for k in range(K):
        time_series_list.append(np.mean(clustering_matrix[clustering_indexes[k],:], axis=0))
    
K = len(time_series_list)
val_scores = []
tst_scores = []
tr_scores = []
svr_tst_hat = []

print "\nPrediction..."
for time_series in time_series_list:
    l = time_series.shape[0]-time_series.shape[0]%(m+1) # time series length
    time_series = time_series[:l]

    # Generate and normalize Toeplitz matrix
    toeplitz = np.zeros((l-m, m+1,))               
    for i in range(l-m):
        toeplitz[i, :] = time_series[i:i+m+1]        
    mx = np.mean(toeplitz, axis=1)
    stdx = np.std(toeplitz, axis=1, dtype=np.float64)
    toeplitz = np.divide((toeplitz-np.array([mx,]*(m+1)).transpose()), 
                           np.array([stdx,]*(m+1)).transpose())
    
    n_tr = int((l-m)*0.4)
    n_val = int((l-m)*0.3)
    n_tst = l-m-n_tr-n_val
    
    X_tr = toeplitz[:n_tr, :m]
    y_tr = toeplitz[:n_tr, m]
    X_val = toeplitz[n_tr:n_tr+n_val,:][:,:m]
    y_val = toeplitz[n_tr:n_tr+n_val, m]
    X_tst = toeplitz[n_tr+n_val:, :m]
    y_tst = toeplitz[n_tr+n_val:, m]
    
    svr = SVR()
    svr.fit(X_tr, y_tr)
    tst_score = svr.score(X_tst, y_tst)  
    val_score = svr.score(X_val, y_val)
    tr_score = svr.score(X_tr, y_tr)

    svr_tst_hat.append(svr.predict(X_tst))
    
    tst_scores.append(tst_score)
    val_scores.append(val_score)
    tr_scores.append(tr_score)

for k, score in enumerate(tst_scores):
    print "\nSVR test score for cluster {} = {}".format(k, score)
print "\nOverall test score = {}".format(np.mean(tst_scores))



#%% Plotting block
    
fig = plt.figure()    
plt.plot(y_tst, 'k', label='Real target')
plt.plot(np.mean(svr_tst_hat, axis=0), 'r--', label='SVR prediciton')
plt.legend()
plt.title('1 week prediction results.')
axes = plt.gca()
axes.set_xlim([2000,2600])
plt.show()
fig.savefig(str(K)+' clusters_week_prediction.png', format='png', dpi=500, bbox_inches='tight')

fig = plt.figure()    
plt.plot(y_tst, 'k', label='Real target')
plt.plot(np.mean(svr_tst_hat, axis=0), 'r--', label='SVR prediciton')
plt.legend()
plt.title('1 year prediction results.')
axes = plt.gca()
plt.show()
fig.savefig(str(K)+' clusters_year_prediction.png', format='png', dpi=500, bbox_inches='tight')