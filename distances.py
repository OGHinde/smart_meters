# -*- coding: utf-8 -*-
'''
K-means clustering preliminary tests.
'''

# Initialise and import.
print '\nInit.'
import resources
import sys
import numpy as np
import matplotlib.pyplot as plt
import KLdiv as KL
import pandas as pd

print 'Importing cPickle...'
# Use cPickle if possible. It's much faster!
try:
    import cPickle as pickle
    print 'Success!'
except:
    import pickle
    print 'Failed. Defaulted to pickle.'

print('Python version ' + sys.version)

x_tick_labels = ['Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri']

# Check that the paths to the data are correct!
print '\nInput path set to '+resources.clustering_dframe

# Load matrix for clustering.
print 'Loading data...'
clustering_dframe = pd.read_csv(resources.clustering_dframe)
clustering_dframe = clustering_dframe.drop('Unnamed: 0', 1)
clustering_dframe = clustering_dframe.set_index('Meter_code')
with open(resources.general_data, 'r') as f:
    unique_codes, n_codes = pickle.load(f)

# Pre-compute per-user cluster frequencies
cluster_frequencies = np.zeros((n_codes, 8))

for i in range(n_codes):
    aux = clustering_dframe.ix[unique_codes[i]]
    counts = aux['Cluster'].value_counts(normalize=True)
    cluster_frequencies[i, counts.index] = counts[counts.index]
    
# Calculate euclidean distance matrix
euc_matrix = np.zeros((n_codes, n_codes))
for i in range(n_codes):
    for j in range(n_codes):
        euc_matrix[i, j] = np.linalg.norm(cluster_frequencies[i, :]-cluster_frequencies[j, :])
        
# Calculate KL divergence matrix
#KL_matrix = np.zeros((n_codes, n_codes))
#weeks = clustering_dframe[(x_tick_labels)]
#for i in range(n_codes):
#    print i
#    for j in range(i, n_codes):
#        X0 = weeks.ix[unique_codes[i]].values
#        X1 = weeks.ix[unique_codes[j]].values
#        KL_matrix[i, j] = KL.ComputeDKL(X0, X1)

#%%
fig = plt.figure()
plt.imshow(euc_matrix)

fig.savefig(resources.kmeans_figs+'euclidean_distance.png',  format='png', dpi=500,  bbox_inches='tight')  
