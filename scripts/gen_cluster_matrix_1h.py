# -*- coding: utf-8 -*-
"""
Resample data to 6 hour averages.

Reorganise data into weekly segments for each meter code.
"""
# Initialise and import.
print '\nInit.'
import resources
import sys
import pandas as pd
pd.options.mode.chained_assignment = None  # Eliminate pandas' warnings. Default='warn'
import numpy as np
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
print '\nInput path set to '+resources.smartMeters_prep_csv
print 'Output path set to '+resources.clustering_matrices_D

periods = 7*24
series_length = 54*7*24
n_tr_weeks = int(resources.tr_ratio*54)
n_val_weeks = int(resources.val_ratio*54)
n_tr = n_tr_weeks*7*24
n_val = n_val_weeks*7*24

# Load preprocessed data respecting datetime index.
data = pd.read_csv(resources.smartMeters_prep_csv, index_col=0, 
                   parse_dates=True)                  

# Generate matrix for clustering.
print '\nGenerating numpy array for clustering purposes...'
codes = list(np.unique(data['COD_METER'].values))
code_groups = data.groupby('COD_METER')
n_entries = 54*len(codes) 

#clustering_matrix = np.empty((n_entries, periods,))
#code_index = []
#cont = 0
tr_clustering_matrix = np.empty((n_tr_weeks*len(codes), periods,))
tr_code_index = []
tr_cont = 0
val_clustering_matrix = np.empty((n_val_weeks*len(codes), periods,))
val_code_index = []
val_cont = 0

for c, code in enumerate(codes):
    series = code_groups.get_group(code)['VAL_AI'][:series_length]
    # Weekly ordering
    for idx in range(n_tr_weeks):
        tr_clustering_matrix[tr_cont, :] = series[idx*periods:idx*periods+periods]
        tr_code_index.append(code)
        tr_cont += 1
    for idx in range(n_tr_weeks, n_tr_weeks+n_val_weeks):
        val_clustering_matrix[val_cont, :] = series[idx*periods:idx*periods+periods]
        val_code_index.append(code)
        val_cont += 1
#    for idx in range(54):
#        clustering_matrix[cont, :] = resampled_series[idx*periods:idx*periods+periods]
#        code_index.append(code)
#        cont += 1


print 'Saving clustering matrix...'
with open(resources.clustering_matrices_1h, 'w') as f:
    pickle.dump([tr_clustering_matrix, tr_code_index, val_clustering_matrix, val_code_index], f)

print '\nDone!'
