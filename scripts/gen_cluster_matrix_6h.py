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
print 'Output path set to '+resources.clustering_matrix_6h

periods = 7*4

# Load preprocessed data respecting datetime index.
data = pd.read_csv(resources.smartMeters_prep_csv, index_col=0, 
                   parse_dates=True)                  

# Generate matrix for clustering.
print '\nGenerating numpy array for clustering purposes...'
codes = list(np.unique(data['COD_METER'].values))
code_groups = data.groupby('COD_METER')
n_entries = 54*len(codes) 
clustering_matrix = np.empty((n_entries, periods,))
code_index = []
cont = 0
for c, code in enumerate(codes):
    # Resample to daily average
    series = code_groups.get_group(code)['VAL_AI'].resample('6H').mean()
    # Weekly ordering
    for idx in range(54):
        clustering_matrix[cont, :] = series[idx*periods:idx*periods+periods]
        code_index.append(code)
        cont += 1

print 'Saving clustering matrix...'
with open(resources.clustering_matrix_6h, 'w') as f:
    pickle.dump([clustering_matrix, code_index], f)

print '\nDone!'
