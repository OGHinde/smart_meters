# -*- coding: utf-8 -*-
"""
Prepare data for baseline clustering.

Resample data into daily periods.
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
print 'Output path set to '+resources.base_matrix_D

periods = 7*54

# Load preprocessed data respecting datetime index.
data = pd.read_csv(resources.smartMeters_prep_csv, index_col=0, 
                   parse_dates=True)

codes = list(np.unique(data['COD_METER'].values))
code_groups = data.groupby('COD_METER')

clustering_matrix = np.empty((len(codes), periods,))

for c, code in enumerate(codes):
    # Resample to daily average
    series = code_groups.get_group(code)['VAL_AI'].resample('D').mean()
    clustering_matrix[c, :] = series[:periods]

print 'Saving clustering matrix...'
with open(resources.base_matrix_D, 'w') as f:
    pickle.dump([clustering_matrix, codes], f)

print '\nDone!'