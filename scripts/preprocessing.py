# -*- coding: utf-8 -*-
"""
Initial data preprocessing.

Eliminate meter codes with incomplete data, drop irrelevant columns,
and reindex the data according to date and time.
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
print '\nInput path set to '+resources.smartMeters_raw_csv
print 'Output path set to '+resources.smartMeters_prep_csv

# Data load
print '\nLoading data...'
data = pd.read_csv(resources.smartMeters_raw_csv)

# Index data with time variable in datetime format for convenience
data['FEC_LECTURA'] = pd.to_datetime(data['FEC_LECTURA'])
data = data.set_index(pd.DatetimeIndex(data['FEC_LECTURA']))
data = data.drop(['FEC_LECTURA'], axis=1)
data.sort_index(inplace=True)

print 'Extracting variables...'
# Extract some useful info and group the data by meter codes
codes = list(np.unique(data['COD_METER'].values))
date_times = list(np.unique(data.index.values))
code_groups = data.groupby('COD_METER')

# Check for missing data
incomplete = []
for i, code in enumerate(codes):
    if (len(code_groups.get_group(code)) < 9150):
        incomplete.append(code)        
print '\nTotal number of incomplete meters = '+str(len(incomplete))
good_codes = list(set(codes)-set(incomplete))

data = data[data['COD_METER'].isin(good_codes)]

print '\nSaving data...'
data.to_csv(resources.smartMeters_prep_csv)

print '\nDone!\n'
