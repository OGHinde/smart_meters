# -*- coding: utf-8 -*-
"""
Prediction module.
"""

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
print "Importing cPickle..."
# Use cPickle if possible. It's much faster!
try:
    import cPickle as pickle
    print "Success!"
except:
    import pickle
    print "Failed. Defaulted to pickle."