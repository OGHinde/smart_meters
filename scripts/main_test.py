# -*- coding: utf-8 -*-
"""Main test execution script"""

# Author: Óscar G. Hinde

print '\nInit.'
import resources
import clustering_box
import prediction_box
import sys
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("scenario", type=str, help="Baseline or weekly scenario.")
parser.add_argument("week_K", type=int, help="The number of week clusters.")
parser.add_argument("user_K", type=int, help="The number of user clusters.")
args = parser.parse_args()

print '\n\nSTART'
print '\Testing for {} scenario, with {} week behaviour groups and {} user groups.'.format(args.scenario, args.week_K, args.user_K)
n_tr = 0.4
n_val = 0.4
n_tst = n_tr-n_val

with open(resources.base_matrix_1h, 'r') as f:
    clustering_matrix, codes = pickle.load(f)

dims = clustering_matrix.shape
mx = np.mean(clustering_matrix, axis=1)
stdx = np.std(clustering_matrix, axis=1, dtype=np.float64)
#clustering_matrix = clustering_matrix-np.array([mx,]*dims[1]).transpose()
clustering_matrix = np.divide((clustering_matrix-np.array([mx,]*dims[1]).transpose()), 
                               np.array([stdx,]*dims[1]).transpose())

# LOAD DATA
if args.scenario == 'weekly':
	
	with open(resources.clustering_matrices_D, 'r') as f:
		tr_clustering_matrix, tr_code_index, val_clustering_matrix, val_code_index = pickle.load(f)

	print '\n\nCLUSTERING:\n'
	cluster_index_list, weights = clustering_box.week_based_clustering(tr_clustering_matrix, 
																	   tr_code_index, 
										 							   n_week_clusters=args.week_K, 
										 							   n_user_clusters=args.user_K)

	print '\n\nPREDICTION:\n'
	tr_scores, tst_scores, svr_tst_hat = prediction_box.toeplitz_SVR_test(clustering_matrix, cluster_index_list)
	avg_val = np.mean(val_scores)
	avg_series = np.mean(svr_val_hat, axis=0)

	print '\n\nRESULTS:\n'
	print 'Cluster validation scores for {} week behaviour groups and {} user groups:'.format(args.week_K, args.user_K)
	print val_scores
	print '\nAverage validation score:'
	print avg_val

	save_path = resources.val_results_path+'val_{}_{}.pickle'.format(args.week_K, args.user_K)
	print '\nSaving results to: '
	print save_path

	with open(save_path, 'w') as f:
		pickle.dump([avg_val, avg_series], f)

	print '\n\nDone!'
