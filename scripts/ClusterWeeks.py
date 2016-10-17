# -*- coding: utf-8 -*-

import resources
import sys
import numpy as np
import pandas as pd
print 'Importing cPickle...'
try:
    import cPickle as pickle
    print "Success!"
except:
    import pickle
    print "Failed. Defaulted to pickle."
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

def week_based_clustering(n_week_clusters=8, n_user_clusters=4, week_cluster_method='GMM', 
					user_cluster_method='GMM', aggregation_period='D'):
	
	"""Perform clustering based on the feature extraction results.

	First we extract weekly behaviours. 
	We then characterise each user in terms of their weekly behaviours.
	Finally, we cluster users according to this characterization.

    Keyword arguments:
    n_week_clusters 	-- number of weekly behaviours to define
    n_user_clusters 	-- the number of user clusters to define
	week_cluster_method -- the clustering algorithm to use for weekly behaviours 
	user_cluster_method -- the clustering algorithm to use for users
	aggregation_period	-- the period used to summarize weekly information
    """

	print 'Performing clustering of weekly behaviours...'
	
	# INIT STAGE
	methods = ['GMM', 'KMeans']
	aggregators = ['D', '8h', '6h']
	if n_week_clusters<2 or n_user_clusters<2:
		raise ValueError('Please specify a valid number of clusters (more than two).')
	if week_cluster_method not in methods or user_cluster_method not in methods:
		raise ValueError('Wrong method or no method specified.')
	if aggregation_period not in aggregators:
		raise ValueError('Wrong period or no period specified.')

	print 'Loading data...'
	EPS = sys.float_info.epsilon 
	with open(resources.clustering_matrix+aggregation_period+'.pickle', 'r') as f:
		clustering_matrix, code_index = pickle.load(f)
	code_index = np.asarray(code_index)
	# np.unique returns an ordered array and we don't want that
	_, idx = np.unique(code_index, return_index=True)
	unique_codes = code_index[np.sort(idx)]
	n_codes = len(unique_codes)
	dims = clustering_matrix.shape

	# NORMALIZE DATA
	mx = np.mean(clustering_matrix, axis=1)
	stdx = np.std(clustering_matrix, axis=1, dtype=np.float64)+EPS
	clustering_matrix = np.divide((clustering_matrix-np.array([mx,]*dims[1]).transpose()), 
                               		np.array([stdx,]*dims[1]).transpose())

	# PERFORM WEEK CLUSTERING
	if week_cluster_method=='KMeans':
		print 'Performing KMeans clustering for weekly behaviours...'
		clst = KMeans(n_clusters=n_week_clusters, precompute_distances=True, max_iter=500, n_init=20)
		clst.fit(clustering_matrix)
		labels = clst.predict(clustering_matrix)
		# compute "pseudo" loglikelyhoods
		weights = np.zeros((dims[0], n_week_clusters))
		for i, k in enumerate(labels):
			weights[i, k] = 1
	
	if week_cluster_method=='GMM':
		print 'Performing GMM clustering for weekly behaviours...'
		clst = GaussianMixture(n_components=n_week_clusters, covariance_type='full', random_state=0)
		clst.fit(clustering_matrix)
		labels = clst.predict(clustering_matrix)
		# compute log likelyhoods
		weights = clst.predict_proba(clustering_matrix)

	# PERFORM USER CLUSTERING
	# Generate a dataframe with users and loglikelyhoods for convenience
	weights_dframe = pd.DataFrame(weights, columns=['C'+str(i) for i in range(n_week_clusters)])
	weights_dframe['Meter_code'] = pd.Series(code_index)
	weights_dframe = weights_dframe.set_index('Meter_code')

	# Aggregate the log likelyhoods for each user -> feature extraction
	user_weight_means = np.zeros((n_codes, n_week_clusters))
	for i in range(n_codes):
   		aux = weights_dframe.ix[unique_codes[i]]
   		user_weight_means[i, :] = np.mean(aux.values, axis=0)

   	if user_cluster_method=='KMeans':
   		print 'Performing KMeans clustering for users...'
   		clst = KMeans(n_clusters=n_user_clusters, precompute_distances=True, max_iter=500, n_init=20)
		clst.fit(user_weight_means)
		labels = clst.predict(user_weight_means)

   	if user_cluster_method=='GMM':
   		print 'Performing GMM clustering for users...'
		clst = GaussianMixture(n_components=n_user_clusters, covariance_type='full', random_state=0)
		clst.fit(user_weight_means)
		labels = clst.predict(user_weight_means)

	# Save results
	print 'Saving results...'
	cluster_index_list = []
	for k in range(n_user_clusters):
	    cluster_index_list.append(np.where(labels == k)[0].tolist())
	    
	return cluster_index_list