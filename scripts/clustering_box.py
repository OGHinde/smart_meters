# -*- coding: utf-8 -*-
'''
Script clustering:
Input - matrix NxM where N=number of samples  M = features (hours,day,week,...)
Output - K-elements list, K=cluster number, each element contains indexes which belong to
        the cluster.
Clustering methods : 'Kmeans';, 'GMM', 'Spectral'
'''

#Some imports
import sys
import resources
print "Importing cPickle..."
# Use cPickle if possible. It's much faster!
try:
    import cPickle as pickle
    print "Success!"
except:
    import pickle
    print "Failed. Defaulted to pickle."
import pandas as pd
import numpy as np
from sklearn import cluster
from sklearn import mixture
from scipy.spatial import distance


def baseline_clustering(clustering_matrix, n_user_clusters=4, cluster_method='GMM'):
    """Perform user clustering using the full time-series matrix.

    Keyword arguments:
    clustering_matrix   -- input data matrix.
    n_user_clusters     -- the number of user clusters to define.
    cluster_method      -- the clustering algorithm to use.
    """
    #KMEANS CLUSTERING
    if cluster_method == 'Kmeans':
        print "Performing KMeans clustering..."
        clf = cluster.KMeans(n_clusters=n_user_clusters, precompute_distances=True, max_iter=500, n_init=20)
        clf.fit(clustering_matrix)
        # Analize clustering results.
        cluster_sizes = []
        codes_per_cluster = []
        for k in range(n_user_clusters):
            # np.where returns a tuple, indexes are in position [0]
            cluster_sizes.append(len(np.where(clf.labels_ == k)[0]))
            codes_for_this_cluster = [np.where(clf.labels_ == k)]
            codes_per_cluster.append(codes_for_this_cluster)
    #GMM CLUSTERING
    if cluster_method == 'GMM':
        print "Performing GMM clustering..."
        clf = mixture.GaussianMixture(n_components=n_user_clusters, covariance_type='full', n_init=2, max_iter=500)
        clf.fit(clustering_matrix)
        codes_labels = clf.predict(clustering_matrix)
        # Analize clustering results.
        cluster_sizes = []
        codes_per_cluster = []
        for k in range(n_user_clusters):
            # np.where returns a tuple, indexes are in position [0]
            cluster_sizes.append(len(np.where(codes_labels == k)[0]))
            codes_for_this_cluster = [np.where(codes_labels == k)]
            codes_per_cluster.append(codes_for_this_cluster)
    # SPECTRAL CLUSTERING
    if cluster_method == 'Spectral':
        print "Performing spectral clustering..."
        clf = cluster.SpectralClustering(n_clusters=n_user_clusters, eigen_solver=None, random_state=None, n_init=10, gamma=0.01,
                                    affinity='rbf', eigen_tol=0.0, assign_labels='kmeans', kernel_params=None)
        clf.fit(clustering_matrix)
        codes_labels = clf.predict(clustering_matrix)
        # Analize clustering results.
        cluster_sizes = []
        codes_per_cluster = []
        for k in range(n_user_clusters):
            # np.where returns a tuple, indexes are in position [0]
            cluster_sizes.append(len(np.where(codes_labels == k)[0]))
            codes_for_this_cluster = [np.where(codes_labels == k)]
            codes_per_cluster.append(codes_for_this_cluster)
    # OUTPUT
    return codes_per_cluster


def week_based_clustering(clustering_matrix, code_index, n_week_clusters=8, n_user_clusters=4, 
                          week_cluster_method='GMM', user_cluster_method='GMM'):
    """Perform user clustering based on the feature extraction results from the 
    weekly behaviour clustering.

    First we extract weekly behaviours. 
    We then characterise each user in terms of their weekly behaviours.
    Finally, we cluster users according to this characterization.

    n_clusterseyword arguments:
    n_week_clusters     -- number of weekly behaviours to define
    n_user_clusters     -- the number of user clusters to define
    week_cluster_method -- the clustering algorithm to use for weekly behaviours 
    user_cluster_method -- the clustering algorithm to use for users
    aggregation_period  -- the period used to summarize weekly information
    """
    # INIT STAGE
    methods = ['GMM', 'KMeans']
    aggregators = ['D', '8h', '6h']
    # Value check
    if n_week_clusters<2 or n_user_clusters<2:
        raise ValueError('Please specify a valid number of clusters (more than two).')
    if week_cluster_method not in methods:
        raise ValueError('Wrong week clustering method specified: '+str(week_cluster_method))
    if user_cluster_method not in methods:
        raise ValueError('Wrong user clustering method specified: '+str(user_cluster_method))

    EPS = sys.float_info.epsilon 
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
        clst = cluster.KMeans(n_clusters=n_week_clusters, precompute_distances=True, max_iter=500, n_init=20)
        clst.fit(clustering_matrix)
        labels = clst.predict(clustering_matrix)
        # compute "pseudo" loglikelyhoods
        weights = np.zeros((dims[0], n_week_clusters))
        for i, k in enumerate(labels):
            weights[i, k] = 1
    if week_cluster_method=='GMM':
        print 'Performing GMM clustering for weekly behaviours...'
        clst = mixture.GaussianMixture(n_components=n_week_clusters, covariance_type='full', random_state=0)
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
        clst = cluster.KMeans(n_clusters=n_user_clusters, precompute_distances=True, max_iter=500, n_init=20)
        clst.fit(user_weight_means)
        labels = clst.predict(user_weight_means)
        weights = np.zeros((n_codes, n_user_clusters))
        for i, k in enumerate(labels):
            weights[i, k] = 1
    if user_cluster_method=='GMM':
        print 'Performing GMM clustering for users...'
        clst = mixture.GaussianMixture(n_components=n_user_clusters, covariance_type='full', random_state=0)
        clst.fit(user_weight_means)
        labels = clst.predict(user_weight_means)
        weights = clst.predict_proba(user_weight_means)
    cluster_index_list = []
    for k in range(n_user_clusters):
        cluster_index_list.append(np.where(labels == k)[0].tolist())
    # OUTPUT
    return cluster_index_list, weights

