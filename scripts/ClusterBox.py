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

import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from scipy.spatial import distance

#General function
def ClusterBox(clustering_matrix, K):
    clust_alg =  {1: 'Kmeans', 2:'GMM', 3:'Spectral'}
    method = None
    while method not in clust_alg.keys():
        try:
            method = int(raw_input('Select clustering method (introduce the number): \n{}'.format(clust_alg)))
        except ValueError:
            print 'A number please...'
        if method not in clust_alg.keys():
            print 'Ah ah ah, you didn\'t say the magic word'


    #KMEANS CLUSTERING
    if clust_alg[method]== 'Kmeans':

        clf = cluster.KMeans(n_clusters=K, precompute_distances=True, max_iter=500, n_init=20)
        clf.fit(clustering_matrix)

        # Analize clustering results.
        cluster_sizes = []
        codes_per_cluster = []
        for k in range(K):
            # np.where returns a tuple, indexes are in position [0]
            cluster_sizes.append(len(np.where(clf.labels_ == k)[0]))
            codes_for_this_cluster = [np.where(clf.labels_ == k)]
            codes_per_cluster.append(codes_for_this_cluster)



    #GMM CLUSTERING
    if clust_alg[method]== 'GMM':
        clf = cluster.mixture.GMM(n_components=K, covariance_type='full', n_init=2, n_iter=10)
        clf.fit(clustering_matrix)
        codes_labels = clf.predict(clustering_matrix)
        # Analize clustering results.
        cluster_sizes = []
        codes_per_cluster = []
        for k in range(K):
            # np.where returns a tuple, indexes are in position [0]
            cluster_sizes.append(len(np.where(codes_labels == k)[0]))
            codes_for_this_cluster = [np.where(codes_labels == k)]
            codes_per_cluster.append(codes_for_this_cluster)



    # SPECTRAL CLUSTERING
    if clust_alg[method]== 'Spectral':
        clf = cluster.SpectralClustering(n_clusters=K, eigen_solver=None, random_state=None, n_init=10, gamma=0.01,
                                    affinity='rbf', eigen_tol=0.0, assign_labels='kmeans', kernel_params=None)
        clf.fit(clustering_matrix)
        codes_labels = clf.predict(clustering_matrix)
        # Analize clustering results.
        cluster_sizes = []
        codes_per_cluster = []
        for k in range(K):
            # np.where returns a tuple, indexes are in position [0]
            cluster_sizes.append(len(np.where(codes_labels == k)[0]))
            codes_for_this_cluster = [np.where(codes_labels == k)]
            codes_per_cluster.append(codes_for_this_cluster)

    #OUTPUT
    return codes_per_cluster

