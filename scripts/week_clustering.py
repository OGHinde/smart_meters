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
from sklearn.cluster import KMeans
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
EPS = sys.float_info.epsilon 

# Check that the paths to the data are correct!
print '\nInput path set to '+resources.clustering_matrix_D
print 'Output path set to '+resources.kmeans_figs

# Load matrix for clustering.
print 'Loading data...'
with open(resources.clustering_matrix_D, 'r') as f:
	clustering_matrix, code_index = pickle.load(f)
code_index = np.asarray(code_index)
# np.unique returns an ordered array and we don't want that
_, idx = np.unique(code_index, return_index=True)
unique_codes = code_index[np.sort(idx)]
n_codes = len(unique_codes)

# Normalize data.
means = np.mean(clustering_matrix, axis=1)
means = means[:, np.newaxis]
stds = np.std(clustering_matrix, axis=1)+EPS
stds = stds[:, np.newaxis]
clustering_matrix = np.divide(clustering_matrix - np.tile(means, (1, 7)), np.tile(stds, (1, 7)))
#clustering_matrix = np.concatenate((clustering_matrix, means, stds), axis=1)

# Initial range of values for K.
K = 8
cluster_size_thr = (clustering_matrix.shape[0]/K)*.1

# K-means clustering.
clf = KMeans(n_clusters=K, precompute_distances=True, n_jobs=-1)
clf.fit(clustering_matrix)
labels = clf.labels_

# Create and save dataframe
clustering_dframe = pd.DataFrame(clustering_matrix, columns=x_tick_labels)
clustering_dframe['Cluster'] = pd.Series(labels)
clustering_dframe['Meter_code'] = pd.Series(code_index)
clustering_dframe['Mean'] = pd.Series(np.squeeze(means))
clustering_dframe['STD'] = pd.Series(np.squeeze(stds))
clustering_dframe = clustering_dframe[['Meter_code', 'Cluster', 'Mean', 'STD']+x_tick_labels]

# Analize clustering results.
cluster_sizes = []
codes_per_cluster = []
Nclusters_per_code = np.zeros((unique_codes.shape[0],1, ))
for k in range(K):
    # np.where returns a tuple, indexes are in position [0]
    weeks = np.where(labels == k)[0]
    cluster_sizes.append(len(weeks))
    _, idx, counts = np.unique(code_index[weeks], return_index=True, return_counts=True)
    codes_for_this_cluster = code_index[weeks][np.sort(idx)]
    codes_per_cluster.append(len(codes_for_this_cluster))
    indices = [np.where(unique_codes == code)[0][0] for code in codes_for_this_cluster]
    Nclusters_per_code[indices] += 1

# Write text log.
with open(resources.kmeans_log, "w") as text_file:
    text_file.write("PRELIMINARY RESULTS FOR K-MEANS CLUSTERING")
    text_file.write("\n\nNumber of clusters: {0}".format(K))
    for k in range(K):
        text_file.write("\n\nCluster {0}:".format(k))
        text_file.write("\n\tTotal cluster size: {0}".format(cluster_sizes[k]))
        text_file.write("\n\tNumber of smart meters: {0}".format(codes_per_cluster[k]))    
        
# Saving block
clustering_dframe.to_csv(resources.clustering_dframe)
with open(resources.general_data, 'w') as f:
	pickle.dump([unique_codes, n_codes], f)
code_index = np.asarray(code_index)
        
    
# Plotting block. Note that this won't work when executed from the terminal.
resources.delete_figures(resources.kmeans_figs)    # Clean previous plots
for k in range(K):
    fig = plt.figure()
    plt.plot(clf.cluster_centers_[k, :7], c=np.random.rand(3, 1), label='Centroid')
    for i in range(5):
        if (i<cluster_sizes[k]):
            # select from cluster k a random week to plot
            index = np.where(labels==k)[0][np.random.randint(0, cluster_sizes[k])]
            plt.plot(clustering_matrix[index, :7], c=np.random.rand(3, 1), linestyle='-.')
    plt.title('Cluster '+str(k)+' information')
    plt.xticks(range(len(x_tick_labels)), x_tick_labels, rotation='vertical', fontsize=10)
    plt.ylabel('Active power')
    plt.legend()
    fig.savefig(resources.kmeans_figs+'cluster_'+str(k)+'info.png',  format='png', dpi=500,  bbox_inches='tight')  

fig = plt.figure()
plt.hist(Nclusters_per_code)
plt.title('Number of clusters per smart meter')
plt.xlabel('Number of clusters')
plt.ylabel('Number of smart meters')
fig.savefig(resources.kmeans_figs+'histogram.png', format='png', dpi=500, bbox_inches='tight')