# -*- coding: utf-8 -*-
'''
GMM clustering preliminary tests.
'''

# Initialise and import.
print '\nInit.'
import resources
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GMM
from sklearn.decomposition import PCA
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

# Number of clusters K.
K = 8

# GMM clustering.
clf = GMM(n_components=K, covariance_type='full', random_state=0)
clf.fit(clustering_matrix)
labels = clf.predict(clustering_matrix)
prob, weights = clf.score_samples(clustering_matrix)
#%%
# Create dataframes
clustering_dframe = pd.DataFrame(clustering_matrix, columns=x_tick_labels)
clustering_dframe['Cluster'] = pd.Series(labels)
clustering_dframe['Meter_code'] = pd.Series(code_index)
clustering_dframe['Mean'] = pd.Series(np.squeeze(means))
clustering_dframe['STD'] = pd.Series(np.squeeze(stds))
clustering_dframe = clustering_dframe[['Meter_code', 'Cluster', 'Mean', 'STD']+x_tick_labels]

weights_dframe = pd.DataFrame(weights, columns=['C'+str(i) for i in range(K)])
weights_dframe['Meter_code'] = pd.Series(code_index)
weights_dframe = weights_dframe.set_index('Meter_code')

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
with open(resources.gmm_log, "w") as text_file:
    text_file.write("PRELIMINARY RESULTS FOR GMM CLUSTERING")
    text_file.write("\n\nNumber of clusters: {0}".format(K))
    for k in range(K):
        text_file.write("\n\nCluster {0}:".format(k))
        text_file.write("\n\tTotal cluster size: {0}".format(cluster_sizes[k]))
        text_file.write("\n\tNumber of smart meters: {0}".format(codes_per_cluster[k]))    
        
# Saving block
clustering_dframe.to_csv(resources.week_clustering_dframe)
with open(resources.general_data, 'w') as f:
	pickle.dump([unique_codes, n_codes], f)
code_index = np.asarray(code_index)
    
# Plotting block. Note that this won't work when executed from the terminal.
resources.delete_figures(resources.gmm_figs)    # Clean previous plots
for k in range(K):
    fig = plt.figure()
    plt.plot(clf.means_[k, :7], c=np.random.rand(3, 1), label='Centroid')
    for i in range(5):
        if (i<cluster_sizes[k]):
            # select from cluster k a random week to plot
            index = np.where(labels==k)[0][np.random.randint(0, cluster_sizes[k])]
            plt.plot(clustering_matrix[index, :7], c=np.random.rand(3, 1), linestyle='-.')
    plt.title('Cluster '+str(k)+' information')
    plt.xticks(range(len(x_tick_labels)), x_tick_labels, rotation='vertical', fontsize=10)
    plt.ylabel('Active power')
    plt.legend()
    fig.savefig(resources.gmm_figs+'cluster_'+str(k)+'info.png',  format='png', dpi=500,  bbox_inches='tight')  

fig = plt.figure()
plt.hist(Nclusters_per_code)
plt.title('Number of clusters per smart meter')
plt.xlabel('Number of clusters')
plt.ylabel('Number of smart meters')
fig.savefig(resources.gmm_figs+'histogram.png', format='png', dpi=500, bbox_inches='tight')

''' CODE METER ANALYSIS'''
# Calculate the likelyhood means for each meter code 
weight_means = np.zeros((n_codes, K))
for i in range(n_codes):
   aux = weights_dframe.ix[unique_codes[i]]
   weight_means[i, :] = np.mean(aux.values, axis=0)

#%% GMM clustering of code meters.
K2 = 3
clf2 = GMM(n_components=K2, covariance_type='full', random_state=0)
clf2.fit(weight_means)
labels = clf2.predict(weight_means)

cluster_index_list = []
for k in range(K2):
    cluster_index_list.append(np.where(labels == k)[0].tolist())
    
with open(resources.cluster_index_list, 'w') as f:
    pickle.dump(cluster_index_list, f)

dec = PCA(n_components=8, whiten=True)
PCA_weights = dec.fit_transform(weight_means)

# Calculate euclidean distance matrix after PCA
PCA_euc_matrix = np.zeros((n_codes, n_codes))
for i in range(n_codes):
    for j in range(n_codes):
        PCA_euc_matrix[i, j] = np.linalg.norm(PCA_weights[i, :]-PCA_weights[j, :])

# Plotting block
# fig = plt.figure()
# for i in range(5):
#     if (i<cluster_sizes[k]):
#         # select from cluster k a random week to plot
#         index = np.where(labels==k)[0][np.random.randint(0, cluster_sizes[k])]
#         plt.plot(clustering_matrix[index, :7], c=np.random.rand(3, 1), linestyle='-.')
# plt.title('Cluster '+str(k)+' information')
# plt.xticks(range(len(x_tick_labels)), x_tick_labels, rotation='vertical', fontsize=10)
# plt.ylabel('Active power')
# plt.legend()
# fig.savefig(resources.kmeans_figs+'cluster_'+str(k)+'info.png',  format='png', dpi=500,  bbox_inches='tight')  

##%%
#fig = plt.figure()
#c = ['r', 'g', 'b', 'y'] 
#ax = Axes3D(fig)
#for i in range(K2):
#    ax.scatter(PCA_weights[labels == i,:][:,0], PCA_weights[labels == i,:][:,1], PCA_weights[labels == i,:][:,2], c=c[i])
#
#for ii in xrange(0,360,1):
#    ax.view_init(elev=10., azim=ii)
#    fig.savefig(resources.gmm_figs+"movie/movie"+str(ii)+".png", format='png', dpi=200)
#
##%%
#fig = plt.figure()
#c = ['r', 'g', 'b']
#for i in range(K2):
#    plt.scatter(PCA_weights[labels == i,:][:,0], PCA_weights[labels == i,:][:,1], c=c[i], marker='o')
#                
#plt.show()  
#fig.savefig(resources.gmm_figs+'2D_meter_code_clusters.png', format='png', dpi=200, bbox_inches='tight')
#
##%%
#
#c = ['maroon', 'orangered','orange', 'peachpuff']
#for i in range(K2):
#    idx = np.argsort(clf2.means_[i])[:4]
#    fig = plt.figure()
#    for j in range(len(idx)):
#        plt.plot(clf.means_[idx[j], :7], c=c[j])
#    plt.xticks(range(len(x_tick_labels)), x_tick_labels, rotation='vertical', fontsize=10)
#    fig.savefig(resources.gmm_figs+'group'+str(i)+'.png', format='png', dpi=200, bbox_inches='tight')



