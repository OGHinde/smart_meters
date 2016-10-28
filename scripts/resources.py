# -*- coding: utf-8 -*-

'''
Path variables and other general resources.

Modify this module with your relevant local paths.
'''

# Import useful modules
import os
import numpy as np
import matplotlib.pyplot as plt

# General data paths.
smartMeters_raw_csv = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/data/extraccionLimpios.csv'
smartMeters_prep_csv = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/data/extraccion_prep.csv'

clustering_matrices = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/data/clustering_matrices_'
clustering_matrices_D = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/data/clustering_matrices_D.pickle'
clustering_matrices_8h = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/data/clustering_matrices_8h.pickle'
clustering_matrices_6h = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/data/clustering_matrices_6h.pickle'

base_matrix_D = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/data/base_matrix_D.pickle'
base_matrix_8h = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/data/base_matrix_8h.pickle'
base_matrix_6h = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/data/base_matrix_6h.pickle'
base_matrix_1h = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/data/base_matrix_1h.pickle'

week_clustering_dframe = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/data/week_clustering_dataframe.csv'
general_data = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/data/general_data.pickle'

cluster_index_list_path = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/data/clustering_results/cluster_index_list_'

kmeans_figs = '/Users/oghinde/Academia/Research_projects/smart_grids/smart_meters/figures/Kmeans/'
kmeans_log = '/Users/oghinde/Academia/Research_projects/smart_grids/smart_meters/figures/Kmeans/K_means_results.txt'
gmm_figs = '/Users/oghinde/Academia/Research_projects/smart_grids/smart_meters/figures/GMM/'
gmm_log = '/Users/oghinde/Academia/Research_projects/smart_grids/smart_meters/figures/Kmeans/GMM_results.txt'

val_results_path = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/results/validation/'

# Partitioning ratios

tr_ratio = 0.4
val_ratio = 0.3
tst_ratio = 1-tr_ratio-val_ratio

# General utility functions.

def delete_figures(path):
    """Delete all images in the directory specified by path"""
    for root, dirs, files in os.walk(path, topdown=False):
        for currentFile in files:    
            exts = ('.png', '.jpg')
            if any(currentFile.lower().endswith(ext) for ext in exts):
                os.remove(os.path.join(root, currentFile))

# Mathematical functions.

def autocorr(x, length=10, graph=False):
	"""Calculates the autocorrelation of a vector for a lag of t"""
	result = np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1] for i in range(1, length)])
	if graph == True:
		file_name = raw_input('Input file name for graph: ')
		fig = plt.figure()
		plt.stem(result)
		fig.savefig(file_name+'.png', format='png', dpi=500, bbox_inches='tight')
	return result
