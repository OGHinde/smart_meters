# -*- coding: utf-8 -*-

'''
Path variables and other general resources.

Modify this module with your relevant local paths.
'''

# Import useful modules
import os

# General data paths.
smartMeters_raw_csv = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/data/extraccionLimpios.csv'
smartMeters_prep_csv = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/data/extraccion_prep.csv'

clustering_matrix_D = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/data/clustering_matrix_D.pickle'
clustering_matrix_8h = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/data/clustering_matrix_8h.pickle'
clustering_matrix_6h = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/data/clustering_matrix_6h.pickle'


base_matrix_D = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/data/base_matrix_D.pickle'
base_matrix_8h = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/data/base_matrix_8h.pickle'
base_matrix_6h = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/data/base_matrix_6h.pickle'
base_matrix_1h = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/data/base_matrix_1h.pickle'

clustering_dframe = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/data/clustering_dataframe.csv'
general_data = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/data/general_data.pickle'

kmeans_figs = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/figures/Kmeans/'
kmeans_log = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/figures/Kmeans/K_means_results.txt'
gmm_figs = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/figures/GMM/'
gmm_log = '/Users/oghinde/Academia/Research_projects/smart_grids/Iberdrola/figures/Kmeans/GMM_results.txt'

# General functions.
def delete_figures(path):
    """Delete all images in the directory specified by path"""
    for root, dirs, files in os.walk(path, topdown=False):
        for currentFile in files:    
            exts = ('.png', '.jpg')
            if any(currentFile.lower().endswith(ext) for ext in exts):
                os.remove(os.path.join(root, currentFile))

