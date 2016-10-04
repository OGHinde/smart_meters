# -*- coding: utf-8 -*-

'''
Path variables and other general resources.

Modify this module with your relevant local paths.
'''

# Import useful modules
import os

# General data paths.
smartMeters_raw_csv = ''
smartMeters_prep_csv = ''

clustering_matrix_D = ''
clustering_matrix_8h = ''
clustering_matrix_6h = ''


base_matrix_D = ''
base_matrix_8h = ''
base_matrix_6h = ''
base_matrix_1h = ''

clustering_dframe = ''
general_data = ''

kmeans_figs = ''
kmeans_log = ''
gmm_figs = ''
gmm_log = ''

# General functions.
def delete_figures(path):
    """Delete all images in the directory specified by path"""
    for root, dirs, files in os.walk(path, topdown=False):
        for currentFile in files:    
            exts = ('.png', '.jpg')
            if any(currentFile.lower().endswith(ext) for ext in exts):
                os.remove(os.path.join(root, currentFile))

