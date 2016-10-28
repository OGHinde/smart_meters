#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 14:33:28 2016

@author: oghinde
"""


import numpy as np
import pickle
import resources
import matplotlib

# Read data
val_matrix = np.zeros((4, 9))
for x, i in enumerate(range(2, 6)):
    for y, j in enumerate(range(2, 11)):
        thing =  'val_{}_{}.pickle'.format(i, j)
        with open(resources.val_results_path+thing, 'r') as f:
            avg_val, avg_series = pickle.load(f)
            val_matrix[x, y] = avg_val

fig = matplotlib.pyplot.figure(1)
# Define colormap
cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('my_colormap',
                                                            ['black','orange','red'],
                                                            512)
# Draw matrix
img2 = matplotlib.pyplot.imshow(val_matrix,
                                interpolation='nearest',
                                cmap = cmap2,
                                origin='upper', aspect='auto')
# Draw colorbar
matplotlib.pyplot.colorbar(img2,cmap=cmap2)
fig.savefig("/Users/oghinde/Desktop/validation_map.png", bbox_inches='tight')


max_acc = np.max(val_matrix)
max_args = np.array(np.where(val_matrix == np.max(val_matrix)))+2
print 'RESULTS:\n'
print 'The maximum validation accuracy is: {}'.format(max_acc)
print 'The optimal cluster sizes are {} week clusters and {} user clusters.'.format(max_args[0][0], max_args[1][0])