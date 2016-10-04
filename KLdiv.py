'''
# ###########################################################################
# ### KLdiv estima la divergencia de Kullback-Leibler entre dos
# ###         distribuciones a partir de una coleccion de observaciones
# ###########################################################################
'''

import numpy as np
from sklearn.neighbors import NearestNeighbors


def ComputeDKL(X0, X1, k=1):

    """
    function DKL = ComputeDKL(X0, X1, alpha, mode) estimates the KL
        divergence among two distributions P0 and P1 based on samples
        INPUTS:
            X0    :Samples from distribution P0
            X1    :Samples from distribution P1
            k     :No. of neighbours    
        OUTPUTS   :DKL.
    """

    Nx, Dim = X0.shape
    eps = np.finfo(float).eps

    # Create knn objects, one for each distribution.
    knn0 = NearestNeighbors(k)
    knn1 = NearestNeighbors(k)

    # Load knn's with data samples.
    knn0.fit(X0)
    knn1.fit(X1)

    # Compute k-nn distances.
    # Note that, to dompute d0, we state n_neighbors = k+1. This is because
    # every sample is its own nearest neighbor, and it should be excluded
    # from the analysis.
    d0, ind0 = knn0.kneighbors(X0, n_neighbors=k+1, return_distance=True)
    d1, ind1 = knn1.kneighbors(X0, n_neighbors=k, return_distance=True)

    # Estimate KL div from distances.
    d0k = d0.T[k]
    d1k = d1.T[k-1]
    LogDR = Dim*np.log((d1k + eps)/(d0k + eps))

    # ### Compute divergence estimate and its variance
    DKL = np.mean(LogDR, axis=0) + np.log(Nx/(Nx - 1))

    return DKL