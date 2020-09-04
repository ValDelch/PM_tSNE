# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=True

"""
---
More informations in PM_tSNE.py
---

---
           Author: Delchevalerie Valentin
            email: valentin.delchevalerie@unamur.be
last modification: 04 September 2020
---
"""


from sklearn.neighbors import NearestNeighbors
from libc.math cimport sqrt, pow


cpdef double euclidian_distance(double[:] x, double[:] y):
    """
    This function returns the euclidean distance between two points
    of an n dimensional space
    
    ----------
    Parameters
    ----------
    * x : 
        First point coordinates.
    * y :
        Second point coordinates.
    
    -------
    Returns
    -------
    * dist : double
        Euclidean distance between x and y.
    """
    
    cdef:
        int n = x.shape[0]
        int i
        double dist = 0
    
    for i in range(n):
        dist += pow(x[i] - y[i], 2.0)
        
    return sqrt(dist)
    
    
cpdef getExactDistances(double[:,:] X, int k):
    """
    This function returns the euclidean distances between each points
    and his k nearest neighbors
    
    ----------
    Parameters
    ----------
    * X :
    
    * k : integer
        Number of nearest neighbors to take into account.
    
    -------
    Returns
    -------
    * distances_nn :
        Euclidean distances between each points and his k nearest
        neighbors.
    """

    cdef:
        object knn

    knn = NearestNeighbors(algorithm='auto', n_neighbors=k, metric=euclidian_distance)
    knn.fit(X)
    distances_nn = knn.kneighbors_graph(mode='distance')
    
    return distances_nn
    
    
    
    
    
    
    
    
