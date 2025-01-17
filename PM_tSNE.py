"""
---
This code implements a Particle-Mesh inspired code for t-SNE

The Particle-Mesh algorithm (PM) is well used in physics to solve
the N-body problem. It is based on the ability of FFTs to transform
convolution products in simple products. The main idea is to compute
a potential, and get back the forces by deriving it.

This work was carried out as part of my master's thesis at the
University of Namur (Belgium, 2020), titled: "Contributions of
Physics in Machine Learning: accelerate the t-SNE algorithm using
the Fourier Transforms and the Particle-Mesh algorithm."
---

---
Made with Python 3.6.9
Check the requirements.txt file for python requirements 
---

---
Detailled structure

.
├── PM_tSNE.py
│   ├── def _joint_probabilities_nn
│   └── class PM_tSNE
│       ├── def __init__
│       └── def fit_transform
│
├── _perform_GD.pyx
│   ├── cdef _compute_Attr
│   ├── cdef _kernel
│   ├── cdef _compute_Repu_NGP
│   ├── cdef _compute_Repu_CIC
│   ├── cdef _gradientDescent
│   └── cpdef gradientDescent
│
└── _utils.pyx
    └── cpdef _binary_search_perplexity
---

---
           Author: Delchevalerie Valentin
            email: valentin.delchevalerie@unamur.be
last modification: 12 September 2020
---
"""


import numpy as np
import cython
from sklearn.neighbors import NearestNeighbors
import annoy
from scipy.sparse import csr_matrix
import _utils
import _perform_GD
import time


MACHINE_EPSILON = np.finfo(np.double).eps


def joint_probabilities_nn(distances, desired_perplexity):
    """
    Computes joint probabilities p_ij from distances using just nearest
    neighbors.

    ----------
    Parameters
    ----------
    * distances : CSR sparse matrix, shape (n_samples, n_samples)
        Distances of samples to its n_neighbors nearest neighbors. All other
        distances are left to zero (and are not materialized in memory).
        
    * desired_perplexity : float
        Desired perplexity of the joint probability distributions.
        
    -------
    Returns
    -------
    * P : csr sparse matrix, shape (n_samples, n_samples)
        Condensed joint probability matrix with only nearest neighbors.
    """
    
    distances.sort_indices()
    n_instances = distances.shape[0]
    distances_data = distances.data.reshape(n_instances, -1)
    distances_data = distances_data.astype(np.float32, copy=False)
    conditional_P = _utils._binary_search_perplexity(distances_data, desired_perplexity, verbose=0)
    
    # Symmetrize the joint probability matrix using sparse operations
    P = csr_matrix((conditional_P.ravel(), distances.indices, distances.indptr), shape=(n_instances, 
                                                                                        n_instances))
    P = P + P.T
    
    # Normalize the joint probability matrix
    sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
    P /= sum_P

    return P.data, P.indices, P.indptr


class PM_tSNE:
    """
    Particle-Mesh based t-distributed Stochastic Neighbor Embedding
    
    t-SNE is a tool to visualize high-dimensional data.
    It is a nonlinear dimensionality reduction technique well-suited
    for embedding high-dimensional data for visualization in a
    low-dimensional space of two or three dimensions.
    
    However, this algorithm show a quadratic numerical complexity
    with N, the number of instances. This drawback motivated this
    work. PM_tSNE show a numerical complexity O(M log M), with M
    the number of grid points.
    """
    
    
    def __init__(self, perplexity=30, coeff=6, grid_meth='NGP', eta=750., early_ex=12., 
                 initial_mom=0.5, final_mom=0.8, min_gain=0.01, k_factor=3.0, exact_nn=False, 
                 n_trees=20, stop_early=100, n_iter=750):
        """
        ----------
        Parameters
        ----------
        * perplexity : double
            Perplexity used for t-SNE.
        
        * coeff : double
            This coefficient is used to define the resolution of the grid as int(pow(2.0, coeff)).
            
        * grid_meth : string
            Interpolation method to use on the grid ['NGP' or 'CIC'].
            
        * eta : double
            Learning rate.
            
        * early_ex : double
            Early exaggeration factor to use.
            
        * initial_mom : double
            Initial momentum to use.
            
        * final_mom : double
            Final momentum to use.
            
        * min_gain : double
            Minimum gain.
            
        * k_factor : double
            Used to determine the number of nearest neighbors to consider w.r.t.
            k = min(n_instances - 1, np.int(k_factor * perplexity + 1)).
            
        * exact_nn : Boolean
            If True, exact nearest neighbors search is used (not recommanded if n_instances >> 1).
            If False, approximate nearest neighbors search is used, with the Annoy library.
            
        * n_trees : integer
            Meta-parameters for the approximate nearest neighbors search (Annoy).
            
        * stop_early : integer
            Number of iterations before stopping early exaggeration.
            
        * n_iter : integer
            Number of iterations.
            
        -------
        Returns
        -------
        * Y : Array of double, of shape [n_instances, 2]
            Embedding.
        """

        self.perplexity = perplexity
        self.coeff = coeff
        self.grid_meth = grid_meth
        self.eta = eta
        self.early_ex = early_ex
        self.initial_mom = initial_mom
        self.final_mom = final_mom
        self.min_gain = min_gain
        self.k_factor = k_factor
        self.exact_nn = exact_nn
        self.n_trees = n_trees
        self.stop_early = stop_early
        self.n_iter = n_iter


    def fit_transform(self, X):
        """
        Compute the embedding

        ----------
        Parameters
        ----------
        * X : Array of double, of shape [n_instances, n_features]
            High dimensional data.
            
        -------
        Returns
        -------
        * Y : Array of double, of shape [n_instances, 2]
            Embedding.
        """
        
        X = np.asarray(X)
        n_instances = X.shape[0]
        initial_dims = X.shape[1]

        # Check for invalid parameters
        if (self.grid_meth not in ['NGP', 'CIC']):
            raise ValueError("'grid_meth' must be 'NGP' or 'CIC'")

        if (self.early_ex < 1.0):
            raise ValueError("early_exaggeration must be at least 1, but is {}".format(self.early_ex))

        if (self.n_iter < self.stop_early):
            raise ValueError("'n_iter' should be greater than 'stop_early'")

        if (self.exact_nn not in [True, False]):
            raise ValueError("'exact_nn' should be True or False")

        k = min(n_instances - 1, np.int(self.k_factor * self.perplexity + 1))
        
        if (self.exact_nn):
            knn = NearestNeighbors(algorithm='auto', n_neighbors=k, metric='euclidean')
            knn.fit(X)
            distances_nn = knn.kneighbors_graph(mode='distance')
            del knn ; del X
        else:
            neighbors = np.empty((n_instances, k+1), dtype=np.int)
            distances = np.empty((n_instances, k+1), dtype=np.float)
            
            t = annoy.AnnoyIndex(initial_dims, 'euclidean')
            for i in range(n_instances):
                t.add_item(i, X[i,:].tolist())
            t.build(self.n_trees)
            
            for i in range(n_instances):
                neighbors[i, :], distances[i, :] = t.get_nns_by_vector(X[i,:].tolist(), k+1, 
                                                                       include_distances=True)
            
            indptr = np.arange(0, n_instances * (k+1) + 1, k+1)
            distances_nn = csr_matrix((distances.ravel(), neighbors.ravel(), indptr), shape=(n_instances, 
                                                                                             n_instances))
            del indptr ; del neighbors ; del distances ; del t ; del X

        data, indices, indptr = joint_probabilities_nn(distances_nn, self.perplexity)
        
        Y = _perform_GD.gradientDescent(n_instances, data, indices, indptr, self.coeff, self.grid_meth.encode('utf-8'), 
                                        self.eta, self.early_ex, self.initial_mom, self.final_mom, self.min_gain, 
                                        self.stop_early, self.n_iter)
        
        return Y
