# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=True

"""
This part of the code impement the gradient descent used for PM_tSNE.

More informations in PM_tSNE.py
"""


import numpy as np
cimport numpy as np
from libc.math cimport pow


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef double[:,:] _compute_Attr(double[:,:] Y, double[:] data, int[:] indices, int[:] indptr, int n, 
                               int no_dims):
    """
    This function compute the attractive term with respect to the k nearest neighbors
    
    ----------
    Parameters
    ----------
    * Y : 
        Blabla.
        
    * data :
        Balbla.
        
    * indices :
        Blabla.
        
    * indptr :
        Blabla.
        
    * n :
        Blabla.
        
    * no_dims :
        Blabla.
        
    -------
    Returns
    -------
    * dY : 
        Blabla.
    """
    
    cdef:
        int i, j, _i, _j
        double qz, diff1, diff2
        double[:,:] dY = np.zeros((n, no_dims), dtype=DTYPE)
        
    for i in range(n):
        _i = indptr[i]
        _j = indptr[i+1]
        
        for j in range(_i, _j):
            diff1 = Y[i,0] - Y[indices[j],0]
            diff2 = Y[i,1] - Y[indices[j],1]
            qz = 4.0 / (1.0 + pow(diff1, 2.0)
                            + pow(diff2, 2.0))
            
            dY[i,0] += data[j] * qz * diff1
            dY[i,1] += data[j] * qz * diff2
            
    return dY


cdef double[:,:] _gradientDescent(int n, int no_dims, double[:] data, int[:] indices, int[:] indptr, 
                                  double coeff, char* grid_meth, double eta, double early_ex, 
                                  double initial_mom, double final_mom, double min_gain, int stop_early, 
                                  int n_iter):
    """
    Blabla
    """

    cdef:
        int i, j, l
        double momentum
        double[:] sum_res = np.zeros((no_dims), dtype=DTYPE)
        double[:,:] dY
        double[:,:] Y     = np.random.randn(n, no_dims).astype(dtype=DTYPE) * 1e-4
        double[:,:] gains = np.zeros((n, no_dims), dtype=DTYPE)
        double[:,:] iY    = np.zeros((n, no_dims), dtype=DTYPE)
        
    # Early exaggeration
    for i in range(data.shape[0]):
        data[i] *= early_ex
        
    # Main loop
    for l in range(n_iter):
        if l < 20:
            momentum = initial_mom
        else:
            momentum = final_mom

        dY = _compute_Attr(Y, data, indices, indptr, n, no_dims)
        
        sum_res[:] = 0.0
        for i in range(n):
            for j in range(no_dims):
                if dY[i,j] > 0.0 != iY[i,j] > 0.0:
                    gains[i,j] += 0.2
                else:
                    gains[i,j] *= 0.8
                    
                if gains[i,j] < min_gain:
                    gains[i,j] = min_gain
                    
                iY[i,j] = momentum * iY[i,j] - eta * gains[i,j] * dY[i,j]
                Y[i,j] += iY[i,j]
                sum_res[j] += Y[i,j]
                
        for i in range(n):
            for j in range(no_dims):
                Y[i,j] -= sum_res[j]/n
                
        if l == stop_early:
            for i in range(data.shape[0]):
                data[i] /= early_ex
                
    return Y


cpdef gradientDescent(int n, int no_dims, double[:] data, int[:] indices, int[:] indptr, double coeff, 
                      char* grid_meth, double eta, double early_ex, double initial_mom, double final_mom, 
                      double min_gain, int stop_early, int n_iter):
    """
    Blabla
    """
    
    return _gradientDescent(n, no_dims, data, indices, indptr, coeff, grid_meth, eta, early_ex, initial_mom, 
                            final_mom, min_gain, stop_early, n_iter)
